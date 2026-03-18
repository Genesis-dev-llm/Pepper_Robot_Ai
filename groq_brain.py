"""
Groq Brain — LLM and Speech-to-Text

Changes from previous version:
- GESTURE_NAMES imported from config (single source of truth)
- Removed compound_model / web-search special path — tools are always passed,
  including web_search when USE_WEB_SEARCH is True in config.
- History trim now preserves the first user/assistant exchange (intro context)
  and removes from the middle when the window fills.
- _sanitize_history strips stale tool_call metadata from assistant turns.
- _clean_response_text now uses a permissive regex for function tag stripping
  that handles malformed tags where the model omits the closing '>' on the
  opening tag (e.g. <function=express_emotion{"emotion":"curious"}</function>).
- chat_with_context retries without tools on Groq 400 (malformed tool call).

Fix (history read outside lock):
- conversation_history is a mutable list shared across threads.  Both chat()
  and chat_with_context() previously read it without holding _history_lock,
  meaning a concurrent _trim_history() or history append could produce a
  stale or torn slice.  Both methods now snapshot the history under the lock
  before building the messages list.
"""

import json
import re
import threading
from typing import Callable, Dict, List, Optional, Tuple

from groq import Groq

import config


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_tool_calls(message) -> List[Dict]:
    if not (hasattr(message, "tool_calls") and message.tool_calls):
        return []
    return [
        {
            "name": tc.function.name,
            "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
        }
        for tc in message.tool_calls
    ]


def _clean_response_text(text: str) -> str:
    """
    Strip function-call artifacts and stage directions from spoken output.

    Two failure modes the model can produce:

      Well-formed:  <function=wave>...</function>
      Malformed:    <function=express_emotion{"emotion":"curious"}</function>
                    (no '>' closing the opening tag)

    The original regex <function=[^>]*>.*?</function> requires a '>' to close
    the opening tag and silently skips the malformed variant.  The replacement
    regex <function=.*?</function> is permissive — it matches from '<function='
    through the nearest '</function>' regardless of what's between them.

    A second pass catches any unclosed bare <function= fragments that have no
    matching </function> at all.
    """
    # Permissive: catches both well-formed and malformed opening tags
    text = re.sub(r"<function=.*?</function>", "", text, flags=re.DOTALL)
    # Catch unclosed / self-closing bare <function= tags (no </function>)
    text = re.sub(r"<function=[^<\n]*", "", text)
    # XML-style tool tags
    text = re.sub(r"<tool[^>]*>.*?</tool>", "", text, flags=re.DOTALL)
    # Asterisk stage directions
    text = re.sub(r"\*[^*]+\*", "", text)
    # Strip "call gesture_name()" / "call gesture_name(...)" lines the model
    # sometimes emits as plain text instead of using the tool call API.
    text = re.sub(r"(?i)^\s*call\s+\w+\([^)]*\)\s*$", "", text, flags=re.MULTILINE)
    # Strip (stage directions in parentheses) — model sometimes uses these instead of asterisks
    text = re.sub(r"\([^)]{1,60}\)", "", text)
    # Strip <web_search> XML tags the model emits when it can't use the tool API
    # e.g. <web_search> {"query": "..."} </web_search>
    text = re.sub(r"<web_search>.*?</web_search>", "", text, flags=re.DOTALL)
    # Also strip unclosed <web_search> fragments
    text = re.sub(r"<web_search>[^<]*", "", text)
    # Bare gesture names on their own line (derived from config — single source of truth)
    lines = text.splitlines()
    lines = [l for l in lines if l.strip().lower() not in config.GESTURE_NAMES]
    return "\n".join(lines).strip()


def _extract_web_search_tag(text: str) -> Optional[str]:
    """
    Extract a search query from a <web_search> tag the model emitted as plain text.
    Handles both {"query": "..."} and {"search_term": "..."} key names.
    e.g. <web_search> {"query": "news March 13 2026"} </web_search>
    """
    match = re.search(r'<web_search[^>]*>(.*?)</web_search>', text, re.DOTALL)
    if not match:
        match = re.search(r'<web_search[^>]*>([^<]+)', text)
    if not match:
        return None
    body = match.group(1).strip()
    # Try to parse JSON body
    q = re.search(r'"(?:query|search_term|q)"\s*:\s*"([^"]+)"', body)
    if q:
        return q.group(1)
    # Fallback: body itself might just be the query string
    if body and not body.startswith("{"):
        return body.strip()
    return None


def _extract_query_from_400(error_str: str) -> Optional[str]:
    """
    Parse the search query out of a Groq 400 error where the model malformed
    the tool name, e.g.: attempted to call tool 'web_search={"query": "..."}'

    Returns the query string if found, None otherwise.
    """
    # Pattern: web_search={"query": "..."} or web_search{"query": "..."}
    match = re.search(r'web_search[=\s]*\{[^}]*"query"\s*:\s*"([^"]+)"', error_str)
    if match:
        return match.group(1)
    # Fallback: <function=web_search={"query": "..."}>
    match = re.search(r'<function=web_search[^>]*"query"\s*:\s*"([^"]+)"', error_str)
    if match:
        return match.group(1)
    return None


def _sanitize_history(history: List[Dict]) -> List[Dict]:
    """Strip tool_calls metadata from assistant messages to avoid stale objects."""
    return [
        {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("role") == "assistant"
        else msg
        for msg in history
    ]


# ── GroqBrain ──────────────────────────────────────────────────────────────────

class GroqBrain:
    def __init__(
        self,
        api_key:        str,
        llm_model:      str,
        whisper_model:  str,
        system_prompt:  str,
        functions:      List[Dict],
        use_web_search: bool = False,
        fallback_models: Optional[List[str]] = None,
    ):
        self.client         = Groq(api_key=api_key)
        self.whisper_model  = whisper_model
        self.use_web_search = use_web_search

        # Full model chain: primary + fallbacks
        self._model_chain  = [llm_model] + (fallback_models or [])
        self._model_index  = 0
        self.llm_model     = self._model_chain[0]

        # gesture-only functions (no web_search) — used for compound models
        self._gesture_functions = [
            f for f in functions
            if f.get("function", {}).get("name") != "web_search"
        ]
        # full functions including web_search — used for llama fallback
        self._all_functions = functions

        # Build system prompt based on current model
        self.system_prompt = config.build_system_prompt(
            native_search=self._is_compound()
        )

        self.conversation_history: List[Dict] = []
        self._history_lock = threading.Lock()

        print(f"🧠 Groq Brain ready — {self.llm_model} "
              f"({'native search' if self._is_compound() else 'web search ON' if use_web_search else 'web search OFF'})")
        if self._model_chain[1:]:
            print(f"   Fallback chain: {' → '.join(self._model_chain[1:])}")

    # ── Model fallback ────────────────────────────────────────────────────────

    def _is_compound(self, model: Optional[str] = None) -> bool:
        """Returns True if the given (or current) model has native built-in search."""
        return (model or self.llm_model) in config.COMPOUND_MODELS

    @property
    def functions(self) -> List[Dict]:
        """
        Return appropriate tool list for the current model.
        Compound: gesture tools only — it has native search built in, so passing
          our web_search definition causes a conflict (two search tools) and 400s.
        Llama fallback: gesture tools + web_search as normal.
        """
        if self._is_compound():
            return self._gesture_functions  # gestures only, no web_search conflict
        return self._all_functions if self.use_web_search else self._gesture_functions

    def _advance_model(self) -> bool:
        """
        Step to the next model in the fallback chain.
        Returns True if a fallback is available, False if chain is exhausted.
        """
        if self._model_index >= len(self._model_chain) - 1:
            print("❌ All models in fallback chain exhausted")
            return False
        self._model_index += 1
        self.llm_model = self._model_chain[self._model_index]
        # Rebuild system prompt for new model type
        self.system_prompt = config.build_system_prompt(
            native_search=self._is_compound()
        )
        print(f"⚠️  Falling back to {self.llm_model} "
              f"({'native search' if self._is_compound() else 'manual search'})")
        return True

    # ── STT ───────────────────────────────────────────────────────────────────

    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        try:
            with open(audio_file_path, "rb") as f:
                result = self.client.audio.transcriptions.create(
                    file=f,
                    model=self.whisper_model,
                    response_format="text",
                    language="en",
                )
            return result.strip() or None
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """
        Standard chat turn. Tools (gestures + optional web_search) are always
        passed — no special-case model switching.

        If the model returns only tool calls with no text, a follow-up call
        is made without tools to get the spoken response.
        """
        try:
            # Snapshot history under lock before building the messages list.
            # Without the lock another thread could be mid-trim or mid-append,
            # producing a stale or torn slice that silently corrupts context.
            with self._history_lock:
                history_snapshot = list(self.conversation_history)

            messages = (
                [{"role": "system", "content": self.system_prompt}]
                + _sanitize_history(history_snapshot)
                + [{"role": "user", "content": user_message}]
            )

            # Try current model, fall back on 429
            response = None
            for _attempt in range(len(self._model_chain)):
                try:
                    _tools = self.functions
                    _kwargs = dict(
                        model       = self.llm_model,
                        messages    = messages,
                        temperature = 0.7,
                        max_tokens  = 400,
                    )
                    if _tools:
                        _kwargs["tools"]       = _tools
                        _kwargs["tool_choice"] = "auto"
                    response = self.client.chat.completions.create(**_kwargs)
                    break  # success
                except Exception as _e:
                    if ("429" in str(_e) or "rate_limit" in str(_e).lower()) and self._advance_model():
                        continue  # retry with next model
                    raise  # non-429 or chain exhausted — let outer handler deal with it
            if response is None:
                return None, None

            message        = response.choices[0].message
            raw_text       = message.content or ""
            response_text  = _clean_response_text(raw_text)
            function_calls = _parse_tool_calls(message)

            # Model output <web_search> as plain text with no tool call
            if not function_calls and not response_text and self.use_web_search:
                search_query = _extract_web_search_tag(raw_text)
                if search_query:
                    print(f"⚠️  <web_search> tag in response — re-routing: '{search_query}'")
                    return None, [{"name": "web_search", "arguments": {"query": search_query}}]

            if function_calls and not response_text:
                response_text = self._get_verbal_response(messages, function_calls)

            if response_text:
                with self._history_lock:
                    self.conversation_history.append({"role": "user",      "content": user_message})
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                    self._trim_history()

            self._log(response_text, function_calls)
            return response_text, function_calls or None

        except Exception as e:
            # Groq 400: model malformed the web_search tool call, e.g.
            # <function=web_search={"query": "..."}> — the query is embedded
            # in the tool name. Extract it and do the search properly.
            if "400" in str(e) or "tool_use_failed" in str(e) or "tool call validation" in str(e):
                query = _extract_query_from_400(str(e))
                if query and self.use_web_search:
                    print(f"⚠️  Malformed search tool call — extracting query: '{query}'")
                    return None, [{"name": "web_search", "arguments": {"query": query}}]
                # No query extractable — retry without tools
                print(f"⚠️  Tool call malformed in chat() — retrying without tools")
                try:
                    response = self.client.chat.completions.create(
                        model       = self.llm_model,
                        messages    = messages,
                        temperature = 0.7,
                        max_tokens  = 400,
                    )
                    raw_text      = response.choices[0].message.content or ""
                    response_text = _clean_response_text(raw_text)

                    if not response_text and self.use_web_search:
                        search_query = _extract_web_search_tag(raw_text)
                        if search_query:
                            print(f"⚠️  <web_search> tag in chat retry — re-routing: '{search_query}'")
                            return None, [{"name": "web_search", "arguments": {"query": search_query}}]

                    if response_text:
                        with self._history_lock:
                            self.conversation_history.append({"role": "user",      "content": user_message})
                            self.conversation_history.append({"role": "assistant", "content": response_text})
                            self._trim_history()
                        self._log(response_text, [], tag="retry")
                    return response_text, None
                except Exception as e2:
                    print(f"❌ Chat retry failed: {e2}")
                    return None, None
            print(f"❌ Chat error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def chat_with_context(
        self,
        user_message: str,
        context: str,
    ) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """
        Single LLM call with injected search/tool context.

        If Groq rejects with a 400 (malformed tool call in model output),
        retries once without tools to still get a spoken response.
        """
        # Snapshot history under lock before building the messages list.
        # Same reasoning as chat() — concurrent trim/append without the lock
        # can produce a stale or torn slice sent to the API.
        with self._history_lock:
            history_snapshot = list(self.conversation_history)

        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + _sanitize_history(history_snapshot)
            + [{"role": "system", "content":
                "[Search result — use this to answer; do NOT repeat verbatim]:\n" + context}]
            + [{"role": "user", "content": user_message}]
        )

        # Exclude web_search from tools — results are already injected above.
        # This prevents the model malforming a search tool call on the context call.
        gesture_tools = [f for f in self.functions
                         if f.get("function", {}).get("name") != "web_search"]

        # Try current model, fall back on 429
        response = None
        for _attempt in range(len(self._model_chain)):
            try:
                _kwargs = dict(
                    model       = self.llm_model,
                    messages    = messages,
                    temperature = 0.7,
                    max_tokens  = 400,
                )
                if gesture_tools:
                    _kwargs["tools"]       = gesture_tools
                    _kwargs["tool_choice"] = "auto"
                response = self.client.chat.completions.create(**_kwargs)
                break
            except Exception as _e:
                if ("429" in str(_e) or "rate_limit" in str(_e).lower()) and self._advance_model():
                    continue
                raise
        try:
            if response is None:
                return None, None

            message        = response.choices[0].message
            response_text  = _clean_response_text(message.content or "")
            function_calls = _parse_tool_calls(message)

            if function_calls and not response_text:
                response_text = self._get_verbal_response(messages, function_calls)

            if response_text:
                with self._history_lock:
                    self.conversation_history.append({"role": "user",      "content": user_message})
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                    self._trim_history()

            self._log(response_text, function_calls, tag="ctx")
            return response_text, function_calls or None

        except Exception as e:
            # Groq 400: model generated a malformed tool call (e.g. express_emotion{...}
            # embedded in the tool name). Retry without tools to still get a response.
            if "400" in str(e) or "tool_use_failed" in str(e) or "tool call validation" in str(e):
                print(f"⚠️  Tool call malformed — retrying without tools")
                try:
                    response = self.client.chat.completions.create(
                        model       = self.llm_model,
                        messages    = messages,
                        temperature = 0.7,
                        max_tokens  = 400,
                    )
                    raw_text      = response.choices[0].message.content or ""
                    response_text = _clean_response_text(raw_text)

                    # Model output a <web_search> tag as plain text instead of a tool call.
                    # Extract the query and hand it back as a tool call so main.py re-searches.
                    if not response_text and self.use_web_search:
                        search_query = _extract_web_search_tag(raw_text)
                        if search_query:
                            print(f"⚠️  <web_search> tag in ctx-retry — re-routing as tool call: '{search_query}'")
                            return None, [{"name": "web_search", "arguments": {"query": search_query}}]

                    if response_text:
                        with self._history_lock:
                            self.conversation_history.append({"role": "user",      "content": user_message})
                            self.conversation_history.append({"role": "assistant", "content": response_text})
                            self._trim_history()
                        self._log(response_text, [], tag="ctx-retry")
                    return response_text, None
                except Exception as e2:
                    print(f"❌ chat_with_context retry failed: {e2}")
                    return None, None

            print(f"❌ chat_with_context error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _get_verbal_response(
        self,
        prior_messages: List[Dict],
        function_calls: List[Dict],
    ) -> str:
        """
        Follow-up call when model returned only tool calls with no text.

        If the follow-up call also returns nothing (compound models occasionally
        return a second tool-only response), fall back to a minimal spoken
        acknowledgement so Pepper always says something rather than going
        silently blank after performing a gesture.
        """
        try:
            gesture_names = [f["name"] for f in function_calls if f["name"] != "web_search"]
            context = f"[You already chose to perform: {', '.join(gesture_names)}] " if gesture_names else ""

            response = self.client.chat.completions.create(
                model      = self.llm_model,
                messages   = prior_messages + [{
                    "role":    "system",
                    "content": context + "Now provide your spoken response. Keep it 1-3 sentences.",
                }],
                temperature = 0.7,
                max_tokens  = 400,
            )
            text = _clean_response_text(response.choices[0].message.content or "")
            if text:
                print("🔄 Follow-up verbal response obtained")
                return text
            # Model returned empty text again (e.g. compound returned another
            # tool-only response).  Return a minimal fallback so TTS has
            # something to say rather than silently dropping the turn.
            print("⚠️  Follow-up verbal response empty — using fallback")
            return "Sure."
        except Exception as e:
            print(f"⚠️  Follow-up verbal response failed: {e}")
            return "Sure."

    # ── Search intent detection ────────────────────────────────────────────────

    def needs_search(self, message: str) -> bool:
        lowered = message.lower()
        return any(kw in lowered for kw in config.SEARCH_KEYWORDS)

    # ── Conversation management ────────────────────────────────────────────────

    def reset_conversation(self):
        with self._history_lock:
            self.conversation_history = []
        print("🔄 Conversation history cleared")

    def _trim_history(self):
        """
        Keep at most MAX_HISTORY turns (pairs).
        Always preserve the first user/assistant exchange for intro context.
        When over limit, remove from the middle (turns 2 & 3), not from turn 0.

        Must be called with _history_lock already held.
        """
        cap = config.MAX_HISTORY * 2
        hist = self.conversation_history
        if len(hist) <= cap:
            return
        # Keep first 2 entries (turn 0) + most recent (cap - 2) entries
        self.conversation_history = hist[:2] + hist[-(cap - 2):]

    def _log(self, text: Optional[str], calls: List[Dict], tag: str = ""):
        label = f"💬 AI{f' ({tag})' if tag else ''}"
        if text:
            print(f"{label}: {text[:100]}{'…' if len(text or '') > 100 else ''}")
        if calls:
            print(f"🎬 Functions: {[f['name'] for f in calls]}")


# ── Connectivity test ──────────────────────────────────────────────────────────

def test_groq_connection(api_key: str) -> bool:
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [{"role": "user", "content": "Say 'hello' in one word"}],
            max_tokens = 10,
        )
        print(f"✅ Groq API OK: {resp.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Groq API failed: {e}")
        return False