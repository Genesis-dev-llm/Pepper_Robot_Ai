"""
Groq Brain - LLM and Speech-to-Text

Key design decisions:
- chat()               — standard turn; if the model returns only tool calls
                         with no text, a follow-up call is made automatically
                         to get the verbal response, so the robot always speaks.
- chat_with_context()  — self-contained; injects search/tool results as a system
                         message for a single LLM call.
- needs_search()       — fast keyword heuristic; only active when USE_WEB_SEARCH
                         is True in config.
"""

import json
import re
import threading
from typing import Callable, Dict, List, Optional, Tuple

from groq import Groq

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tool_calls(message) -> List[Dict]:
    calls = []
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            calls.append({
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments)
                             if tc.function.arguments else {},
            })
    return calls


def _clean_response_text(text: str) -> str:
    """
    Strip any function/tool call artifacts or stage directions the model
    leaked into its spoken text.

    Patterns removed:
    - <function=name></function> and variants (XML-style tool calls)
    - *action* asterisk stage directions (e.g. *nods*, *pauses*, *waves*)
    - Bare gesture names on their own line (e.g. "Watch this.\nwave")
    """
    # XML-style function tags
    text = re.sub(r"<function=[^>]*>.*?</function>", "", text, flags=re.DOTALL)
    text = re.sub(r"<function=[^>/]*/?>", "", text)
    text = re.sub(r"<tool[^>]*>.*?</tool>", "", text, flags=re.DOTALL)
    # Asterisk stage directions — *word* or *multiple words*
    text = re.sub(r"\*[^*]+\*", "", text)
    # Bare gesture/function names on their own line
    _GESTURE_NAMES = {
        "wave", "nod", "shake_head", "thinking_gesture", "explaining_gesture",
        "excited_gesture", "point_forward", "shrug", "celebrate", "look_around",
        "bow", "look_at_sound", "web_search",
    }
    lines = text.splitlines()
    lines = [l for l in lines if l.strip().lower() not in _GESTURE_NAMES]
    return "\n".join(lines).strip()


def _sanitize_history(history: List[Dict]) -> List[Dict]:
    """
    Strip tool_calls metadata from assistant messages so the API never
    sees stale/malformed tool call objects from prior turns.
    """
    clean = []
    for msg in history:
        if msg.get("role") == "assistant":
            clean.append({"role": "assistant", "content": msg.get("content") or ""})
        else:
            clean.append(msg)
    return clean


# ---------------------------------------------------------------------------
# GroqBrain
# ---------------------------------------------------------------------------

class GroqBrain:
    def __init__(
        self,
        api_key:        str,
        llm_model:      str,
        whisper_model:  str,
        system_prompt:  str,
        functions:      List[Dict],
        use_web_search: bool = False,
        compound_model: Optional[str] = None,
    ):
        self.client         = Groq(api_key=api_key)
        self.use_web_search = use_web_search
        self.llm_model      = compound_model if use_web_search and compound_model else llm_model
        self.whisper_model  = whisper_model
        self.system_prompt  = system_prompt
        self.functions      = functions

        self.conversation_history: List[Dict] = []
        self.max_history = 10
        self._history_lock = threading.Lock()

        search_mode = "with WEB SEARCH" if use_web_search else "without web search"
        print(f"🧠 Groq Brain initialised — {self.llm_model} ({search_mode})")

    # ------------------------------------------------------------------
    # Public: Speech-to-Text
    # ------------------------------------------------------------------

    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        try:
            with open(audio_file_path, "rb") as f:
                result = self.client.audio.transcriptions.create(
                    file=f,
                    model=self.whisper_model,
                    response_format="text",
                    language="en",
                )
            text = result.strip()
            return text or None
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None

    # ------------------------------------------------------------------
    # Public: Chat
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """
        Standard chat turn.

        If the model returns ONLY tool calls with no text (common when it
        decides to gesture without speaking), a follow-up call is made
        without tools to get the verbal response. This ensures the robot
        always has something to say rather than falling back to the generic
        "Sorry, I didn't catch that" message.
        """
        try:
            messages = (
                [{"role": "system", "content": self.system_prompt}]
                + _sanitize_history(self.conversation_history)
                + [{"role": "user", "content": user_message}]
            )

            if self.use_web_search or "compound" in self.llm_model.lower():
                print("🌐 Using compound/web-search model (no gesture support)")
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=200,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    tools=self.functions,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=150,
                )

            message        = response.choices[0].message
            response_text  = _clean_response_text(message.content or "")
            function_calls = _parse_tool_calls(message)

            # ── Follow-up if model returned only tool calls ────────────
            # When the model calls gestures but forgets to include text,
            # ask it once more (no tools) to get the verbal response.
            if function_calls and not response_text:
                response_text = self._get_verbal_response(messages, function_calls)

            # ── Commit to history only when we have a real text reply ──
            if response_text:
                with self._history_lock:
                    self.conversation_history.append(
                        {"role": "user", "content": user_message}
                    )
                    self.conversation_history.append(
                        {"role": "assistant", "content": response_text}
                    )
                    self._trim_history()

            self._log_response(response_text, function_calls)
            return response_text, function_calls or None

        except Exception as e:
            print(f"❌ Chat error: {e}")
            import traceback; traceback.print_exc()
            return None, None

    def _get_verbal_response(
        self,
        prior_messages: List[Dict],
        function_calls: List[Dict],
    ) -> str:
        """
        Follow-up call when the model returned only tool calls with no text.
        Tells the model which gestures it already chose, asks for the verbal
        response only (no tools this time so it can't loop).
        """
        try:
            gesture_names = [f["name"] for f in function_calls if f["name"] != "web_search"]
            context = ""
            if gesture_names:
                context = f"[You already chose to perform: {', '.join(gesture_names)}] "

            followup_messages = prior_messages + [{
                "role": "system",
                "content": (
                    context +
                    "Now provide your spoken response to the user. "
                    "Keep it to 1-3 sentences."
                ),
            }]

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=followup_messages,
                temperature=0.7,
                max_tokens=150,
                # No tools — we just want text
            )
            text = _clean_response_text(response.choices[0].message.content or "")
            if text:
                print("🔄 Follow-up call got verbal response")
            return text
        except Exception as e:
            print(f"⚠️  Follow-up verbal response failed: {e}")
            return ""

    def chat_with_context(
        self,
        user_message: str,
        context: str,
    ) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """
        Single LLM call with injected context (search results, tool output, etc.).
        Self-contained — commits user message and final answer to history.
        """
        try:
            messages = (
                [{"role": "system", "content": self.system_prompt}]
                + _sanitize_history(self.conversation_history)
                + [{
                    "role": "system",
                    "content": (
                        "[Search / tool result — use this to answer the user; "
                        "do NOT repeat it verbatim]:\n" + context
                    ),
                }]
                + [{"role": "user", "content": user_message}]
            )

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                tools=self.functions,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=150,
            )

            message        = response.choices[0].message
            response_text  = _clean_response_text(message.content or "")
            function_calls = _parse_tool_calls(message)

            # Same follow-up logic as chat()
            if function_calls and not response_text:
                response_text = self._get_verbal_response(messages, function_calls)

            with self._history_lock:
                self.conversation_history.append(
                    {"role": "user", "content": user_message}
                )
                self.conversation_history.append(
                    {"role": "assistant", "content": response_text}
                )
                self._trim_history()

            self._log_response(response_text, function_calls, tag="ctx")
            return response_text, function_calls or None

        except Exception as e:
            print(f"❌ chat_with_context error: {e}")
            import traceback; traceback.print_exc()
            return None, None

    # ------------------------------------------------------------------
    # Public: Search intent detection
    # ------------------------------------------------------------------

    def needs_search(self, message: str) -> bool:
        """
        Keyword heuristic for the search fast-path.
        Only called when USE_WEB_SEARCH = True in config.
        """
        lowered = message.lower()
        return any(kw in lowered for kw in config.SEARCH_KEYWORDS)

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def reset_conversation(self):
        with self._history_lock:
            self.conversation_history = []
        print("🔄 Conversation history cleared")

    def add_context(self, context: str):
        with self._history_lock:
            self.conversation_history.insert(
                0, {"role": "system", "content": f"Current context: {context}"}
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trim_history(self):
        cap = self.max_history * 2
        if len(self.conversation_history) > cap:
            self.conversation_history = self.conversation_history[-cap:]

    def _log_response(
        self,
        response_text: str,
        function_calls: List[Dict],
        tag: str = "",
    ):
        label = f"💬 AI{f' ({tag})' if tag else ''}"
        if response_text:
            preview = response_text[:100] + ("…" if len(response_text) > 100 else "")
            print(f"{label}: {preview}")
        if function_calls:
            print(f"🎬 Functions: {[f['name'] for f in function_calls]}")


# ---------------------------------------------------------------------------
# Quick connectivity test
# ---------------------------------------------------------------------------

def test_groq_connection(api_key: str) -> bool:
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say 'hello' in one word"}],
            max_tokens=10,
        )
        print(f"✅ Groq API test OK: {resp.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        return False