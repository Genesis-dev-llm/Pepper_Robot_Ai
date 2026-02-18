"""
Groq Brain - LLM and Speech-to-Text

Key design decisions:
- chat()               — standard turn; user message only added to history on success
- chat_with_context()  — self-contained; does NOT require chat() to have been called first;
                         injects search/tool results as a system message for a single LLM call
- needs_search()       — fast keyword heuristic so main.py can skip the intent-detection
                         round-trip and go straight to search → single LLM call
"""

import json
import threading
from typing import Callable, Dict, List, Optional, Tuple

from groq import Groq

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tool_calls(message) -> List[Dict]:
    """Extract function-call dicts from a Groq response message."""
    calls = []
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            calls.append({
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments)
                             if tc.function.arguments else {},
            })
    return calls


def _sanitize_history(history: List[Dict]) -> List[Dict]:
    """
    Return a clean copy of conversation history safe to send to the Groq API.

    Groq validates that any tool_calls in assistant messages are also listed
    in request.tools with matching names.  If a prior model turn generated a
    malformed tool name (e.g. 'look_around{}') or the history was built from
    a different tools list, the API returns a 400.

    Fix: strip all tool_calls metadata from assistant messages — keep only
    the text content.  We never need the raw tool call objects in history;
    the actual results were always handled at call time, not replayed.
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
        self.client        = Groq(api_key=api_key)
        self.use_web_search = use_web_search
        self.llm_model     = compound_model if use_web_search and compound_model else llm_model
        self.whisper_model = whisper_model
        self.system_prompt = system_prompt
        self.functions     = functions

        # Conversation history — only confirmed turns live here
        self.conversation_history: List[Dict] = []
        self.max_history = 10

        # Thread safety: only one LLM call should write history at a time
        self._history_lock = threading.Lock()

        search_mode = "with WEB SEARCH" if use_web_search else "without web search"
        print(f"🧠 Groq Brain initialised — {self.llm_model} ({search_mode})")

    # ------------------------------------------------------------------
    # Public: Speech-to-Text
    # ------------------------------------------------------------------

    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """Transcribe an audio file using Groq Whisper."""
        try:
            with open(audio_file_path, "rb") as f:
                result = self.client.audio.transcriptions.create(
                    file=f,
                    model=self.whisper_model,
                    response_format="text",
                    language="en",
                )
            text = result.strip()
            if text:
                print(f"🎤 Transcribed: '{text}'")
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

        The user message is only committed to history after a successful
        API response — preventing dangling messages on failure.

        Returns:
            (response_text, function_calls | None)
        """
        try:
            # Build message list WITHOUT mutating history yet
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

            message       = response.choices[0].message
            response_text  = message.content or ""
            function_calls = _parse_tool_calls(message)

            # ✅ Only commit when there is a real text reply.
            # If the model returned ONLY a tool call (empty response_text), do NOT
            # commit yet — the caller will follow up with chat_with_context() which
            # will commit the complete user+final_answer exchange.  Committing here
            # would leave a dangling user message that chat_with_context() then
            # duplicates, corrupting history.
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

    def chat_with_context(
        self,
        user_message: str,
        context: str,
    ) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """
        Single LLM call with injected context (search results, tool output, etc.).

        Completely self-contained — does NOT require chat() to have been called
        first.  The user message and final answer are both committed to history.

        The context is injected as a system message so it doesn't pollute the
        visible conversation history with raw data blobs.

        Returns:
            (response_text, function_calls | None)
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
            response_text  = message.content or ""
            function_calls = _parse_tool_calls(message)

            # ✅ Commit user message + answer to history
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
        Fast keyword heuristic to detect whether a message is likely to
        benefit from a web search.

        When this returns True, main.py should:
          1. Run the search immediately
          2. Call chat_with_context(message, results)   ← ONE LLM call total

        When False, call chat(message) as normal.  The model may still emit
        a web_search function call for edge cases not caught by keywords.
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
        """Prepend a system-level context note (e.g. vision info)."""
        with self._history_lock:
            self.conversation_history.insert(
                0, {"role": "system", "content": f"Current context: {context}"}
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _trim_history(self):
        """Keep only the most recent max_history exchanges. Call inside lock."""
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