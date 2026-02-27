"""
Groq Brain — LLM and Speech-to-Text

Changes from previous version:
- GESTURE_NAMES imported from config (single source of truth)
- Removed compound_model / web-search special path — tools are always passed,
  including web_search when USE_WEB_SEARCH is True in config.
- History trim now preserves the first user/assistant exchange (intro context)
  and removes from the middle when the window fills.
- _sanitize_history strips stale tool_call metadata from assistant turns.
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
    """Strip function-call artifacts and stage directions from spoken output."""
    # XML-style tool tags
    text = re.sub(r"<function=[^>]*>.*?</function>", "", text, flags=re.DOTALL)
    text = re.sub(r"<function=[^>/]*/?>", "", text)
    text = re.sub(r"<tool[^>]*>.*?</tool>", "", text, flags=re.DOTALL)
    # Asterisk stage directions
    text = re.sub(r"\*[^*]+\*", "", text)
    # Bare gesture names on their own line (derived from config — single source of truth)
    lines = text.splitlines()
    lines = [l for l in lines if l.strip().lower() not in config.GESTURE_NAMES]
    return "\n".join(lines).strip()


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
        api_key:       str,
        llm_model:     str,
        whisper_model: str,
        system_prompt: str,
        functions:     List[Dict],
        use_web_search: bool = False,
    ):
        self.client        = Groq(api_key=api_key)
        self.llm_model     = llm_model
        self.whisper_model = whisper_model
        self.system_prompt = system_prompt
        self.functions     = functions
        self.use_web_search = use_web_search

        self.conversation_history: List[Dict] = []
        self._history_lock = threading.Lock()

        print(f"🧠 Groq Brain ready — {self.llm_model} "
              f"({'web search ON' if use_web_search else 'web search OFF'})")

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
            messages = (
                [{"role": "system", "content": self.system_prompt}]
                + _sanitize_history(self.conversation_history)
                + [{"role": "user", "content": user_message}]
            )

            response = self.client.chat.completions.create(
                model       = self.llm_model,
                messages    = messages,
                tools       = self.functions,
                tool_choice = "auto",
                temperature = 0.7,
                max_tokens  = 150,
            )

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

            self._log(response_text, function_calls)
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
        """Single LLM call with injected search/tool context."""
        try:
            messages = (
                [{"role": "system", "content": self.system_prompt}]
                + _sanitize_history(self.conversation_history)
                + [{"role": "system", "content":
                    "[Search result — use this to answer; do NOT repeat verbatim]:\n" + context}]
                + [{"role": "user", "content": user_message}]
            )

            response = self.client.chat.completions.create(
                model       = self.llm_model,
                messages    = messages,
                tools       = self.functions,
                tool_choice = "auto",
                temperature = 0.7,
                max_tokens  = 150,
            )

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
            print(f"❌ chat_with_context error: {e}")
            import traceback; traceback.print_exc()
            return None, None

    def _get_verbal_response(
        self,
        prior_messages: List[Dict],
        function_calls: List[Dict],
    ) -> str:
        """Follow-up call when model returned only tool calls with no text."""
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
                max_tokens  = 150,
            )
            text = _clean_response_text(response.choices[0].message.content or "")
            if text:
                print("🔄 Follow-up verbal response obtained")
            return text
        except Exception as e:
            print(f"⚠️  Follow-up verbal response failed: {e}")
            return ""

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