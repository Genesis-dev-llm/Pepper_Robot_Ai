"""
Chat Logger — single rolling log file at logs/chat.log

All sessions append to the same file. Each session is separated by a
header line with the timestamp.

Session rotation: at most 3 sessions are kept. On startup, if the log
already contains 3 or more sessions, the oldest are trimmed before the
new session header is written — so the file never grows beyond 3 sessions.
Delete logs/chat.log manually if you want to start completely fresh.
"""

import os
import re
import threading
from datetime import datetime

_LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "chat.log")

# Maximum number of past sessions to keep in the log file.
_MAX_SESSIONS = 3


class ChatLogger:
    def __init__(self):
        self._lock = threading.Lock()
        os.makedirs(_LOG_DIR, exist_ok=True)
        # Trim BEFORE writing the new header so the new session counts toward
        # the limit.  We keep (_MAX_SESSIONS - 1) old sessions, then add 1 new.
        self._trim_old_sessions(keep=_MAX_SESSIONS - 1)
        self._write_session_header()

    # ── Session rotation ───────────────────────────────────────────────────────

    def _trim_old_sessions(self, keep: int) -> None:
        """
        Rewrite the log file keeping only the `keep` most recent sessions.

        Session boundaries are detected by the pattern:
            \\n={60}\\nSESSION STARTED ...
        which is the exact header written by _write_session_header().

        Edge cases handled:
        - File does not exist yet → nothing to do.
        - File exists but has fewer than `keep` sessions → nothing to do.
        - Any read/write error → silently ignored so init never raises.
        """
        if not os.path.isfile(_LOG_FILE):
            return
        try:
            with open(_LOG_FILE, "r", encoding="utf-8") as f:
                content = f.read()

            # Split on the start of each session header.
            # lookahead keeps the separator attached to the session that follows it.
            parts    = re.split(r'(?=\n={60}\nSESSION STARTED)', content)
            sessions = [p for p in parts if "SESSION STARTED" in p]

            if len(sessions) <= keep:
                return  # nothing to trim

            trimmed = "".join(sessions[-keep:])
            with open(_LOG_FILE, "w", encoding="utf-8") as f:
                f.write(trimmed)
        except Exception as e:
            print(f"⚠️  Could not trim chat log: {e}")

    # ── Session header ─────────────────────────────────────────────────────────

    def _write_session_header(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"SESSION STARTED  {ts}\n")
            f.write(f"{'='*60}\n")

    # ── Write helpers ──────────────────────────────────────────────────────────

    def _write(self, line: str):
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {line}\n")

    # ── Public API ─────────────────────────────────────────────────────────────

    def log_user(self, text: str, source: str = "text"):
        icon = "🎙️" if source == "voice" else "💬"
        self._write(f"{icon} USER: {text}")

    def log_pepper(self, text: str):
        self._write(f"🤖 PEPPER: {text}")

    def log_search(self, query: str):
        self._write(f"🔍 SEARCH: {query}")

    def log_system(self, text: str):
        self._write(f"   {text}")