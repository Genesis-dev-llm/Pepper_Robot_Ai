"""
Chat Logger — single rolling log file at logs/chat.log

All sessions append to the same file. Each session is separated by a
header line with the timestamp. No rotation, no multiple files.
Delete logs/chat.log manually if you want to start fresh.
"""

import os
import threading
from datetime import datetime

_LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "chat.log")


class ChatLogger:
    def __init__(self):
        self._lock = threading.Lock()
        os.makedirs(_LOG_DIR, exist_ok=True)
        self._write_session_header()

    def _write_session_header(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"SESSION STARTED  {ts}\n")
            f.write(f"{'='*60}\n")

    def _write(self, line: str):
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {line}\n")

    def log_user(self, text: str, source: str = "text"):
        icon = "🎙️" if source == "voice" else "💬"
        self._write(f"{icon} USER: {text}")

    def log_pepper(self, text: str):
        self._write(f"🤖 PEPPER: {text}")

    def log_search(self, query: str):
        self._write(f"🔍 SEARCH: {query}")

    def log_system(self, text: str):
        self._write(f"   {text}")