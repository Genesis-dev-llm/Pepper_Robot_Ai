"""
Wake Word Handler — Picovoice Porcupine

Always-on background thread that listens for a configurable wake word
and fires on_wake() when detected.

Why Porcupine:
- Runs entirely on-device (no cloud, no network dependency)
- Very low CPU (~1%) — designed for always-on use
- Free tier includes built-in keywords: "jarvis", "computer", "bumblebee",
  "grasshopper", "picovoice", "porcupine", "alexa", "hey google", "hey siri",
  "ok google", "terminator"
- Custom keywords (e.g. "hey pepper") are FREE — train one at console.picovoice.ai,
  download the .ppn file, and set WAKE_WORD to its path in config.py

Install:
    pip install pvporcupine pvrecorder

Free access key:
    https://console.picovoice.ai/

Graceful degradation:
    If pvporcupine/pvrecorder are not installed, _PORCUPINE_AVAILABLE is set
    to False at import time.  main.py checks this before constructing the
    handler and prints install instructions if it is False.
"""

import os
import threading
import traceback
from typing import Callable, Optional

try:
    import pvporcupine
    import pvrecorder
    _PORCUPINE_AVAILABLE = True
except ImportError:
    _PORCUPINE_AVAILABLE = False
    print("⚠️  pvporcupine/pvrecorder not installed — wake word will be disabled")
    print("   Install: pip install pvporcupine pvrecorder")


class WakeWordHandler:
    """
    Background listener that fires on_wake() whenever the configured
    wake word is detected in the microphone stream.

    Usage:
        handler = WakeWordHandler(
            keyword    = "jarvis",
            access_key = "your-picovoice-key",
            sensitivity = 0.5,
            on_wake    = my_callback,
        )
        handler.start()
        # … later …
        handler.stop()

    The on_wake callback is called from the WakeWordListener background
    thread, so it must be thread-safe.  In main.py, _on_wake_word() is
    designed with this in mind (all GUI calls go through message_queue,
    and state.ptt_lock is acquired with blocking=False).
    """

    def __init__(
        self,
        keyword:     str,
        access_key:  str,
        sensitivity: float = 0.5,
        on_wake:     Optional[Callable[[], None]] = None,
    ):
        """
        Args:
            keyword:     Built-in keyword name (e.g. "jarvis") or absolute
                         path to a custom .ppn keyword file.
            access_key:  Picovoice access key (free at console.picovoice.ai).
            sensitivity: Detection sensitivity, 0.0–1.0.  Higher values catch
                         more utterances but also produce more false positives.
            on_wake:     Callable fired (no arguments) on each detection.
        """
        if not _PORCUPINE_AVAILABLE:
            raise RuntimeError(
                "pvporcupine is not installed. "
                "Run: pip install pvporcupine pvrecorder"
            )

        self._keyword     = keyword
        self._access_key  = access_key
        self._sensitivity = float(max(0.0, min(1.0, sensitivity)))
        self._on_wake     = on_wake

        self._running    = False
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self):
        """Start the background wake word listener thread."""
        if self._is_running:
            return
        self._running    = True
        self._is_running = True   # set before thread starts to close race window
        self._thread     = threading.Thread(
            target=self._run,
            daemon=True,
            name="WakeWordListener",
        )
        self._thread.start()

    def stop(self):
        """
        Signal the listener thread to stop and wait for it to exit.
        Safe to call even if the thread never started or already stopped.
        """
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread     = None
        self._is_running = False

    # ── Background thread ──────────────────────────────────────────────────────

    def _run(self):
        """
        Main loop for the WakeWordListener thread.

        pvporcupine.create() accepts either a built-in keyword name (string)
        or a path to a .ppn file.  We pass it inside a list because the API
        supports detecting multiple keywords simultaneously — we only need one.

        pvrecorder.PvRecorder.read() is a blocking call that returns exactly
        porcupine.frame_length samples (~512 at 16kHz).  The loop naturally
        yields between frames, so setting self._running = False from another
        thread causes exit within one frame period (~32ms).

        porcupine.process() returns the index of the detected keyword (>= 0)
        or -1 for no detection.  Since we only have one keyword (index 0),
        any result >= 0 is a match.

        Resources (recorder, porcupine) are always cleaned up in finally,
        even if an exception is raised during init or the loop.
        """
        porcupine = None
        recorder  = None

        try:
            # Determine whether keyword is a built-in name or a .ppn file path
            if os.path.isfile(self._keyword):
                # Custom .ppn file (paid Picovoice account)
                porcupine = pvporcupine.create(
                    access_key       = self._access_key,
                    keyword_paths    = [self._keyword],
                    sensitivities    = [self._sensitivity],
                )
            else:
                # Built-in keyword (free tier)
                porcupine = pvporcupine.create(
                    access_key   = self._access_key,
                    keywords     = [self._keyword],
                    sensitivities = [self._sensitivity],
                )

            recorder = pvrecorder.PvRecorder(
                frame_length = porcupine.frame_length,
            )
            recorder.start()

            print(f"👂 Wake word active — say '{self._keyword}' to activate Pepper")

            while self._running:
                pcm    = recorder.read()
                result = porcupine.process(pcm)
                if result >= 0:
                    print(f"🎤 Wake word detected: '{self._keyword}'")
                    if self._on_wake:
                        try:
                            self._on_wake()
                        except Exception as e:
                            print(f"⚠️  on_wake callback error: {e}")

        except Exception as e:
            print(f"❌ Wake word listener error: {e}")
            traceback.print_exc()

        finally:
            if recorder is not None:
                try:
                    recorder.stop()
                    recorder.delete()
                except Exception:
                    pass
            if porcupine is not None:
                try:
                    porcupine.delete()
                except Exception:
                    pass
            self._is_running = False
            print("👂 Wake word listener stopped")