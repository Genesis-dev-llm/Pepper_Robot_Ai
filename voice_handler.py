"""
Voice Handler - Audio Recording + Speech-to-Text
Uses laptop microphone + Groq Whisper for STT
Architecture designed so Pepper's mics can be swapped in Phase 3

Push-to-Talk:  Hold R → speak → release → transcribed automatically
"""

import os
import threading
import tempfile
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Optional, Callable


class VoiceHandler:

    @staticmethod
    def validate_setup() -> bool:
        """
        Check that the audio back-end (PortAudio) is available.

        Returns True if everything looks good.
        Raises RuntimeError with a human-friendly fix message otherwise.
        """
        try:
            devices = sd.query_devices()
            input_devs = [d for d in devices if d["max_input_channels"] > 0]
            if not input_devs:
                raise RuntimeError(
                    "No input devices found. "
                    "Plug in a microphone or enable the built-in mic."
                )
            return True
        except OSError as e:
            if "PortAudio" in str(e):
                raise RuntimeError(
                    "PortAudio library not found!\n"
                    "  Fix on Ubuntu/Debian:  sudo apt-get install libportaudio2\n"
                    "  Fix on macOS:          brew install portaudio\n"
                    "  Fix on Windows:        pip install sounddevice  (ships PortAudio)"
                ) from e
            raise

    def __init__(
        self,
        transcribe_fn: Callable[[str], Optional[str]],
        sample_rate: int = 16000,
        channels: int = 1,
        min_duration: float = 0.5,   # Ignore recordings shorter than this
        max_duration: float = 30.0,  # Max recording length in seconds
    ):
        """
        Initialize Voice Handler

        Args:
            transcribe_fn:  Callable that takes a WAV file path and returns text
                            (injected from GroqBrain.transcribe_audio)
            sample_rate:    Audio sample rate (16 kHz optimal for Whisper)
            channels:       1 = mono (all we need for speech)
            min_duration:   Minimum clip length to bother transcribing
            max_duration:   Safety cap - auto-stop after this many seconds
        """
        self.transcribe_fn   = transcribe_fn
        self.sample_rate     = sample_rate
        self.channels        = channels
        self.min_duration    = min_duration
        self.max_duration    = max_duration

        # Recording state (thread-safe via lock)
        self._lock           = threading.Lock()
        self._is_recording   = False
        self._audio_chunks: list = []
        self._stream: Optional[sd.InputStream] = None
        self._auto_stop_timer: Optional[threading.Timer] = None

        # Status callbacks (set externally to update GUI)
        self.on_recording_start: Optional[Callable] = None
        self.on_recording_stop:  Optional[Callable] = None
        self.on_transcribing:    Optional[Callable] = None
        self.on_transcribed:     Optional[Callable[[str], None]] = None
        self.on_error:           Optional[Callable[[str], None]] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start_recording(self) -> bool:
        """
        Start capturing audio from the default microphone.

        Returns:
            True if recording started, False if already recording
        """
        with self._lock:
            if self._is_recording:
                return False

            self._audio_chunks = []
            self._is_recording = True

        try:
            # Sounddevice callback - called on each audio block
            def _audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"⚠️ Audio stream status: {status}")
                with self._lock:
                    if self._is_recording:
                        self._audio_chunks.append(indata.copy())

            self._stream = sd.InputStream(
                samplerate = self.sample_rate,
                channels   = self.channels,
                dtype      = "float32",
                callback   = _audio_callback,
            )
            self._stream.start()

            # Auto-stop safety timer
            self._auto_stop_timer = threading.Timer(
                self.max_duration, self._auto_stop
            )
            self._auto_stop_timer.daemon = True
            self._auto_stop_timer.start()

            print(f"🎙️ Recording started (max {self.max_duration}s)")

            if self.on_recording_start:
                self.on_recording_start()

            return True

        except Exception as e:
            with self._lock:
                self._is_recording = False
            error_msg = f"Failed to start recording: {e}"
            print(f"❌ {error_msg}")
            if self.on_error:
                self.on_error(error_msg)
            return False

    def stop_recording_and_transcribe(self) -> None:
        """
        Stop recording, save audio, transcribe in background thread.
        Non-blocking - result delivered via on_transcribed callback.
        """
        audio_path = self._stop_stream()
        if audio_path is None:
            return

        # Run transcription in a background thread so we don't block
        threading.Thread(
            target   = self._transcribe_worker,
            args     = (audio_path,),
            daemon   = True,
            name     = "VoiceTranscribeThread",
        ).start()

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _stop_stream(self) -> Optional[str]:
        """
        Stop the audio stream and write captured audio to a temp WAV file.

        Returns:
            Path to WAV file, or None if nothing was captured / too short.
        """
        # Cancel auto-stop timer if still running
        if self._auto_stop_timer:
            self._auto_stop_timer.cancel()
            self._auto_stop_timer = None

        with self._lock:
            if not self._is_recording:
                return None
            self._is_recording = False
            chunks = list(self._audio_chunks)
            self._audio_chunks = []

        # Cleanly close the stream
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self.on_recording_stop:
            self.on_recording_stop()

        if not chunks:
            print("⚠️ No audio data captured")
            if self.on_error:
                self.on_error("No audio data captured")
            return None

        # Check duration
        audio  = np.concatenate(chunks, axis=0)
        duration = len(audio) / self.sample_rate

        if duration < self.min_duration:
            print(f"⚠️ Recording too short ({duration:.1f}s < {self.min_duration}s), ignoring")
            if self.on_error:
                self.on_error(f"Recording too short ({duration:.1f}s)")
            return None

        print(f"⏹️ Recording stopped ({duration:.1f}s captured)")

        # Write to temp WAV file
        try:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(path, audio, self.sample_rate)
            return path
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")
            if self.on_error:
                self.on_error(f"Failed to save audio: {e}")
            return None

    def _transcribe_worker(self, audio_path: str) -> None:
        """Background worker: send audio to Whisper, fire callback with result."""
        try:
            if self.on_transcribing:
                self.on_transcribing()

            print("🔄 Transcribing audio via Groq Whisper...")
            text = self.transcribe_fn(audio_path)

            # Clean up temp file
            try:
                os.remove(audio_path)
            except Exception:
                pass

            if text:
                print(f"✅ Transcribed: \"{text}\"")
                if self.on_transcribed:
                    self.on_transcribed(text)
            else:
                print("⚠️ Empty transcription — nothing spoken?")
                if self.on_error:
                    self.on_error("No speech detected")

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            try:
                os.remove(audio_path)
            except Exception:
                pass
            if self.on_error:
                self.on_error(f"Transcription failed: {e}")

    def _auto_stop(self) -> None:
        """Called by timer when max_duration is reached."""
        print(f"⏱️ Max recording duration ({self.max_duration}s) reached, auto-stopping")
        self.stop_recording_and_transcribe()


# ------------------------------------------------------------------ #
#  Utility: list available microphones                                #
# ------------------------------------------------------------------ #

def list_microphones():
    """Print all available input devices — useful for debugging."""
    print("\n🎙️ Available microphones:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " ← default" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{marker}")
    print()


# ------------------------------------------------------------------ #
#  Standalone test                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=== VoiceHandler standalone test ===")
    print("This test only checks recording (no Groq API needed)\n")

    list_microphones()

    # Dummy transcribe function for testing
    def dummy_transcribe(path):
        size = os.path.getsize(path)
        print(f"  (dummy) Would send {size} byte WAV to Whisper")
        return "test transcription"

    handler = VoiceHandler(transcribe_fn=dummy_transcribe)

    handler.on_recording_start  = lambda: print("  → recording started callback")
    handler.on_recording_stop   = lambda: print("  → recording stopped callback")
    handler.on_transcribing     = lambda: print("  → transcribing callback")
    handler.on_transcribed      = lambda t: print(f"  → transcribed: '{t}'")
    handler.on_error            = lambda e: print(f"  → error: {e}")

    input("Press ENTER to start recording (3 seconds)...")
    handler.start_recording()
    time.sleep(3)
    handler.stop_recording_and_transcribe()
    time.sleep(2)  # Wait for background thread
    print("Test complete.")