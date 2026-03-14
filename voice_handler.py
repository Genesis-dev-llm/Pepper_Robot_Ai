"""
Voice Handler — Audio Recording + Speech-to-Text

Changes from previous version:
- RMS-based VAD: recordings whose energy is below config.VAD_THRESHOLD
  are discarded before sending to Whisper (saves API credits, reduces latency).
- on_audio_level callback: fired during recording with a normalized 0.0–1.0
  RMS value so the GUI can show a live audio level meter.
- Thread spawning for transcription/message dispatch is handled in main.py;
  this module only delivers the transcribed text via on_transcribed.
- vad_mode: when start_recording(vad_mode=True) is called and webrtcvad is
  installed, a background thread monitors audio in 20ms frames and triggers
  auto-stop once silence follows speech for VAD_SILENCE_SECONDS. Falls back
  to timer-based stop if webrtcvad is not installed.
"""

import os
import threading
import tempfile
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

import config

try:
    import webrtcvad
    _WEBRTCVAD_AVAILABLE = True
except ImportError:
    _WEBRTCVAD_AVAILABLE = False
    print("ℹ️  webrtcvad not installed — wake word will use timer-based stop")
    print("   Install: pip install webrtcvad")


class VoiceHandler:

    @staticmethod
    def validate_setup() -> bool:
        try:
            devices = sd.query_devices()
            if not any(d["max_input_channels"] > 0 for d in devices):
                raise RuntimeError(
                    "No input devices found. Plug in a microphone or enable the built-in mic."
                )
            return True
        except OSError as e:
            if "PortAudio" in str(e):
                raise RuntimeError(
                    "PortAudio library not found!\n"
                    "  Ubuntu/Debian: sudo apt-get install libportaudio2\n"
                    "  macOS:         brew install portaudio\n"
                    "  Windows:       pip install sounddevice  (ships PortAudio)"
                ) from e
            raise

    def __init__(
        self,
        transcribe_fn:  Callable[[str], Optional[str]],
        sample_rate:    int   = 16000,
        channels:       int   = 1,
        min_duration:   float = 0.5,
        max_duration:   float = 30.0,
    ):
        self.transcribe_fn = transcribe_fn
        self.sample_rate   = sample_rate
        self.channels      = channels
        self.min_duration  = min_duration
        self.max_duration  = max_duration

        self._lock            = threading.Lock()
        self._is_recording    = False
        self._audio_chunks: list = []
        self._stream: Optional[sd.InputStream] = None
        self._auto_stop_timer: Optional[threading.Timer] = None
        self._vad_thread: Optional[threading.Thread] = None

        # Level meter update interval (seconds)
        self._level_timer: Optional[threading.Timer] = None

        # Callbacks (all optional)
        self.on_recording_start: Optional[Callable] = None
        self.on_recording_stop:  Optional[Callable] = None
        self.on_transcribing:    Optional[Callable] = None
        self.on_transcribed:     Optional[Callable[[str], None]] = None
        self.on_error:           Optional[Callable[[str], None]] = None
        # on_audio_level(float 0.0–1.0) — called ~10x/sec during recording
        self.on_audio_level:     Optional[Callable[[float], None]] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start_recording(self, vad_mode: bool = False) -> bool:
        """
        Start recording from the default microphone.

        Args:
            vad_mode: When True and webrtcvad is installed, recording stops
                      automatically once silence follows speech for
                      config.VAD_SILENCE_SECONDS. Falls back to timer if
                      webrtcvad is unavailable. PTT should always pass
                      vad_mode=False (default).
        """
        with self._lock:
            if self._is_recording:
                return False
            self._audio_chunks = []
            self._is_recording = True

        try:
            def _audio_callback(indata, frames, time_info, status):
                if status:
                    print(f"⚠️ Audio status: {status}")
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

            # Hard cap — always active regardless of vad_mode
            self._auto_stop_timer = threading.Timer(self.max_duration, self._auto_stop)
            self._auto_stop_timer.daemon = True
            self._auto_stop_timer.start()

            # VAD monitor — only when requested and library is available
            if vad_mode and _WEBRTCVAD_AVAILABLE:
                self._vad_thread = threading.Thread(
                    target=self._vad_monitor,
                    daemon=True,
                    name="VADMonitor",
                )
                self._vad_thread.start()
            elif vad_mode:
                # webrtcvad not installed — caller's timer acts as fallback
                pass

            # Start level meter updates
            self._schedule_level_update()

            mode_label = " (VAD)" if vad_mode and _WEBRTCVAD_AVAILABLE else ""
            print(f"🎙️ Recording started{mode_label} (max {self.max_duration}s)")
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
        audio_path = self._stop_stream()
        if audio_path is None:
            return
        threading.Thread(
            target=self._transcribe_worker,
            args=(audio_path,),
            daemon=True,
            name="VoiceTranscribeThread",
        ).start()

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    # ── VAD monitor ────────────────────────────────────────────────────────────

    def _vad_monitor(self):
        """
        Background thread: monitors incoming audio and stops recording once
        silence follows speech for config.VAD_SILENCE_SECONDS.

        webrtcvad requires 16-bit PCM in specific frame sizes. At 16kHz:
          10ms = 160 samples, 20ms = 320 samples, 30ms = 480 samples.
        We use 20ms frames as a good balance of responsiveness and accuracy.

        Float32 chunks from sounddevice are converted to int16 on the fly.
        A rolling buffer handles alignment between chunk boundaries and frames.
        """
        vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)

        frame_ms      = 20                                         # ms per frame
        frame_samples = int(self.sample_rate * frame_ms / 1000)   # 320 at 16kHz
        frame_bytes   = frame_samples * 2                         # int16 = 2 bytes

        buffer: bytes  = b""
        chunk_index    = 0
        speech_detected = False
        silence_start: Optional[float] = None

        while True:
            with self._lock:
                if not self._is_recording:
                    return
                new_chunks = self._audio_chunks[chunk_index:]
                chunk_index = len(self._audio_chunks)

            # Convert float32 → int16 → bytes and append to rolling buffer
            for chunk in new_chunks:
                pcm = (chunk.flatten() * 32767).astype(np.int16).tobytes()
                buffer += pcm

            # Process as many complete frames as possible
            while len(buffer) >= frame_bytes:
                frame  = buffer[:frame_bytes]
                buffer = buffer[frame_bytes:]

                try:
                    is_speech = vad.is_speech(frame, self.sample_rate)
                except Exception:
                    continue

                if is_speech:
                    speech_detected = True
                    silence_start   = None
                elif speech_detected:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= config.VAD_SILENCE_SECONDS:
                        print(f"🔇 Silence detected — auto-stopping")
                        self.stop_recording_and_transcribe()
                        return

            time.sleep(0.02)  # Check every 20ms — keeps up with frame rate

    # ── Level meter ────────────────────────────────────────────────────────────

    def _schedule_level_update(self):
        """Fire on_audio_level ~10x per second while recording."""
        if not self._is_recording:
            return
        self._emit_level()
        self._level_timer = threading.Timer(0.1, self._schedule_level_update)
        self._level_timer.daemon = True
        self._level_timer.start()

    def _emit_level(self):
        if not self.on_audio_level:
            return
        with self._lock:
            if not self._audio_chunks:
                return
            # Use the most recent chunk for responsiveness
            chunk = self._audio_chunks[-1]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        # Normalize to 0–1 with a reasonable ceiling for speech (~0.3 RMS)
        level = min(rms / 0.3, 1.0)
        try:
            self.on_audio_level(level)
        except Exception:
            pass

    # ── Internal ───────────────────────────────────────────────────────────────

    def _stop_stream(self) -> Optional[str]:
        if self._auto_stop_timer:
            self._auto_stop_timer.cancel()
            self._auto_stop_timer = None
        if self._level_timer:
            self._level_timer.cancel()
            self._level_timer = None
        # _vad_thread is daemon — it will exit on its own once _is_recording=False

        with self._lock:
            if not self._is_recording:
                return None
            self._is_recording = False
            chunks = list(self._audio_chunks)
            self._audio_chunks = []

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self.on_recording_stop:
            self.on_recording_stop()
        if self.on_audio_level:
            try:
                self.on_audio_level(0.0)
            except Exception:
                pass

        if not chunks:
            if self.on_error:
                self.on_error("No audio data captured")
            return None

        audio    = np.concatenate(chunks, axis=0)
        duration = len(audio) / self.sample_rate

        if duration < self.min_duration:
            msg = f"Recording too short ({duration:.1f}s)"
            print(f"⚠️ {msg}")
            if self.on_error:
                self.on_error(msg)
            return None

        # VAD: discard recordings that are silence
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < config.VAD_THRESHOLD:
            msg = f"No speech detected (RMS {rms:.4f} < {config.VAD_THRESHOLD})"
            print(f"⚠️ {msg}")
            if self.on_error:
                self.on_error("No speech detected — try speaking louder")
            return None

        print(f"⏹️ Recording stopped ({duration:.1f}s, RMS={rms:.4f})")

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
        try:
            if self.on_transcribing:
                self.on_transcribing()
            print("🔄 Transcribing via Groq Whisper…")
            text = self.transcribe_fn(audio_path)
            try:
                os.remove(audio_path)
            except Exception:
                pass
            if text:
                print(f"✅ Transcribed: \"{text}\"")
                if self.on_transcribed:
                    self.on_transcribed(text)
            else:
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
        print(f"⏱️ Max recording duration reached, auto-stopping")
        self.stop_recording_and_transcribe()


# ── Utility ────────────────────────────────────────────────────────────────────

def list_microphones():
    print("\n🎙️ Available microphones:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " ← default" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{marker}")
    print()