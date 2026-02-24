"""
Hybrid TTS Handler — 3-Tier System
  Tier 1: Groq TTS (Orpheus)   — fastest, primary
  Tier 2: Edge TTS             — unlimited, always available
  Tier 3: ElevenLabs           — rate-limited, last resort

Each tier generates its own temp file with the correct suffix for that format.
This avoids the fragile .replace(".wav", ".mp3") path derivation that broke
silently if the base path had no .wav extension.

Edge TTS uses asyncio.run() instead of manually managing an event loop.
"""

import asyncio
import os
import subprocess
import tempfile
from typing import Optional

import edge_tts
from groq import Groq


try:
    from elevenlabs import ElevenLabs as _ElevenLabsClient
    _ELEVENLABS_IMPORTABLE = True
except ImportError:
    _ElevenLabsClient      = None
    _ELEVENLABS_IMPORTABLE = False


class HybridTTSHandler:
    def __init__(
        self,
        groq_api_key:        str,
        groq_voice:          str = "hannah",
        groq_model:          str = "canopylabs/orpheus-v1-english",
        elevenlabs_api_key:  Optional[str] = None,
        elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        edge_voice:          str = "en-US-AriaNeural",
        edge_rate:           str = "+0%",
    ):
        # ── Tier 1: Groq ──────────────────────────────────────────────
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_voice  = groq_voice
        self.groq_model  = groq_model

        # ── Tier 2: Edge TTS ──────────────────────────────────────────
        self.edge_voice = edge_voice
        self.edge_rate  = edge_rate

        # ── Tier 3: ElevenLabs ────────────────────────────────────────
        self._el_client          = None
        self.elevenlabs_enabled  = False
        self.elevenlabs_voice_id = elevenlabs_voice_id

        if elevenlabs_api_key:
            if not _ELEVENLABS_IMPORTABLE:
                print("⚠️  ElevenLabs key provided but package not installed.")
                print("    Install: pip install elevenlabs --break-system-packages")
            else:
                try:
                    self._el_client         = _ElevenLabsClient(api_key=elevenlabs_api_key)
                    self.elevenlabs_enabled = True
                except Exception as e:
                    print(f"⚠️  ElevenLabs init failed: {e}")

        # Daily-limit flags (reset on restart)
        self._groq_daily_limit_hit       = False
        self._elevenlabs_daily_limit_hit = False

        print("🔊 Hybrid TTS ready (3-tier):")
        print(f"   Tier 1: Groq Orpheus   ({groq_voice})")
        print(f"   Tier 2: Edge TTS       ({edge_voice}) [unlimited]")
        el_status = f"({elevenlabs_voice_id})" if self.elevenlabs_enabled else "(disabled)"
        print(f"   Tier 3: ElevenLabs     {el_status}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(
        self,
        text:        str,
        output_file: Optional[str] = None,
        emotion:     Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate speech via Groq → Edge TTS → ElevenLabs fallback chain.

        Each tier creates its own temp file with the correct suffix for its
        format (.wav for Groq, .mp3 for Edge and ElevenLabs). The output_file
        parameter is kept for API compatibility but only used as the Groq target;
        other tiers always generate their own paths.

        Returns path to a valid, non-empty audio file, or None on total failure.
        """
        # Groq writes WAV — use the provided path or make a fresh one
        if output_file is None:
            fd, output_file = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        # Tier 1 — Groq Orpheus
        if not self._groq_daily_limit_hit:
            if self._speak_groq(text, output_file, emotion):
                if self._valid_audio(output_file):
                    return output_file
                print("⚠️  Groq wrote empty/missing file — falling to Tier 2")

        # Tier 2 — Edge TTS (unlimited, always available)
        # Always creates its own .mp3 temp file — no path string manipulation.
        fd, edge_file = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        if self._speak_edge(text, edge_file):
            if self._valid_audio(edge_file):
                return edge_file
            print("⚠️  Edge TTS wrote empty/missing file — falling to Tier 3")
            self._cleanup(edge_file)

        # Tier 3 — ElevenLabs (last resort)
        if self.elevenlabs_enabled and not self._elevenlabs_daily_limit_hit:
            fd, el_file = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            if self._speak_elevenlabs(text, el_file):
                if self._valid_audio(el_file):
                    return el_file
                print("⚠️  ElevenLabs wrote empty/missing file")
                self._cleanup(el_file)

        return None

    def play_audio(self, audio_file: str):
        """Play an audio file using the best available system player."""
        players = ["aplay", "mpg123", "ffplay", "paplay"]
        for player in players:
            try:
                if player == "ffplay":
                    subprocess.run(
                        [player, "-nodisp", "-autoexit", "-loglevel", "quiet", audio_file],
                        check=True,
                    )
                else:
                    subprocess.run(
                        [player, audio_file],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        print("⚠️  No audio player found (install aplay, mpg123, or ffplay)")

    def speak_and_play(self, text: str, emotion: Optional[str] = None) -> bool:
        """Generate speech and play it locally (offline mode / testing)."""
        path = self.speak(text, emotion=emotion)
        if not path:
            return False
        self.play_audio(path)
        self._cleanup(path)
        return True

    def reset_daily_limits(self):
        self._groq_daily_limit_hit       = False
        self._elevenlabs_daily_limit_hit = False
        print("🔄 Daily limit flags reset")

    # ------------------------------------------------------------------
    # Tier implementations
    # ------------------------------------------------------------------

    def _speak_groq(self, text: str, output_file: str, emotion: Optional[str]) -> bool:
        if self._groq_daily_limit_hit:
            return False
        try:
            input_text = f"[{emotion}] {text}" if emotion else text
            resp = self.groq_client.audio.speech.create(
                model=self.groq_model,
                voice=self.groq_voice,
                input=input_text,
                response_format="wav",
            )
            try:
                resp.stream_to_file(output_file)
            except AttributeError:
                with open(output_file, "wb") as f:
                    f.write(resp.read())
            print("🎤 Tier 1: Groq TTS")
            return True
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate limit" in err or "daily" in err:
                print("⏳ Groq TTS rate-limited → falling to Tier 2 (Edge)")
                self._groq_daily_limit_hit = True
            else:
                print(f"⚠️  Groq TTS error: {e}")
            return False

    def _speak_elevenlabs(self, text: str, output_file: str) -> bool:
        if not self.elevenlabs_enabled or self._elevenlabs_daily_limit_hit:
            return False
        try:
            audio_iter = self._el_client.text_to_speech.convert(
                voice_id=self.elevenlabs_voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128",
            )
            with open(output_file, "wb") as f:
                for chunk in audio_iter:
                    if chunk:
                        f.write(chunk)
            print("🎤 Tier 3: ElevenLabs TTS")
            return True
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "limit" in err or "429" in err or "subscription" in err:
                print("⚠️  ElevenLabs limit/auth → marking as exhausted")
                self._elevenlabs_daily_limit_hit = True
            else:
                print(f"⚠️  ElevenLabs error: {e}")
            return False

    async def _speak_edge_async(self, text: str, output_file: str) -> bool:
        try:
            comm = edge_tts.Communicate(text, self.edge_voice, rate=self.edge_rate)
            await comm.save(output_file)
            print("🎤 Tier 2: Edge TTS")
            return True
        except Exception as e:
            print(f"❌ Edge TTS error: {e}")
            return False

    def _speak_edge(self, text: str, output_file: str) -> bool:
        """
        Run the async Edge TTS call using asyncio.run().

        asyncio.run() creates a fresh event loop, runs the coroutine to
        completion, then closes the loop cleanly — replacing the previous
        manual new_event_loop / set_event_loop / close pattern which had
        unnecessary overhead and left the thread's event loop in a None state.
        """
        try:
            return asyncio.run(self._speak_edge_async(text, output_file))
        except Exception as e:
            print(f"❌ Edge TTS runner error: {e}")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_audio(path: str) -> bool:
        """Return True if path exists and is a non-empty file."""
        try:
            return os.path.exists(path) and os.path.getsize(path) > 0
        except OSError:
            return False

    @staticmethod
    def _cleanup(path: str):
        """Remove a temp file, ignoring errors."""
        try:
            os.remove(path)
        except OSError:
            pass


# Backward-compat alias
TTSHandler = HybridTTSHandler