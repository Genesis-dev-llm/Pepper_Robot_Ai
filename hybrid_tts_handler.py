"""
Hybrid TTS Handler — 3-Tier System
  Tier 1: Groq TTS (Orpheus)   — fastest
  Tier 2: ElevenLabs           — highest quality (optional)
  Tier 3: Edge TTS             — unlimited, always available

Changes from original:
- ElevenLabs updated to current SDK (v1+) using ElevenLabs client object.
  Old `from elevenlabs import generate, set_api_key, save` is no longer valid.
- asyncio: replaced asyncio.run() with explicit new_event_loop() + close() to
  avoid "event loop already running" errors when called from threaded contexts.
- Rate-limit handling: on any 429/rate-limit, fall through to next tier
  immediately (no blocking waits that freeze the robot).
"""

import asyncio
import os
import subprocess
import tempfile
from typing import Optional

import edge_tts
from groq import Groq


# ---------------------------------------------------------------------------
# ElevenLabs — optional, gracefully degraded if not installed / not keyed
# ---------------------------------------------------------------------------
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
        elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM",   # ElevenLabs voice ID (Rachel)
        edge_voice:          str = "en-US-AriaNeural",
        edge_rate:           str = "+0%",
    ):
        # ── Tier 1: Groq ──────────────────────────────────────────────
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_voice  = groq_voice
        self.groq_model  = groq_model

        # ── Tier 2: ElevenLabs ────────────────────────────────────────
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

        # ── Tier 3: Edge TTS ──────────────────────────────────────────
        self.edge_voice = edge_voice
        self.edge_rate  = edge_rate

        # Daily-limit flags (reset on restart)
        self._groq_daily_limit_hit       = False
        self._elevenlabs_daily_limit_hit = False

        # Status summary
        print("🔊 Hybrid TTS ready (3-tier):")
        print(f"   Tier 1: Groq Orpheus   ({groq_voice})")
        el_status = f"({elevenlabs_voice_id})" if self.elevenlabs_enabled else "(disabled)"
        print(f"   Tier 2: ElevenLabs     {el_status}")
        print(f"   Tier 3: Edge TTS       ({edge_voice}) [unlimited]")

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
        Generate speech via Groq → ElevenLabs → Edge TTS fallback chain.

        Returns:
            Path to the generated audio file, or None on total failure.
        """
        if output_file is None:
            fd, output_file = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        # Tier 1 — Groq
        if not self._groq_daily_limit_hit:
            if self._speak_groq(text, output_file, emotion):
                return output_file

        # Tier 2 — ElevenLabs
        if self.elevenlabs_enabled and not self._elevenlabs_daily_limit_hit:
            el_file = output_file.replace(".wav", ".mp3")
            if self._speak_elevenlabs(text, el_file):
                return el_file

        # Tier 3 — Edge TTS (unlimited)
        edge_file = output_file.replace(".wav", ".mp3")
        if self._speak_edge(text, edge_file):
            return edge_file

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
        """Generate speech and play it locally (for testing without Pepper)."""
        path = self.speak(text, emotion=emotion)
        if not path:
            return False
        self.play_audio(path)
        try:
            os.remove(path)
        except OSError:
            pass
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
            # The Groq audio API returns a BinaryAPIResponse — use
            # stream_to_file() or read() depending on SDK version.
            # stream_to_file() is the official method; fall back to read().
            try:
                resp.stream_to_file(output_file)
            except AttributeError:
                with open(output_file, "wb") as f:
                    f.write(resp.read())
            print("🎤 Tier 1: Groq TTS")
            return True
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate limit" in err:
                print("⏳ Groq rate-limited → falling to Tier 2")
            else:
                print(f"⚠️  Groq TTS error: {e}")
            return False

    def _speak_elevenlabs(self, text: str, output_file: str) -> bool:
        """
        ElevenLabs Tier 2 using the current SDK (v1+).
        Uses eleven_turbo_v2_5 — the current free-tier supported model.
        The deprecated eleven_monolingual_v1 has been removed from free tier.
        """
        if not self.elevenlabs_enabled or self._elevenlabs_daily_limit_hit:
            return False
        try:
            audio_iter = self._el_client.text_to_speech.convert(
                voice_id=self.elevenlabs_voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",   # Free-tier compatible
                output_format="mp3_44100_128",
            )
            with open(output_file, "wb") as f:
                for chunk in audio_iter:
                    if chunk:
                        f.write(chunk)
            print("🎤 Tier 2: ElevenLabs TTS")
            return True
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "limit" in err or "429" in err or "subscription" in err or "deprecated" in err:
                print("⚠️  ElevenLabs limit/auth → Tier 3")
                self._elevenlabs_daily_limit_hit = True
            else:
                print(f"⚠️  ElevenLabs error: {e}")
            return False

    async def _speak_edge_async(self, text: str, output_file: str) -> bool:
        try:
            comm = edge_tts.Communicate(text, self.edge_voice, rate=self.edge_rate)
            await comm.save(output_file)
            print("🎤 Tier 3: Edge TTS")
            return True
        except Exception as e:
            print(f"❌ Edge TTS error: {e}")
            return False

    def _speak_edge(self, text: str, output_file: str) -> bool:
        """
        Synchronous wrapper for Edge TTS.

        Uses an explicit new event loop instead of asyncio.run() to avoid
        "event loop already running" errors in threaded contexts.
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._speak_edge_async(text, output_file))
        finally:
            loop.close()
            asyncio.set_event_loop(None)


# Backward-compat alias
TTSHandler = HybridTTSHandler