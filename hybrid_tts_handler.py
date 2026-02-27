"""
Hybrid TTS Handler — 3-Tier System
  Tier 1: Groq TTS (Orpheus)   — fastest, primary
  Tier 2: Edge TTS             — unlimited, always available
  Tier 3: ElevenLabs           — rate-limited, last resort

Fixes from previous version:
- Edge TTS emotion rate now stacks additively on top of self.edge_rate
  (parses the % value, adds the emotion offset, reconstructs the string)
  instead of replacing it entirely.
- tier_callback: optional callable(str) fired with "Tier 1/2/3" so GUI
  can display which tier is active.
- _schedule_midnight_reset(): sets a threading.Timer to auto-reset daily
  limit flags at midnight. Called once at init.
"""

import asyncio
import datetime
import os
import subprocess
import tempfile
import threading
from typing import Callable, Optional

import edge_tts
from groq import Groq

try:
    from elevenlabs import ElevenLabs as _ElevenLabsClient
    _ELEVENLABS_IMPORTABLE = True
except ImportError:
    _ElevenLabsClient      = None
    _ELEVENLABS_IMPORTABLE = False


# ── Emotion mappings ───────────────────────────────────────────────────────────

# Edge TTS: delta % added to whatever self.edge_rate is configured as.
_EDGE_EMOTION_DELTA = {
    "happy":     {"rate_delta": +15, "pitch": "+5Hz"},
    "excited":   {"rate_delta": +25, "pitch": "+10Hz"},
    "sad":       {"rate_delta": -15, "pitch": "-5Hz"},
    "curious":   {"rate_delta":  +5, "pitch": "+3Hz"},
    "surprised": {"rate_delta": +10, "pitch": "+8Hz"},
    "neutral":   {"rate_delta":   0, "pitch": "+0Hz"},
}

# ElevenLabs
_EL_EMOTION_MAP = {
    "happy":     {"stability": 0.40, "style": 0.70},
    "excited":   {"stability": 0.30, "style": 0.90},
    "sad":       {"stability": 0.80, "style": 0.20},
    "curious":   {"stability": 0.50, "style": 0.50},
    "surprised": {"stability": 0.30, "style": 0.70},
    "neutral":   {"stability": 0.60, "style": 0.30},
}


def _parse_rate_pct(rate_str: str) -> int:
    """Parse '+15%' or '-5%' → int (±15 or ±5). Returns 0 on failure."""
    try:
        return int(rate_str.replace("%", "").replace("+", ""))
    except (ValueError, AttributeError):
        return 0


def _make_rate_str(pct: int) -> str:
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


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
        tier_callback:       Optional[Callable[[str], None]] = None,
    ):
        # Tier 1: Groq
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_voice  = groq_voice
        self.groq_model  = groq_model

        # Tier 2: Edge TTS
        self.edge_voice     = edge_voice
        self.edge_rate      = edge_rate
        self._edge_rate_pct = _parse_rate_pct(edge_rate)

        # Tier 3: ElevenLabs
        self._el_client          = None
        self.elevenlabs_enabled  = False
        self.elevenlabs_voice_id = elevenlabs_voice_id

        if elevenlabs_api_key:
            if not _ELEVENLABS_IMPORTABLE:
                print("⚠️  ElevenLabs key provided but package not installed.")
            else:
                try:
                    self._el_client         = _ElevenLabsClient(api_key=elevenlabs_api_key)
                    self.elevenlabs_enabled = True
                except Exception as e:
                    print(f"⚠️  ElevenLabs init failed: {e}")

        self._groq_daily_limit_hit       = False
        self._elevenlabs_daily_limit_hit = False

        # Optional GUI callback: called with "Tier 1", "Tier 2", or "Tier 3"
        self._tier_callback = tier_callback

        # Schedule auto-reset of daily limit flags at midnight
        self._schedule_midnight_reset()

        print("🔊 Hybrid TTS ready (3-tier):")
        print(f"   Tier 1: Groq Orpheus ({groq_voice})")
        print(f"   Tier 2: Edge TTS     ({edge_voice}) [unlimited]")
        el_status = f"({elevenlabs_voice_id})" if self.elevenlabs_enabled else "(disabled)"
        print(f"   Tier 3: ElevenLabs   {el_status}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def speak(
        self,
        text:        str,
        output_file: Optional[str] = None,
        emotion:     Optional[str] = None,
    ) -> Optional[str]:
        # Tier 1 — Groq Orpheus
        if not self._groq_daily_limit_hit:
            if output_file is None:
                fd, output_file = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
            if self._speak_groq(text, output_file, emotion):
                if self._valid_audio(output_file):
                    self._notify_tier("Tier 1 (Groq)")
                    return output_file
                print("⚠️  Groq wrote empty/bad file — falling to Tier 2")
            self._cleanup(output_file)

        # Tier 2 — Edge TTS
        fd, edge_file = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        if self._speak_edge(text, edge_file, emotion):
            if self._valid_audio(edge_file):
                self._notify_tier("Tier 2 (Edge)")
                return edge_file
            print("⚠️  Edge TTS wrote empty/bad file — falling to Tier 3")
        self._cleanup(edge_file)

        # Tier 3 — ElevenLabs
        if self.elevenlabs_enabled and not self._elevenlabs_daily_limit_hit:
            fd, el_file = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            if self._speak_elevenlabs(text, el_file, emotion):
                if self._valid_audio(el_file):
                    self._notify_tier("Tier 3 (ElevenLabs)")
                    return el_file
                print("⚠️  ElevenLabs wrote empty/bad file")
            self._cleanup(el_file)

        return None

    def play_audio(self, audio_file: str):
        for player, args in [
            ("aplay",   [audio_file]),
            ("mpg123",  [audio_file]),
            ("ffplay",  ["-nodisp", "-autoexit", "-loglevel", "quiet", audio_file]),
            ("paplay",  [audio_file]),
        ]:
            try:
                subprocess.run(
                    [player] + args,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        print("⚠️  No audio player found (install aplay, mpg123, or ffplay)")

    def speak_and_play(self, text: str, emotion: Optional[str] = None) -> bool:
        path = self.speak(text, emotion=emotion)
        if not path:
            return False
        self.play_audio(path)
        self._cleanup(path)
        return True

    def reset_daily_limits(self):
        self._groq_daily_limit_hit       = False
        self._elevenlabs_daily_limit_hit = False
        print("🔄 TTS daily limit flags reset")
        # Schedule next reset
        self._schedule_midnight_reset()

    # ── Midnight reset scheduler ───────────────────────────────────────────────

    def _schedule_midnight_reset(self):
        """Schedule reset_daily_limits() to fire at the next midnight."""
        now      = datetime.datetime.now()
        midnight = (now + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        seconds_until = (midnight - now).total_seconds()
        t = threading.Timer(seconds_until, self.reset_daily_limits)
        t.daemon = True
        t.start()

    # ── Tier implementations ───────────────────────────────────────────────────

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
                print("⏳ Groq TTS rate-limited → Tier 2")
                self._groq_daily_limit_hit = True
            else:
                print(f"⚠️  Groq TTS error: {e}")
            return False

    def _speak_elevenlabs(self, text: str, output_file: str,
                          emotion: Optional[str] = None) -> bool:
        if not self.elevenlabs_enabled or self._elevenlabs_daily_limit_hit:
            return False
        try:
            voice_settings = None
            if emotion:
                settings = _EL_EMOTION_MAP.get(emotion)
                if settings:
                    try:
                        from elevenlabs import VoiceSettings
                        voice_settings = VoiceSettings(
                            stability        = settings["stability"],
                            similarity_boost = 0.80,
                            style            = settings["style"],
                        )
                    except ImportError:
                        pass

            kwargs = dict(
                voice_id      = self.elevenlabs_voice_id,
                text          = text,
                model_id      = "eleven_turbo_v2_5",
                output_format = "mp3_44100_128",
            )
            if voice_settings:
                kwargs["voice_settings"] = voice_settings

            with open(output_file, "wb") as f:
                for chunk in self._el_client.text_to_speech.convert(**kwargs):
                    if chunk:
                        f.write(chunk)

            emotion_tag = f" [{emotion}]" if emotion else ""
            print(f"🎤 Tier 3: ElevenLabs TTS{emotion_tag}")
            return True
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("quota", "limit", "429", "subscription")):
                print("⚠️  ElevenLabs limit/auth — marking exhausted")
                self._elevenlabs_daily_limit_hit = True
            else:
                print(f"⚠️  ElevenLabs error: {e}")
            return False

    async def _speak_edge_async(self, text: str, output_file: str,
                                emotion: Optional[str] = None) -> bool:
        try:
            # Stack emotion delta on top of base rate — not replace
            base_pct = self._edge_rate_pct
            if emotion and emotion in _EDGE_EMOTION_DELTA:
                params    = _EDGE_EMOTION_DELTA[emotion]
                final_pct = base_pct + params["rate_delta"]
                pitch     = params["pitch"]
            else:
                final_pct = base_pct
                pitch     = "+0Hz"

            rate = _make_rate_str(final_pct)
            comm = edge_tts.Communicate(text, self.edge_voice, rate=rate, pitch=pitch)
            await comm.save(output_file)

            emotion_tag = f" [{emotion}]" if emotion else ""
            print(f"🎤 Tier 2: Edge TTS{emotion_tag} (rate={rate}, pitch={pitch})")
            return True
        except Exception as e:
            print(f"❌ Edge TTS error: {e}")
            return False

    def _speak_edge(self, text: str, output_file: str,
                    emotion: Optional[str] = None) -> bool:
        try:
            return asyncio.run(self._speak_edge_async(text, output_file, emotion))
        except Exception as e:
            print(f"❌ Edge TTS runner error: {e}")
            return False

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _valid_audio(path: str) -> bool:
        try:
            return os.path.exists(path) and os.path.getsize(path) > 0
        except OSError:
            return False

    @staticmethod
    def _cleanup(path: str):
        try:
            os.remove(path)
        except OSError:
            pass

    def _notify_tier(self, label: str):
        if self._tier_callback:
            try:
                self._tier_callback(label)
            except Exception:
                pass


# Backward-compat alias
TTSHandler = HybridTTSHandler