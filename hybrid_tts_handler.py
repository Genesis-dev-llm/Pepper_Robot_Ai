"""
Hybrid TTS Handler - 3-Tier System
1. Groq TTS (Orpheus) - PRIMARY (fast, 140 chars/sec)
2. ElevenLabs - SECONDARY (best quality, natural)
3. Edge TTS - FALLBACK (unlimited, reliable)

Strategy:
- Try Groq first (fastest)
- On Groq rate limit → try ElevenLabs (best quality)
- On ElevenLabs rate limit or not configured → Edge TTS (unlimited)
- Per-minute limits: wait and retry
- Daily limits: skip to next tier for session
"""

import time
from typing import Optional
from groq import Groq
import edge_tts
import asyncio
import tempfile
import os
import subprocess

# ElevenLabs is optional
try:
    from elevenlabs import generate, set_api_key, save
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

class HybridTTSHandler:
    def __init__(self, groq_api_key: str, 
                 groq_voice: str = "hannah", 
                 groq_model: str = "canopylabs/orpheus-v1-english",
                 elevenlabs_api_key: Optional[str] = None,
                 elevenlabs_voice: str = "Rachel",
                 edge_voice: str = "en-US-AriaNeural",
                 edge_rate: str = "+0%"):
        """
        Initialize hybrid TTS with 3-tier fallback
        
        Args:
            groq_api_key: Groq API key (required)
            groq_voice: Groq voice (hannah, austin, troy)
            groq_model: Groq TTS model
            elevenlabs_api_key: ElevenLabs API key (optional)
            elevenlabs_voice: ElevenLabs voice ID or name
            edge_voice: Edge TTS voice for fallback
            edge_rate: Edge TTS speech rate
        """
        # Groq setup
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_voice = groq_voice
        self.groq_model = groq_model
        
        # ElevenLabs setup
        self.elevenlabs_enabled = False
        if elevenlabs_api_key and ELEVENLABS_AVAILABLE:
            try:
                set_api_key(elevenlabs_api_key)
                self.elevenlabs_voice = elevenlabs_voice
                self.elevenlabs_enabled = True
            except Exception as e:
                print(f"⚠️ ElevenLabs setup failed: {e}")
        elif elevenlabs_api_key and not ELEVENLABS_AVAILABLE:
            print("⚠️ ElevenLabs API key provided but package not installed")
            print("   Install with: pip install elevenlabs")
        
        # Edge TTS setup
        self.edge_voice = edge_voice
        self.edge_rate = edge_rate
        
        # State tracking
        self.groq_daily_limit_hit = False
        self.elevenlabs_daily_limit_hit = False
        
        # Print status
        print(f"🔊 Hybrid TTS initialized (3-tier)")
        print(f"   Tier 1: Groq Orpheus ({groq_voice})")
        if self.elevenlabs_enabled:
            print(f"   Tier 2: ElevenLabs ({elevenlabs_voice})")
        else:
            print(f"   Tier 2: ElevenLabs (disabled - no API key)")
        print(f"   Tier 3: Edge TTS ({edge_voice}) [unlimited]")
    
    def _speak_groq(self, text: str, output_file: str, emotion: Optional[str] = None) -> bool:
        """Try to generate speech with Groq"""
        if self.groq_daily_limit_hit:
            return False
            
        try:
            # Add emotion tag if specified
            if emotion:
                text = f"[{emotion}] {text}"
            
            # Generate speech
            response = self.groq_client.audio.speech.create(
                model=self.groq_model,
                voice=self.groq_voice,
                input=text,
                response_format="wav"
            )
            
            # Save to file
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            print("🎤 Used: Groq TTS")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for rate limit errors
            if '429' in error_msg or 'rate limit' in error_msg:
                print("⏳ Groq rate limit hit → falling back to next tier immediately")
                return False
            else:
                print(f"⚠️ Groq TTS error: {e}")
                return False
    
    def _speak_elevenlabs(self, text: str, output_file: str) -> bool:
        """Try to generate speech with ElevenLabs"""
        if not self.elevenlabs_enabled or self.elevenlabs_daily_limit_hit:
            return False
        
        try:
            # Generate speech
            audio = generate(
                text=text,
                voice=self.elevenlabs_voice,
                model="eleven_monolingual_v1"
            )
            
            # Save to file
            save(audio, output_file)
            print("🎤 Used: ElevenLabs TTS")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for rate/quota limits
            if 'quota' in error_msg or 'limit' in error_msg or '429' in error_msg:
                print("⚠️ ElevenLabs limit reached → using Edge TTS")
                self.elevenlabs_daily_limit_hit = True
                return False
            else:
                print(f"⚠️ ElevenLabs error: {e}")
                return False
    
    async def _speak_edge_async(self, text: str, output_file: str) -> bool:
        """Generate speech with Edge TTS"""
        try:
            communicate = edge_tts.Communicate(text, self.edge_voice, rate=self.edge_rate)
            await communicate.save(output_file)
            print("🎤 Used: Edge TTS")
            return True
        except Exception as e:
            print(f"❌ Edge TTS error: {e}")
            return False
    
    def _speak_edge(self, text: str, output_file: str) -> bool:
        """Synchronous wrapper for Edge TTS"""
        return asyncio.run(self._speak_edge_async(text, output_file))
    
    def speak(self, text: str, output_file: Optional[str] = None, emotion: Optional[str] = None) -> Optional[str]:
        """
        Generate speech with 3-tier fallback: Groq → ElevenLabs → Edge
        
        Args:
            text: Text to speak
            output_file: Optional path to save audio
            emotion: Optional emotion for Groq (cheerful, sad, etc.)
        
        Returns:
            Path to generated audio file or None if error
        """
        # Create temp file if no output specified
        if output_file is None:
            # Use .wav for Groq/ElevenLabs, .mp3 for Edge
            fd, output_file = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
        
        # Tier 1: Try Groq (fastest)
        if not self.groq_daily_limit_hit:
            if self._speak_groq(text, output_file, emotion):
                return output_file
        
        # Tier 2: Try ElevenLabs (best quality)
        if self.elevenlabs_enabled and not self.elevenlabs_daily_limit_hit:
            # Convert to mp3 for ElevenLabs
            elevenlabs_file = output_file.replace('.wav', '.mp3')
            if self._speak_elevenlabs(text, elevenlabs_file):
                return elevenlabs_file
        
        # Tier 3: Edge TTS (unlimited fallback)
        # Convert to mp3 if not already
        edge_file = output_file.replace('.wav', '.mp3')
        if self._speak_edge(text, edge_file):
            return edge_file
        
        return None
    
    def play_audio(self, audio_file: str):
        """Play audio file using system audio player"""
        try:
            # Try multiple players
            players = ["aplay", "mpg123", "ffplay", "paplay"]
            
            for player in players:
                try:
                    if player == "ffplay":
                        subprocess.run(
                            [player, "-nodisp", "-autoexit", "-loglevel", "quiet", audio_file],
                            check=True
                        )
                    else:
                        subprocess.run([player, audio_file], check=True, 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            print("⚠️ No audio player found. Install aplay, mpg123, or ffplay")
            
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
    
    def speak_and_play(self, text: str, emotion: Optional[str] = None) -> bool:
        """
        Generate speech and play it immediately
        
        Args:
            text: Text to speak
            emotion: Optional emotion (only for Groq)
        
        Returns:
            True if successful, False otherwise
        """
        audio_file = self.speak(text, emotion=emotion)
        if audio_file:
            self.play_audio(audio_file)
            # Clean up temp file
            try:
                os.remove(audio_file)
            except:
                pass
            return True
        return False
    
    def reset_daily_limits(self):
        """Call this on restart to reset daily limit flags"""
        self.groq_daily_limit_hit = False
        self.elevenlabs_daily_limit_hit = False
        print("🔄 Daily limit flags reset, will try all tiers again")


# For backward compatibility
TTSHandler = HybridTTSHandler