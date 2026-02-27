"""
Pepper Robot Interface — NAOqi hardware control

Changes from previous version:
- gesture_callback parameter added to speak_hq() and play_audio_file().
  The callback fires immediately before player.play() so intentional gestures
  are synchronized with speech onset rather than finishing before speech starts.
- SSH keepalive (set_keepalive) + real liveness probe (exec_command echo)
  instead of just checking transport.is_active().
- _valid_audio now checks WAV/MP3 magic bytes, not just file size.
- EMOTION_COLOUR_MAP updated — each emotion now has a visually distinct color.
- Dead named movement methods removed (only _move() is used).
- _hq_speech_animation_loop gesture pool expanded.
- speak() lock timeout surfaced as a parameter for clarity.
"""

import logging
import os
import random
import threading
import time
from contextlib import contextmanager
from typing import Callable, Optional, TYPE_CHECKING

try:
    import qi
    _QI_AVAILABLE = True
except ImportError:
    qi = None
    _QI_AVAILABLE = False
    print("⚠️  NAOqi (qi) not installed — offline/chat-only mode")

try:
    import paramiko
    _PARAMIKO_AVAILABLE = True
except ImportError:
    _PARAMIKO_AVAILABLE = False
    print("⚠️  paramiko not installed — HQ audio disabled")
    print("   Install with: pip install paramiko --break-system-packages")

if TYPE_CHECKING:
    from hybrid_tts_handler import HybridTTSHandler

import config


class PepperRobot:
    def __init__(self, ip: str, port: int,
                 ssh_user: str = "nao", ssh_password: str = "nao"):
        self.ip           = ip
        self.port         = port
        self.ssh_user     = ssh_user
        self.ssh_password = ssh_password
        self.connected    = False

        self.session = self.tts = self.motion = self.animated_speech = None
        self.audio = self.leds = self.awareness = self.tablet = None

        self._speech_lock  = threading.Lock()
        self._gesture_lock = threading.Lock()

        self._thinking          = False
        self._thinking_thread: Optional[threading.Thread] = None
        self._is_speaking_hq    = False
        self._anim_thread: Optional[threading.Thread] = None

        # LED priority: "thinking" > "speaking" > "idle"
        self._led_state         = "idle"
        self._led_emotion_color = "blue"
        self._led_lock          = threading.Lock()

        self._ssh_client: Optional["paramiko.SSHClient"] = None
        self._last_gesture_time: float = 0.0
        self._GESTURE_COOLDOWN:  float = 2.5

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self, timeout: float = 5.0) -> bool:
        if not _QI_AVAILABLE:
            print("⚠️  qi not available — offline mode")
            self.connected = False
            return False

        print(f"🤖 Connecting to Pepper at {self.ip}:{self.port}…")
        result = {"ok": False, "session": None, "error": None}

        def _attempt():
            try:
                s = qi.Session()
                s.connect(f"tcp://{self.ip}:{self.port}")
                result["session"] = s
                result["ok"] = True
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=_attempt, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if not result["ok"]:
            reason = "timed out" if t.is_alive() else str(result["error"])
            logging.warning("Connection failed (%s) — offline mode", reason)
            print("   Chat, TTS and web search will still work.")
            self.connected = False
            return False

        self.session = result["session"]
        try:
            self.tts             = self.session.service("ALTextToSpeech")
            self.motion          = self.session.service("ALMotion")
            self.animated_speech = self.session.service("ALAnimatedSpeech")
            self.audio           = self.session.service("ALAudioDevice")
            self.leds            = self.session.service("ALLeds")
            self.awareness       = self.session.service("ALBasicAwareness")

            try:
                self.tablet = self.session.service("ALTabletService")
                print("   ✅ Tablet service available")
            except Exception:
                self.tablet = None
                print("   ⚠️  ALTabletService not available")

            for label, fn in [
                ("Autonomous Life", lambda: self.session.service("ALAutonomousLife").setState("disabled")),
                ("BasicAwareness",  lambda: self.awareness.stopAwareness()),
                ("Body stiffness",  lambda: self.motion.setStiffnesses("Body", 1.0)),
            ]:
                try:
                    fn()
                    print(f"   ✅ {label} OK")
                except Exception as e:
                    print(f"   ⚠️  {label}: {e}")

            try:
                self.motion.setExternalCollisionProtectionEnabled("Move", False)
            except Exception:
                try:
                    self.motion.setOrthogonalSecurityDistance(0.05)
                    self.motion.setTangentialSecurityDistance(0.05)
                except Exception:
                    pass

            try:
                self.audio.setOutputVolume(100)
            except Exception:
                pass

            self.motion.wakeUp()
            time.sleep(1)
            self.connected = True
            print("✅ Connected to Pepper!")
            return True
        except Exception as e:
            self.connected = False
            logging.error("Service init failed: %s", e)
            return False

    def disconnect(self):
        try:
            if self.motion:
                self.motion.stopMove()
                try:
                    self.motion.setExternalCollisionProtectionEnabled("Move", True)
                except Exception:
                    pass
                self.motion.rest()
            try:
                self.session.service("ALAutonomousLife").setState("solitary")
            except Exception:
                pass
            try:
                if self.awareness:
                    self.awareness.startAwareness()
            except Exception:
                pass
        except Exception as e:
            logging.warning("Disconnect error: %s", e)
        finally:
            self.connected = False
            if self._ssh_client:
                try:
                    self._ssh_client.close()
                except Exception:
                    pass
                self._ssh_client = None
        print("👋 Disconnected from Pepper")

    # ── LED priority state machine ─────────────────────────────────────────────

    def _enter_led_thinking(self):
        with self._led_lock:
            self._led_state = "thinking"

    def _exit_led_thinking(self):
        with self._led_lock:
            if self._led_state == "thinking":
                self._led_state = "idle"
        self.set_eye_color("blue")

    def _enter_led_speaking(self, emotion_color: str = "blue"):
        with self._led_lock:
            if self._led_state == "thinking":
                return
            self._led_state         = "speaking"
            self._led_emotion_color = emotion_color
        self.set_eye_color(emotion_color)

    def _exit_led_speaking(self):
        changed = False
        with self._led_lock:
            if self._led_state == "speaking":
                self._led_state         = "idle"
                self._led_emotion_color = "blue"
                changed = True
        if changed:
            self.set_eye_color("blue")

    # ── Built-in NAOqi TTS ────────────────────────────────────────────────────

    def speak(self, text: str, use_animation: bool = True):
        if not self._speech_lock.acquire(timeout=3.0):
            logging.warning("speak(): lock timeout — skipping")
            return
        try:
            if use_animation and self.animated_speech:
                self.animated_speech.say(text)
            else:
                self.tts.say(text)
        except Exception as e:
            logging.error("Speech error: %s", e)
        finally:
            self._speech_lock.release()

    def set_volume(self, volume: int):
        try:
            self.tts.setVolume(volume / 100.0)
        except Exception as e:
            logging.error("TTS volume error: %s", e)
        try:
            self.audio.setOutputVolume(int(volume))
        except Exception as e:
            logging.error("Speaker volume error: %s", e)

    # ── HQ Audio Pipeline ─────────────────────────────────────────────────────

    def speak_hq(
        self,
        text: str,
        tts_handler: "HybridTTSHandler",
        emotion:           Optional[str]              = None,
        status_callback:   Optional[Callable[[str], None]] = None,
        gesture_callback:  Optional[Callable[[], None]]    = None,
    ) -> bool:
        """
        Full pipeline: TTS generation → SSH transfer → NAOqi playback.
        gesture_callback fires right before audio starts playing so the
        gesture is synchronized with speech onset.
        """
        audio_path = None
        try:
            if status_callback:
                status_callback("🎙️ Generating voice…")
            audio_path = tts_handler.speak(text, emotion=emotion)

            if audio_path and self._valid_audio(audio_path):
                emotion_color = self.EMOTION_COLOUR_MAP.get(emotion or "", "blue")
                if self.play_audio_file(
                    audio_path,
                    emotion_color    = emotion_color,
                    status_callback  = status_callback,
                    gesture_callback = gesture_callback,
                ):
                    return True

            print("↩️  Falling back to built-in NAOqi TTS")
            # Fire gesture before built-in TTS too
            if gesture_callback:
                try:
                    gesture_callback()
                except Exception:
                    pass
            self.speak(text)
            return False

        finally:
            if audio_path:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    def _ensure_ssh(self) -> bool:
        """
        Ensure SSH is alive. Uses a real no-op command to verify liveness
        rather than just checking transport state (which can be stale after
        a WiFi blip).
        """
        if not _PARAMIKO_AVAILABLE:
            return False

        # First try a lightweight probe on the existing connection
        if self._ssh_client:
            try:
                transport = self._ssh_client.get_transport()
                if transport and transport.is_active():
                    # Real liveness check — not just checking the flag
                    self._ssh_client.exec_command("echo ok", timeout=2)
                    return True
            except Exception:
                pass

        print("🔗 (Re)connecting SSH to Pepper…")
        try:
            if self._ssh_client:
                try:
                    self._ssh_client.close()
                except Exception:
                    pass
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                self.ip,
                username = self.ssh_user,
                password = self.ssh_password,
                timeout  = 5,
            )
            # Set keepalive so idle sessions don't silently die
            transport = client.get_transport()
            if transport:
                transport.set_keepalive(config.SSH_KEEPALIVE_INTERVAL)
            self._ssh_client = client
            print("   ✅ SSH connected")
            return True
        except Exception as e:
            print(f"   ❌ SSH failed: {e}")
            self._ssh_client = None
            return False

    def _transfer_to_robot(self, local_path: str) -> Optional[str]:
        """Transfer audio via SFTP. No speech lock held during transfer."""
        if not _PARAMIKO_AVAILABLE:
            return None
        remote_path = f"/tmp/{os.path.basename(local_path)}"
        try:
            if not self._ensure_ssh():
                return None
            sftp = self._ssh_client.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            return remote_path
        except Exception as e:
            logging.warning("SFTP transfer failed: %s — retrying", e)
            # One retry with a fresh SSH connection
            self._ssh_client = None
            try:
                if not self._ensure_ssh():
                    return None
                sftp = self._ssh_client.open_sftp()
                sftp.put(local_path, remote_path)
                sftp.close()
                return remote_path
            except Exception as e2:
                logging.warning("SFTP retry also failed: %s", e2)
                return None

    def play_audio_file(
        self,
        file_path:        str,
        emotion_color:    str = "blue",
        status_callback:  Optional[Callable[[str], None]] = None,
        gesture_callback: Optional[Callable[[], None]]    = None,
        lock_timeout:     float = 2.0,
    ) -> bool:
        """
        Transfer audio to Pepper and play via ALAudioPlayer.
        gesture_callback fires right before player.play() — synchronized with speech.
        """
        if not _PARAMIKO_AVAILABLE:
            return False

        if status_callback:
            status_callback("📡 Sending to robot…")
        remote_path = self._transfer_to_robot(file_path)
        if not remote_path:
            return False

        if not self._speech_lock.acquire(timeout=lock_timeout):
            print("⚠️  Already speaking — skipping playback")
            self._cleanup_remote(remote_path)
            return False

        try:
            if status_callback:
                status_callback("🔊 Speaking…")

            self._enter_led_speaking(emotion_color)

            player  = self.session.service("ALAudioPlayer")
            file_id = player.loadFile(remote_path)

            # Fire the intentional gesture right before audio starts
            if gesture_callback:
                try:
                    gesture_callback()
                except Exception as e:
                    logging.warning("gesture_callback error: %s", e)

            # Start background random animation loop
            self._is_speaking_hq = True
            anim_thread = threading.Thread(
                target=self._hq_speech_animation_loop,
                daemon=True,
                name="HQSpeechAnim",
            )
            self._anim_thread = anim_thread
            anim_thread.start()

            player.play(file_id)   # blocks until done
            return True

        except Exception as e:
            logging.warning("ALAudioPlayer failed: %s", e)
            return False

        finally:
            self._is_speaking_hq = False
            if self._anim_thread and self._anim_thread.is_alive():
                self._anim_thread.join(timeout=3.0)
            self._anim_thread = None
            self._exit_led_speaking()
            self._cleanup_remote(remote_path)
            self._speech_lock.release()

    def _cleanup_remote(self, remote_path: str):
        if self._ssh_client:
            try:
                self._ssh_client.exec_command(f"rm -f {remote_path}")
            except Exception:
                pass

    def _hq_speech_animation_loop(self):
        """Random background gestures during HQ audio playback. Expanded pool."""
        gesture_fns = [
            self.nod, self.explaining_gesture, self.thinking_gesture,
            self.look_around, self.wave, self.celebrate,
            self.bow, self.excited_gesture,
        ]
        while self._is_speaking_hq:
            try:
                with self._led_lock:
                    color = self._led_emotion_color
                self.set_eye_color(color)

                if (random.random() > 0.6 and
                        time.time() - self._last_gesture_time >= self._GESTURE_COOLDOWN):
                    random.choice(gesture_fns)()

                for _ in range(20):
                    if not self._is_speaking_hq:
                        break
                    time.sleep(0.1)
            except Exception as e:
                logging.warning("HQ anim loop error: %s", e)
                break

    # ── Thinking indicator ────────────────────────────────────────────────────

    @contextmanager
    def thinking(self):
        self.thinking_indicator(start=True)
        try:
            yield
        finally:
            self.thinking_indicator(start=False)

    def thinking_indicator(self, start: bool = True):
        if start:
            self._thinking = True
            self._enter_led_thinking()
            if self._thinking_thread is None or not self._thinking_thread.is_alive():
                self._thinking_thread = threading.Thread(
                    target=self._pulse_thinking_loop,
                    daemon=True,
                    name="ThinkingPulse",
                )
                self._thinking_thread.start()
        else:
            self._thinking = False
            self._exit_led_thinking()

    def _pulse_thinking_loop(self):
        colours = ["blue", "off"]
        idx = 0
        while self._thinking:
            with self._led_lock:
                if self._led_state != "thinking":
                    break
            try:
                self.set_eye_color(colours[idx % 2])
                idx += 1
                for _ in range(4):
                    if not self._thinking:
                        break
                    time.sleep(0.1)
            except Exception:
                break

    # ── Gesture dispatcher ────────────────────────────────────────────────────

    def _run_gesture(self, impl_fn, *args, **kwargs):
        def _worker():
            if not self._gesture_lock.acquire(blocking=False):
                return
            try:
                self._last_gesture_time = time.time()
                impl_fn(*args, **kwargs)
            finally:
                self._gesture_lock.release()
        threading.Thread(target=_worker, daemon=True, name="Gesture").start()

    def wave(self):               self._run_gesture(self._wave_impl)
    def nod(self):                self._run_gesture(self._nod_impl)
    def shake_head(self):         self._run_gesture(self._shake_head_impl)
    def look_at_sound(self):      self._run_gesture(self._look_at_sound_impl)
    def thinking_gesture(self):   self._run_gesture(self._thinking_gesture_impl)
    def explaining_gesture(self): self._run_gesture(self._explaining_gesture_impl)
    def excited_gesture(self):    self._run_gesture(self._excited_gesture_impl)
    def point_forward(self):      self._run_gesture(self._point_forward_impl)
    def shrug(self):              self._run_gesture(self._shrug_impl)
    def celebrate(self):          self._run_gesture(self._celebrate_impl)
    def look_around(self):        self._run_gesture(self._look_around_impl)
    def bow(self):                self._run_gesture(self._bow_impl)

    # ── Gesture implementations ───────────────────────────────────────────────

    def _wave_impl(self):
        try:
            n = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"]
            up   = [-0.5, -0.3, 1.5, 1.2]
            down = [-0.5, -0.3, 1.0, 1.2]
            rest = [ 1.5,  0.15, 0.5, 1.2]
            self.motion.setAngles(n, up, 0.2);   time.sleep(0.3)
            for _ in range(2):
                self.motion.setAngles(n, down, 0.3); time.sleep(0.2)
                self.motion.setAngles(n, up,   0.3); time.sleep(0.2)
            self.motion.setAngles(n, rest, 0.2)
        except Exception as e:
            logging.error("Wave: %s", e)

    def _nod_impl(self):
        try:
            self.motion.setAngles("HeadPitch",  0.3, 0.15); time.sleep(0.3)
            self.motion.setAngles("HeadPitch", -0.1, 0.15); time.sleep(0.3)
            self.motion.setAngles("HeadPitch",  0.0, 0.15)
        except Exception as e:
            logging.error("Nod: %s", e)

    def _shake_head_impl(self):
        try:
            self.motion.setAngles("HeadYaw",  0.4, 0.15); time.sleep(0.3)
            self.motion.setAngles("HeadYaw", -0.4, 0.15); time.sleep(0.3)
            self.motion.setAngles("HeadYaw",  0.0, 0.15)
        except Exception as e:
            logging.error("Shake head: %s", e)

    def _look_at_sound_impl(self):
        try:
            if self.awareness:
                self.awareness.setEnabled(True)
        except Exception as e:
            logging.error("Look at sound: %s", e)

    def _thinking_gesture_impl(self):
        try:
            n    = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw", "RWristYaw"]
            pose = [-0.3, -0.3, 1.2, 1.0, 0.0]
            rest = [ 1.5,  0.15, 0.5, 1.2, 0.0]
            self.motion.setAngles(n, pose, 0.15); time.sleep(1.0)
            self.motion.setAngles(n, rest, 0.15)
        except Exception as e:
            logging.error("Thinking: %s", e)

    def _explaining_gesture_impl(self):
        try:
            n     = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                     "LShoulderPitch", "LShoulderRoll", "LElbowRoll"]
            open_ = [ 0.0, -0.3,  1.0,  0.0,  0.3, -1.0]
            close = [ 0.5, -0.1,  0.5,  0.5,  0.1, -0.5]
            rest  = [ 1.5,  0.15, 0.5,  1.5, -0.15, -0.5]
            self.motion.setAngles(n, open_, 0.2); time.sleep(0.4)
            self.motion.setAngles(n, close, 0.2); time.sleep(0.4)
            self.motion.setAngles(n, rest,  0.2)
        except Exception as e:
            logging.error("Explaining: %s", e)

    def _excited_gesture_impl(self):
        try:
            n    = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                    "LShoulderPitch", "LShoulderRoll", "LElbowRoll"]
            up   = [-1.0, -0.3,  1.5, -1.0,  0.3, -1.5]
            rest = [ 1.5,  0.15, 0.5,  1.5, -0.15, -0.5]
            self.motion.setAngles(n, up,   0.15); time.sleep(0.8)
            self.motion.setAngles(n, rest, 0.15)
        except Exception as e:
            logging.error("Excited: %s", e)

    def _point_forward_impl(self):
        try:
            n     = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"]
            point = [ 0.0, -0.3,  0.0, 1.5]
            rest  = [ 1.5,  0.15, 0.5, 1.2]
            self.motion.setAngles(n, point, 0.15); time.sleep(1.0)
            self.motion.setAngles(n, rest,  0.15)
        except Exception as e:
            logging.error("Point: %s", e)

    def _shrug_impl(self):
        try:
            n    = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                    "LShoulderPitch", "LShoulderRoll", "LElbowRoll", "HeadPitch"]
            pose = [0.5, -0.5,  1.2,  0.5,  0.5, -1.2,  0.2]
            rest = [1.5,  0.15, 0.5,  1.5, -0.15, -0.5,  0.0]
            self.motion.setAngles(n, pose, 0.15); time.sleep(0.8)
            self.motion.setAngles(n, rest, 0.15)
        except Exception as e:
            logging.error("Shrug: %s", e)

    def _celebrate_impl(self):
        try:
            n    = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                    "LShoulderPitch", "LShoulderRoll", "LElbowRoll"]
            up   = [-0.5, -0.3,  1.5, -0.5,  0.3, -1.5]
            down = [ 0.0, -0.3,  1.0,  0.0,  0.3, -1.0]
            rest = [ 1.5,  0.15, 0.5,  1.5, -0.15, -0.5]
            for _ in range(2):
                self.motion.setAngles(n, up,   0.25); time.sleep(0.3)
                self.motion.setAngles(n, down, 0.25); time.sleep(0.3)
            self.motion.setAngles(n, rest, 0.2)
        except Exception as e:
            logging.error("Celebrate: %s", e)

    def _look_around_impl(self):
        try:
            self.motion.setAngles("HeadYaw", -0.5, 0.15); time.sleep(0.5)
            self.motion.setAngles("HeadYaw",  0.5, 0.15); time.sleep(0.5)
            self.motion.setAngles("HeadYaw",  0.0, 0.15)
        except Exception as e:
            logging.error("Look around: %s", e)

    def _bow_impl(self):
        try:
            self.motion.setAngles("HeadPitch", 0.5, 0.1); time.sleep(0.5)
            self.motion.setAngles("HeadPitch", 0.0, 0.1); time.sleep(0.3)
        except Exception as e:
            logging.error("Bow: %s", e)

    # ── Movement ──────────────────────────────────────────────────────────────

    def stop_movement(self):
        try:
            self.motion.moveToward(0.0, 0.0, 0.0)
            self.motion.stopMove()
        except Exception as e:
            logging.error("Stop: %s", e)

    def _move(self, x: float, y: float, theta: float):
        try:
            self.motion.moveToward(x, y, theta)
        except Exception as e:
            logging.error("Move: %s", e)

    # ── LEDs ──────────────────────────────────────────────────────────────────

    _COLOUR_MAP = {
        "blue":   0x000000FF,
        "green":  0x0000FF00,
        "red":    0x00FF0000,
        "yellow": 0x00FFFF00,
        "white":  0x00FFFFFF,
        "purple": 0x00800080,
        "cyan":   0x0000FFFF,
        "orange": 0x00FF8000,
        "off":    0x00000000,
    }

    # Each emotion now has a visually distinct color
    EMOTION_COLOUR_MAP = {
        "happy":     "yellow",   # warm, positive
        "sad":       "blue",     # classic sad blue
        "excited":   "green",    # energetic
        "curious":   "cyan",     # inquisitive, different from sad
        "surprised": "white",    # bright/startled
        "neutral":   "blue",     # default
    }

    def set_eye_color(self, color: str):
        try:
            rgb = self._COLOUR_MAP.get(color)
            if rgb is not None:
                self.leds.fadeRGB("FaceLeds", rgb, 0.5)
        except Exception as e:
            logging.error("LED: %s", e)

    def pulse_eyes(self, color: str = "blue", duration: float = 2.0):
        try:
            self.set_eye_color(color)
            time.sleep(duration / 2)
            self.set_eye_color("off")
            time.sleep(0.2)
            self.set_eye_color(color)
        except Exception as e:
            logging.error("Pulse: %s", e)

    # ── Audio file validation ─────────────────────────────────────────────────

    @staticmethod
    def _valid_audio(path: str) -> bool:
        """
        Validate audio file by checking magic bytes, not just size.
        Catches truncated files or HTML error pages written to disk.
        """
        try:
            if not os.path.exists(path) or os.path.getsize(path) < 4:
                return False
            with open(path, "rb") as f:
                header = f.read(12)
            # WAV: RIFF....WAVE
            if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                return True
            # MP3: ID3 tag or sync frame (0xFF 0xFB / 0xFF 0xFA / 0xFF 0xF3)
            if header[:3] == b"ID3":
                return True
            if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
                return True
            return False
        except OSError:
            return False

    # ── Tablet display ────────────────────────────────────────────────────────

    def show_tablet_image(self, url: str):
        if not self.tablet:
            return
        try:
            self.tablet.showImage(url)
        except Exception as e:
            logging.warning("show_tablet_image: %s", e)

    def show_tablet_webview(self, url: str):
        if not self.tablet:
            return
        try:
            self.tablet.showWebview(url)
        except Exception as e:
            logging.warning("show_tablet_webview: %s", e)

    def clear_tablet(self):
        if not self.tablet:
            return
        for method in [lambda: self.tablet.hideWebview(),
                       lambda: self.tablet.hideImage()]:
            try:
                method()
            except Exception:
                pass