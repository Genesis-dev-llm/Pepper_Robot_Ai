"""
Pepper Robot Interface — NAOqi hardware control

Changes from previous version:
- self.connected flag added so callers can guard all hardware calls gracefully
  (covers both "never connected" and "dropped mid-session" cases).
- qi import is now guarded — if NAOqi is not installed the module still loads
  and PepperRobot can be instantiated; connect() will simply return False and
  set connected=False, allowing the rest of the system to run in offline mode.
- _hq_speech_animation_loop now calls public gesture methods (nod, explaining_gesture,
  etc.) instead of _impl methods directly, so the gesture lock is always respected
  and user-triggered gestures can't fight the animation loop.
- play_audio_file reuses the original SSH connection for remote cleanup instead of
  opening a second connection, halving the auth handshakes on every audio play.
"""

import os
import random
import threading
import time
from typing import Optional, TYPE_CHECKING

try:
    import qi
    _QI_AVAILABLE = True
except ImportError:
    qi = None
    _QI_AVAILABLE = False
    print("⚠️  NAOqi (qi) not installed — Pepper hardware unavailable")
    print("   Running in offline/chat-only mode")

try:
    import paramiko
    _PARAMIKO_AVAILABLE = True
except ImportError:
    _PARAMIKO_AVAILABLE = False
    print("⚠️  paramiko not installed — HQ audio via ALAudioPlayer disabled")
    print("   Install with: pip install paramiko --break-system-packages")

if TYPE_CHECKING:
    from hybrid_tts_handler import HybridTTSHandler


class PepperRobot:
    def __init__(self, ip: str, port: int,
                 ssh_user: str = "nao", ssh_password: str = "nao"):
        self.ip           = ip
        self.port         = port
        self.ssh_user     = ssh_user
        self.ssh_password = ssh_password

        # Set to True only after a successful connect() call.
        # All callers should guard: if pepper.connected: pepper.wave()
        self.connected = False

        # NAOqi services (set in connect())
        self.session         = None
        self.tts             = None
        self.motion          = None
        self.animated_speech = None
        self.audio           = None
        self.leds            = None
        self.awareness       = None

        self._speech_lock  = threading.Lock()
        self._gesture_lock = threading.Lock()

        self._thinking        = False
        self._thinking_thread: Optional[threading.Thread] = None

        self._is_speaking_hq = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, timeout: float = 5.0) -> bool:
        if not _QI_AVAILABLE:
            print("⚠️  qi not available — cannot connect to Pepper (offline mode)")
            self.connected = False
            return False

        print(f"🤖 Connecting to Pepper at {self.ip}:{self.port} (timeout {timeout}s)…")

        # qi.Session.connect() blocks indefinitely on unreachable IPs.
        # Run it in a thread so we can enforce a timeout and fall through
        # to offline mode instead of hanging the entire startup.
        _result = {"success": False, "error": None}

        def _attempt():
            try:
                session = qi.Session()
                session.connect(f"tcp://{self.ip}:{self.port}")
                _result["session"] = session
                _result["success"] = True
            except Exception as e:
                _result["error"] = e

        t = threading.Thread(target=_attempt, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if not _result["success"]:
            if t.is_alive():
                print(f"⏱️  Connection timed out after {timeout}s — launching in offline mode")
            else:
                print(f"❌ Connection failed: {_result['error']} — launching in offline mode")
            print("   Chat, TTS and web search will still work.")
            print("   Robot controls and gestures will be silently ignored.")
            self.connected = False
            return False

        # Connection succeeded — grab the session and initialise services
        self.session = _result["session"]
        try:
            self.tts             = self.session.service("ALTextToSpeech")
            self.motion          = self.session.service("ALMotion")
            self.animated_speech = self.session.service("ALAnimatedSpeech")
            self.audio           = self.session.service("ALAudioDevice")
            self.leds            = self.session.service("ALLeds")
            self.awareness       = self.session.service("ALBasicAwareness")

            self.motion.wakeUp()
            time.sleep(1)
            self.connected = True
            print("✅ Connected to Pepper!")
            return True
        except Exception as e:
            self.connected = False
            print(f"❌ Connection failed: {e}")
            return False

    def disconnect(self):
        try:
            if self.motion:
                self.motion.rest()
            print("👋 Disconnected from Pepper")
        except Exception as e:
            print(f"⚠️  Disconnect error: {e}")
        finally:
            self.connected = False

    # ------------------------------------------------------------------
    # Speech — built-in NAOqi TTS
    # ------------------------------------------------------------------

    def speak(self, text: str, use_animation: bool = True):
        with self._speech_lock:
            try:
                if use_animation and self.animated_speech:
                    self.animated_speech.say(text)
                else:
                    self.tts.say(text)
            except Exception as e:
                print(f"❌ Speech error: {e}")

    def set_volume(self, volume: int):
        try:
            self.tts.setVolume(volume / 100.0)
        except Exception as e:
            print(f"❌ Volume error: {e}")

    # ------------------------------------------------------------------
    # Speech — HQ audio pipeline
    # ------------------------------------------------------------------

    def speak_hq(self, text: str, tts_handler: "HybridTTSHandler") -> bool:
        try:
            audio_path = tts_handler.speak(text)
            if audio_path and self.play_audio_file(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
                return True
        except Exception as e:
            print(f"⚠️  HQ TTS pipeline error: {e}")

        print("↩️  Falling back to built-in NAOqi TTS")
        self.speak(text)
        return False

    def play_audio_file(self, file_path: str, lock_timeout: float = 0.5) -> bool:
        """
        SCP audio to Pepper, play via ALAudioPlayer, clean up — all using a
        single SSH connection (previously opened a second connection just for rm).
        """
        if not _PARAMIKO_AVAILABLE:
            print("⚠️  paramiko unavailable — cannot transfer audio to robot")
            return False

        acquired = self._speech_lock.acquire(timeout=lock_timeout)
        if not acquired:
            print("⚠️  Already speaking — skipping audio playback")
            return False

        remote_path = f"/tmp/{os.path.basename(file_path)}"
        ssh = None
        try:
            # ── 1. Connect once, upload via SFTP ─────────────────────
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                self.ip,
                username=self.ssh_user,
                password=self.ssh_password,
                timeout=5,
            )
            sftp = ssh.open_sftp()
            sftp.put(file_path, remote_path)
            sftp.close()

            # ── 2. Play via ALAudioPlayer ─────────────────────────────
            player  = self.session.service("ALAudioPlayer")
            file_id = player.loadFile(remote_path)

            self._is_speaking_hq = True
            anim = threading.Thread(
                target=self._hq_speech_animation_loop,
                daemon=True,
                name="HQSpeechAnim",
            )
            anim.start()

            player.play(file_id)      # Blocks until playback done
            self._is_speaking_hq = False

            # ── 3. Clean up — reuse the same SSH connection ───────────
            try:
                ssh.exec_command(f"rm -f {remote_path}")
            except Exception:
                pass  # Non-critical — ignore

            return True

        except Exception as e:
            self._is_speaking_hq = False
            print(f"⚠️  ALAudioPlayer failed: {e}")
            return False
        finally:
            # Always close the single SSH connection and release speech lock
            if ssh:
                try:
                    ssh.close()
                except Exception:
                    pass
            self._speech_lock.release()

    def _hq_speech_animation_loop(self):
        """
        Background gestures + LED pulse while HQ audio is playing.

        Calls PUBLIC gesture methods (not _impl directly) so the gesture lock
        is always respected — user-triggered gestures and this loop can't
        fight over joint positions.
        """
        gesture_fns = [
            self.nod,
            self.explaining_gesture,
            self.thinking_gesture,
            self.look_around,
        ]
        while self._is_speaking_hq:
            try:
                self.set_eye_color("blue")
                if random.random() > 0.6:
                    random.choice(gesture_fns)()
                for _ in range(20):
                    if not self._is_speaking_hq:
                        break
                    time.sleep(0.1)
            except Exception as e:
                print(f"⚠️  HQ anim loop error: {e}")
                break
        self.set_eye_color("blue")

    # ------------------------------------------------------------------
    # Thinking indicator
    # ------------------------------------------------------------------

    def thinking_indicator(self, start: bool = True):
        if start:
            self._thinking = True
            if self._thinking_thread is None or not self._thinking_thread.is_alive():
                self._thinking_thread = threading.Thread(
                    target=self._pulse_thinking_loop,
                    daemon=True,
                    name="ThinkingPulse",
                )
                self._thinking_thread.start()
        else:
            self._thinking = False
            self.set_eye_color("blue")

    def _pulse_thinking_loop(self):
        colours = ["blue", "off"]
        idx = 0
        while self._thinking:
            try:
                self.set_eye_color(colours[idx % 2])
                idx += 1
                for _ in range(4):
                    if not self._thinking:
                        break
                    time.sleep(0.1)
            except Exception:
                break

    # ------------------------------------------------------------------
    # Gesture dispatcher
    # ------------------------------------------------------------------

    def _run_gesture(self, impl_fn, *args, **kwargs):
        def _worker():
            if not self._gesture_lock.acquire(blocking=False):
                return
            try:
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

    # ------------------------------------------------------------------
    # Gesture implementations
    # ------------------------------------------------------------------

    def _wave_impl(self):
        try:
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"]
            up    = [-0.5, -0.3, 1.5, 1.2]
            down  = [-0.5, -0.3, 1.0, 1.2]
            rest  = [ 1.5,  0.15, 0.5, 1.2]
            self.motion.setAngles(names, up, 0.2);   time.sleep(0.3)
            for _ in range(2):
                self.motion.setAngles(names, down, 0.3); time.sleep(0.2)
                self.motion.setAngles(names, up,   0.3); time.sleep(0.2)
            self.motion.setAngles(names, rest, 0.2)
        except Exception as e:
            print(f"❌ Wave error: {e}")

    def _nod_impl(self):
        try:
            self.motion.setAngles("HeadPitch",  0.3, 0.15); time.sleep(0.3)
            self.motion.setAngles("HeadPitch", -0.1, 0.15); time.sleep(0.3)
            self.motion.setAngles("HeadPitch",  0.0, 0.15)
        except Exception as e:
            print(f"❌ Nod error: {e}")

    def _shake_head_impl(self):
        try:
            self.motion.setAngles("HeadYaw",  0.4, 0.15); time.sleep(0.3)
            self.motion.setAngles("HeadYaw", -0.4, 0.15); time.sleep(0.3)
            self.motion.setAngles("HeadYaw",  0.0, 0.15)
        except Exception as e:
            print(f"❌ Shake head error: {e}")

    def _look_at_sound_impl(self):
        try:
            if self.awareness:
                self.awareness.setEnabled(True)
        except Exception as e:
            print(f"❌ Look at sound error: {e}")

    def _thinking_gesture_impl(self):
        try:
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw", "RWristYaw"]
            pose  = [-0.3, -0.3, 1.2, 1.0, 0.0]
            rest  = [ 1.5,  0.15, 0.5, 1.2, 0.0]
            self.motion.setAngles(names, pose, 0.15); time.sleep(1.0)
            self.motion.setAngles(names, rest, 0.15)
        except Exception as e:
            print(f"❌ Thinking gesture error: {e}")

    def _explaining_gesture_impl(self):
        try:
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                     "LShoulderPitch", "LShoulderRoll", "LElbowRoll"]
            open_ = [ 0.0, -0.3,  1.0,  0.0,  0.3, -1.0]
            close = [ 0.5, -0.1,  0.5,  0.5,  0.1, -0.5]
            rest  = [ 1.5,  0.15, 0.5,  1.5, -0.15, -0.5]
            self.motion.setAngles(names, open_, 0.2); time.sleep(0.4)
            self.motion.setAngles(names, close, 0.2); time.sleep(0.4)
            self.motion.setAngles(names, rest,  0.2)
        except Exception as e:
            print(f"❌ Explaining gesture error: {e}")

    def _excited_gesture_impl(self):
        try:
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                     "LShoulderPitch", "LShoulderRoll", "LElbowRoll"]
            up   = [-1.0, -0.3,  1.5, -1.0,  0.3, -1.5]
            rest = [ 1.5,  0.15, 0.5,  1.5, -0.15, -0.5]
            self.motion.setAngles(names, up,   0.15); time.sleep(0.8)
            self.motion.setAngles(names, rest, 0.15)
        except Exception as e:
            print(f"❌ Excited gesture error: {e}")

    def _point_forward_impl(self):
        try:
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"]
            point = [ 0.0, -0.3,  0.0,  1.5]
            rest  = [ 1.5,  0.15, 0.5,  1.2]
            self.motion.setAngles(names, point, 0.15); time.sleep(1.0)
            self.motion.setAngles(names, rest,  0.15)
        except Exception as e:
            print(f"❌ Point error: {e}")

    def _shrug_impl(self):
        try:
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                     "LShoulderPitch", "LShoulderRoll", "LElbowRoll",
                     "HeadPitch"]
            pose = [0.5, -0.5,  1.2,  0.5,  0.5, -1.2,  0.2]
            rest = [1.5,  0.15, 0.5,  1.5, -0.15, -0.5,  0.0]
            self.motion.setAngles(names, pose, 0.15); time.sleep(0.8)
            self.motion.setAngles(names, rest, 0.15)
        except Exception as e:
            print(f"❌ Shrug error: {e}")

    def _celebrate_impl(self):
        try:
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                     "LShoulderPitch", "LShoulderRoll", "LElbowRoll"]
            up   = [-0.5, -0.3,  1.5, -0.5,  0.3, -1.5]
            down = [ 0.0, -0.3,  1.0,  0.0,  0.3, -1.0]
            rest = [ 1.5,  0.15, 0.5,  1.5, -0.15, -0.5]
            for _ in range(2):
                self.motion.setAngles(names, up,   0.25); time.sleep(0.3)
                self.motion.setAngles(names, down, 0.25); time.sleep(0.3)
            self.motion.setAngles(names, rest, 0.2)
        except Exception as e:
            print(f"❌ Celebrate error: {e}")

    def _look_around_impl(self):
        try:
            self.motion.setAngles("HeadYaw", -0.5, 0.15); time.sleep(0.5)
            self.motion.setAngles("HeadYaw",  0.5, 0.15); time.sleep(0.5)
            self.motion.setAngles("HeadYaw",  0.0, 0.15)
        except Exception as e:
            print(f"❌ Look around error: {e}")

    def _bow_impl(self):
        try:
            self.motion.setAngles("HeadPitch", 0.5, 0.1); time.sleep(0.5)
            self.motion.setAngles("HeadPitch", 0.0, 0.1); time.sleep(0.3)
        except Exception as e:
            print(f"❌ Bow error: {e}")

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move_forward(self,  speed: float = 0.5): self._move( speed,  0,  0)
    def move_backward(self, speed: float = 0.5): self._move(-speed,  0,  0)
    def turn_left(self,     speed: float = 0.5): self._move( 0,      0,  speed)
    def turn_right(self,    speed: float = 0.5): self._move( 0,      0, -speed)
    def strafe_left(self,   speed: float = 0.3): self._move( 0,  speed,  0)
    def strafe_right(self,  speed: float = 0.3): self._move( 0, -speed,  0)

    def stop_movement(self):
        try:
            self.motion.stopMove()
        except Exception as e:
            print(f"❌ Stop error: {e}")

    def _move(self, x, y, theta):
        try:
            self.motion.move(x, y, theta)
        except Exception as e:
            print(f"❌ Move error: {e}")

    # ------------------------------------------------------------------
    # LEDs
    # ------------------------------------------------------------------

    _COLOUR_MAP = {
        "blue":   0x000000FF,
        "green":  0x0000FF00,
        "red":    0x00FF0000,
        "yellow": 0x00FFFF00,
        "white":  0x00FFFFFF,
        "off":    0x00000000,
    }

    def set_eye_color(self, color: str):
        try:
            rgb = self._COLOUR_MAP.get(color)
            if rgb is not None:
                self.leds.fadeRGB("FaceLeds", rgb, 0.5)
        except Exception as e:
            print(f"❌ LED error: {e}")

    def pulse_eyes(self, color: str = "blue", duration: float = 2.0):
        try:
            self.set_eye_color(color)
            time.sleep(duration / 2)
            self.set_eye_color("off")
            time.sleep(0.2)
            self.set_eye_color(color)
        except Exception as e:
            print(f"❌ Pulse error: {e}")