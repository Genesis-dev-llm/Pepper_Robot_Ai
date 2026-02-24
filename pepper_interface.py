"""
Pepper Robot Interface — NAOqi hardware control
"""

import logging
import os
import random
import threading
import time
from contextlib import contextmanager
from typing import Optional, TYPE_CHECKING

try:
    import qi
    _QI_AVAILABLE = True
except ImportError:
    qi = None
    _QI_AVAILABLE = False
    print("⚠️  NAOqi (qi) not installed — running in offline/chat-only mode")

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

        self.connected = False

        self.session         = None
        self.tts             = None
        self.motion          = None
        self.animated_speech = None
        self.audio           = None
        self.leds            = None
        self.awareness       = None
        self.tablet          = None   # ALTabletService — None if unavailable

        self._speech_lock  = threading.Lock()
        self._gesture_lock = threading.Lock()

        self._thinking        = False
        self._thinking_thread: Optional[threading.Thread] = None
        self._is_speaking_hq  = False

        # Persistent SSH connection for HQ audio transfer.
        # Kept alive across utterances — only reconnects when the transport
        # drops. Saves ~1-2s per response vs opening a new connection each time.
        self._ssh_client: Optional["paramiko.SSHClient"] = None

        # --- Gesture cooldown ---
        # Tracks when any gesture last fired so the background animation loop
        # doesn't spam conflicting setAngles commands mid-motion.
        # AI-triggered gestures bypass this (they're intentional) but update
        # the timestamp, which makes the loop back off after they fire.
        self._last_gesture_time: float = 0.0
        self._GESTURE_COOLDOWN: float  = 2.5   # seconds

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, timeout: float = 5.0) -> bool:
        if not _QI_AVAILABLE:
            print("⚠️  qi not available — offline mode")
            self.connected = False
            return False

        print(f"🤖 Connecting to Pepper at {self.ip}:{self.port} (timeout {timeout}s)…")

        _result = {"success": False, "error": None, "session": None}

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
                logging.warning("Connection timed out after %.1fs — launching in offline mode", timeout)
            else:
                logging.error("Connection failed: %s — launching in offline mode",
                              _result['error'])
            print("   Chat, TTS and web search will still work.")
            print("   Robot controls and gestures will be silently ignored.")
            self.connected = False
            return False

        self.session = _result["session"]
        try:
            self.tts             = self.session.service("ALTextToSpeech")
            self.motion          = self.session.service("ALMotion")
            self.animated_speech = self.session.service("ALAnimatedSpeech")
            self.audio           = self.session.service("ALAudioDevice")
            self.leds            = self.session.service("ALLeds")
            self.awareness       = self.session.service("ALBasicAwareness")

            # ALTabletService is available on Pepper 1.8+ but not all firmware
            # versions expose it. Catch the error gracefully so the rest of
            # the system works fine even without tablet support.
            try:
                self.tablet = self.session.service("ALTabletService")
                print("   ✅ Tablet service available")
            except Exception:
                self.tablet = None
                print("   ⚠️  ALTabletService not available — tablet display disabled")

            # Disable Autonomous Life — fights direct motion commands
            try:
                al = self.session.service("ALAutonomousLife")
                al.setState("disabled")
                print("   ✅ Autonomous Life disabled")
            except Exception as e:
                print(f"   ⚠️  Autonomous Life: {e}")

            # Stop BasicAwareness — overrides head/body orientation
            try:
                self.awareness.stopAwareness()
                print("   ✅ BasicAwareness stopped")
            except Exception as e:
                print(f"   ⚠️  BasicAwareness: {e}")

            # Shrink collision security distances as much as possible
            try:
                self.motion.setExternalCollisionProtectionEnabled("Move", False)
                print("   ✅ Collision protection disabled")
            except Exception:
                try:
                    self.motion.setOrthogonalSecurityDistance(0.05)
                    self.motion.setTangentialSecurityDistance(0.05)
                    print("   ✅ Collision security distances minimised")
                except Exception as e2:
                    print(f"   ⚠️  Collision protection unchanged: {e2}")

            # Ensure body stiffness is on
            try:
                self.motion.setStiffnesses("Body", 1.0)
                print("   ✅ Body stiffness set to 1.0")
            except Exception as e:
                print(f"   ⚠️  Stiffness: {e}")

            # Max speaker volume on connect
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
                al = self.session.service("ALAutonomousLife")
                al.setState("solitary")
            except Exception:
                pass
            try:
                if self.awareness:
                    self.awareness.startAwareness()
            except Exception:
                pass
            print("👋 Disconnected from Pepper")
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
                logging.error("Speech error: %s", e)

    def set_volume(self, volume: int):
        """Set both TTS voice volume and speaker output volume (0–100)."""
        try:
            self.tts.setVolume(volume / 100.0)
        except Exception as e:
            logging.error("TTS volume error: %s", e)
        try:
            self.audio.setOutputVolume(int(volume))
        except Exception as e:
            logging.error("Speaker volume error: %s", e)

    # ------------------------------------------------------------------
    # Speech — HQ audio pipeline
    # ------------------------------------------------------------------

    def speak_hq(self, text: str, tts_handler: "HybridTTSHandler",
                 emotion: Optional[str] = None) -> bool:
        try:
            audio_path = tts_handler.speak(text, emotion=emotion)
            if audio_path and self.play_audio_file(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
                return True
        except Exception as e:
            logging.warning("HQ TTS pipeline error: %s", e)

        print("↩️  Falling back to built-in NAOqi TTS")
        self.speak(text)
        return False

    def _ensure_ssh(self) -> bool:
        """
        Ensure the persistent SSH connection to Pepper is alive.

        Checks the underlying transport before attempting to reconnect, so
        healthy connections pay only a cheap attribute lookup per call.
        Called inside play_audio_file which already holds _speech_lock,
        so no additional locking is needed here.
        """
        if not _PARAMIKO_AVAILABLE:
            return False
        try:
            transport = self._ssh_client.get_transport() if self._ssh_client else None
            if transport and transport.is_active():
                return True
        except Exception:
            pass

        # Transport is dead or never opened — (re)connect.
        print("🔗 (Re)connecting SSH to Pepper…")
        try:
            if self._ssh_client:
                try:
                    self._ssh_client.close()
                except Exception:
                    pass
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.ip, username=self.ssh_user,
                           password=self.ssh_password, timeout=5)
            self._ssh_client = client
            print("   ✅ SSH connected")
            return True
        except Exception as e:
            print(f"   ❌ SSH connection failed: {e}")
            self._ssh_client = None
            return False

    def play_audio_file(self, file_path: str, lock_timeout: float = 0.5) -> bool:
        """
        Transfer an audio file to Pepper via SFTP and play it through ALAudioPlayer.

        Uses a persistent SSH connection — the TCP handshake only happens once
        per session (or after a network drop), not on every utterance.
        The SFTP channel is opened fresh per transfer; it's cheap once the
        transport is already established.
        """
        if not _PARAMIKO_AVAILABLE:
            print("⚠️  paramiko unavailable — cannot transfer audio to robot")
            return False

        acquired = self._speech_lock.acquire(timeout=lock_timeout)
        if not acquired:
            print("⚠️  Already speaking — skipping audio playback")
            return False

        remote_path = f"/tmp/{os.path.basename(file_path)}"
        try:
            if not self._ensure_ssh():
                return False

            # Open a fresh SFTP channel on the existing transport.
            sftp = self._ssh_client.open_sftp()
            sftp.put(file_path, remote_path)
            sftp.close()

            player  = self.session.service("ALAudioPlayer")
            file_id = player.loadFile(remote_path)

            self._is_speaking_hq = True
            threading.Thread(
                target=self._hq_speech_animation_loop,
                daemon=True, name="HQSpeechAnim"
            ).start()

            player.play(file_id)

        except Exception as e:
            logging.warning("ALAudioPlayer failed: %s", e)
            return False
        finally:
            # Always clear flag and release lock — even on exception.
            # SSH stays open; only clean up the remote temp file.
            self._is_speaking_hq = False
            if self._ssh_client:
                try:
                    self._ssh_client.exec_command(f"rm -f {remote_path}")
                except Exception:
                    pass
            self._speech_lock.release()

        return True

    def _hq_speech_animation_loop(self):
        """
        Fires random background gestures during HQ audio playback.

        Respects _GESTURE_COOLDOWN — if an AI-triggered gesture fired recently
        (or another background gesture is still running), this loop backs off
        instead of sending conflicting setAngles commands mid-motion.
        Movement (base/wheels) is completely independent and never blocked here.
        """
        gesture_fns = [self.nod, self.explaining_gesture,
                       self.thinking_gesture, self.look_around]
        while self._is_speaking_hq:
            try:
                self.set_eye_color("blue")
                # Only attempt a gesture if enough time has passed since the
                # last one — prevents arm jerk from overlapping commands.
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
        self.set_eye_color("blue")

    # ------------------------------------------------------------------
    # Thinking indicator + context manager
    # ------------------------------------------------------------------

    @contextmanager
    def thinking(self):
        """
        Context manager for the thinking indicator.

            with pepper.thinking():
                response = brain.chat(message)   # eyes pulse while thinking
            # eyes stop automatically here, even if an exception was raised

        Replaces the old start=True / start=False pair and the fragile
        _thinking_started flag in main.py.
        """
        self.thinking_indicator(start=True)
        try:
            yield
        finally:
            self.thinking_indicator(start=False)

    def thinking_indicator(self, start: bool = True):
        if start:
            self._thinking = True
            if self._thinking_thread is None or not self._thinking_thread.is_alive():
                self._thinking_thread = threading.Thread(
                    target=self._pulse_thinking_loop,
                    daemon=True, name="ThinkingPulse"
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
        """
        Spawns a daemon thread to run a gesture non-blocking.
        Records the start time so the animation loop can respect the cooldown.
        The gesture lock prevents two gestures running simultaneously on the arms.
        Base movement is on a completely separate motion API and is unaffected.
        """
        def _worker():
            if not self._gesture_lock.acquire(blocking=False):
                return
            try:
                # Stamp the time before executing so the animation loop
                # immediately backs off, not just after the physical move ends.
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
            logging.error("Wave error: %s", e)

    def _nod_impl(self):
        try:
            self.motion.setAngles("HeadPitch",  0.3, 0.15); time.sleep(0.3)
            self.motion.setAngles("HeadPitch", -0.1, 0.15); time.sleep(0.3)
            self.motion.setAngles("HeadPitch",  0.0, 0.15)
        except Exception as e:
            logging.error("Nod error: %s", e)

    def _shake_head_impl(self):
        try:
            self.motion.setAngles("HeadYaw",  0.4, 0.15); time.sleep(0.3)
            self.motion.setAngles("HeadYaw", -0.4, 0.15); time.sleep(0.3)
            self.motion.setAngles("HeadYaw",  0.0, 0.15)
        except Exception as e:
            logging.error("Shake head error: %s", e)

    def _look_at_sound_impl(self):
        try:
            if self.awareness:
                self.awareness.setEnabled(True)
        except Exception as e:
            logging.error("Look at sound error: %s", e)

    def _thinking_gesture_impl(self):
        try:
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw", "RWristYaw"]
            pose  = [-0.3, -0.3, 1.2, 1.0, 0.0]
            rest  = [ 1.5,  0.15, 0.5, 1.2, 0.0]
            self.motion.setAngles(names, pose, 0.15); time.sleep(1.0)
            self.motion.setAngles(names, rest, 0.15)
        except Exception as e:
            logging.error("Thinking gesture error: %s", e)

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
            logging.error("Explaining gesture error: %s", e)

    def _excited_gesture_impl(self):
        try:
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                     "LShoulderPitch", "LShoulderRoll", "LElbowRoll"]
            up   = [-1.0, -0.3,  1.5, -1.0,  0.3, -1.5]
            rest = [ 1.5,  0.15, 0.5,  1.5, -0.15, -0.5]
            self.motion.setAngles(names, up,   0.15); time.sleep(0.8)
            self.motion.setAngles(names, rest, 0.15)
        except Exception as e:
            logging.error("Excited gesture error: %s", e)

    def _point_forward_impl(self):
        try:
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"]
            point = [ 0.0, -0.3,  0.0,  1.5]
            rest  = [ 1.5,  0.15, 0.5,  1.2]
            self.motion.setAngles(names, point, 0.15); time.sleep(1.0)
            self.motion.setAngles(names, rest,  0.15)
        except Exception as e:
            logging.error("Point error: %s", e)

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
            logging.error("Shrug error: %s", e)

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
            logging.error("Celebrate error: %s", e)

    def _look_around_impl(self):
        try:
            self.motion.setAngles("HeadYaw", -0.5, 0.15); time.sleep(0.5)
            self.motion.setAngles("HeadYaw",  0.5, 0.15); time.sleep(0.5)
            self.motion.setAngles("HeadYaw",  0.0, 0.15)
        except Exception as e:
            logging.error("Look around error: %s", e)

    def _bow_impl(self):
        try:
            self.motion.setAngles("HeadPitch", 0.5, 0.1); time.sleep(0.5)
            self.motion.setAngles("HeadPitch", 0.0, 0.1); time.sleep(0.3)
        except Exception as e:
            logging.error("Bow error: %s", e)

    # ------------------------------------------------------------------
    # Movement — moveToward() normalised velocity, -1.0 to 1.0.
    # Called continuously at 20 Hz by movement_controller while key held.
    # Completely independent of gesture lock — arms and base move freely
    # at the same time.
    # ------------------------------------------------------------------

    def move_forward(self,  speed: float = 0.6): self._move( speed,  0,  0)
    def move_backward(self, speed: float = 0.6): self._move(-speed,  0,  0)
    def turn_left(self,     speed: float = 0.5): self._move( 0,      0,  speed)
    def turn_right(self,    speed: float = 0.5): self._move( 0,      0, -speed)
    def strafe_left(self,   speed: float = 0.4): self._move( 0,  speed,  0)
    def strafe_right(self,  speed: float = 0.4): self._move( 0, -speed,  0)

    def stop_movement(self):
        try:
            self.motion.moveToward(0.0, 0.0, 0.0)
            self.motion.stopMove()
        except Exception as e:
            logging.error("Stop error: %s", e)

    def _move(self, x: float, y: float, theta: float):
        try:
            self.motion.moveToward(x, y, theta)
        except Exception as e:
            logging.error("Move error: %s", e)

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

    # Maps emotion strings (from express_emotion function calls) to eye colours.
    # Eyes are set before Pepper starts speaking so the colour is visible while
    # she talks. The _hq_speech_animation_loop reverts to blue after speaking.
    EMOTION_COLOUR_MAP = {
        "happy":     "yellow",
        "sad":       "blue",
        "excited":   "green",
        "curious":   "blue",
        "surprised": "white",
        "neutral":   "blue",
    }

    def set_eye_color(self, color: str):
        try:
            rgb = self._COLOUR_MAP.get(color)
            if rgb is not None:
                self.leds.fadeRGB("FaceLeds", rgb, 0.5)
        except Exception as e:
            logging.error("LED error: %s", e)

    def pulse_eyes(self, color: str = "blue", duration: float = 2.0):
        try:
            self.set_eye_color(color)
            time.sleep(duration / 2)
            self.set_eye_color("off")
            time.sleep(0.2)
            self.set_eye_color(color)
        except Exception as e:
            logging.error("Pulse error: %s", e)

    # ------------------------------------------------------------------
    # Tablet display
    # ------------------------------------------------------------------

    def show_tablet_image(self, url: str):
        """
        Tell Pepper's tablet to display the image at the given URL.

        The URL must be reachable from the tablet's Android browser on the
        local network — i.e. served by PepperDisplayManager's HTTP server.
        This is a one-shot call; the image persists on the tablet until
        clear_tablet() is called or the tablet is woken/reset.
        """
        if not self.tablet:
            logging.warning("show_tablet_image: ALTabletService not available")
            return
        try:
            self.tablet.showImage(url)
        except Exception as e:
            logging.warning("show_tablet_image failed: %s", e)

    def clear_tablet(self):
        """Hide any displayed image and return the tablet to its default state."""
        if not self.tablet:
            return
        try:
            self.tablet.hideImage()
        except Exception as e:
            logging.warning("clear_tablet failed: %s", e)