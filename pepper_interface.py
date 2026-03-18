"""
Pepper Robot Interface — NAOqi hardware control

Changes from previous version:
- gesture_callback parameter added to speak_hq() and play_audio_file().
  The callback fires immediately before player.play() so intentional gestures
  are synchronized with speech onset rather than finishing before speech starts.
- SSH keepalive (set_keepalive) to keep idle sessions alive between clips.
- _valid_audio now checks WAV/MP3 magic bytes, not just file size.
- EMOTION_COLOUR_MAP updated — each emotion now has a visually distinct color.
- Dead named movement methods removed (only _move() is used).
- _hq_speech_animation_loop gesture pool expanded.
- speak() lock timeout surfaced as a parameter for clarity.

Latest changes:
- show_tablet_webview: fixed call order to loadUrl() → showWebview() per NAOqi
  spec. Previously called showWebview(url) directly which is wrong on most
  firmware versions.
- _ensure_ssh: removed exec_command("echo ok") liveness probe from the hot
  path. The 30s keepalive already handles silent drops; the probe was blocking
  for up to 2s on every audio playback call. SFTP failures now surface
  reactively through the existing single-retry path.
- open_browser: new method — loadUrl() + showWebview() + enableTabletAccess()
  for a fully interactive browser with touch/typing support.
- free_tablet: new method — hideWebview() to exit NAOqi kiosk and return to
  Android home screen / Chrome.
- show_tablet_webview now also calls enableTabletAccess() so all webview
  calls are interactive by default.

connect() rework:
- All service lookups, hardware init calls (Autonomous Life, BasicAwareness,
  Body stiffness, wakeUp) moved inside the _attempt daemon thread so the
  t.join(timeout) covers everything. If Pepper is unreachable or any NAOqi
  call hangs, the whole connect() returns cleanly in ≤ timeout seconds
  instead of blocking the main thread indefinitely. timeout raised to 15s
  to comfortably cover wakeUp() on a slow robot.

play_audio_file fix:
- player.play(file_id) is a blocking NAOqi call that only returns when
  playback finishes. A NAOqi stall or WiFi drop mid-play meant _speech_lock
  was held forever. Now runs in a daemon thread with a 60s threading.Event
  timeout so the lock is always released.

camera stream fix:
- Replaced time.sleep(2.0) with a TCP socket poll loop (10 × 0.5s) so we
  break as soon as the server is ready instead of always waiting 2 full seconds.
"""

import logging
import os
import random
import socket
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

# Absolute path of the directory containing this file.
# Used to locate camera_stream.py for deployment to Pepper.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class PepperRobot:
    # Minimum seconds between autonomous background gestures during HQ speech.
    # Class-level constant — never varies per instance.
    _GESTURE_COOLDOWN: float = 2.5

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

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self, timeout: float = 15.0) -> bool:
        """
        Connect to Pepper and initialise all NAOqi services.

        Everything — TCP session, service lookups, hardware init, wakeUp() —
        runs inside a daemon thread so t.join(timeout) is the single ceiling
        for the entire connect sequence.  If Pepper is unreachable or any
        call hangs, this method returns False cleanly within `timeout` seconds.
        timeout default is 15s to comfortably cover wakeUp() on a slow robot.
        """
        if not _QI_AVAILABLE:
            print("⚠️  qi not available — offline mode")
            self.connected = False
            return False

        print(f"🤖 Connecting to Pepper at {self.ip}:{self.port}…")
        # [success, error_reason]
        result = [False, "timed out"]

        def _attempt():
            try:
                s = qi.Session()
                s.connect(f"tcp://{self.ip}:{self.port}")
                self.session = s

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

                result[0] = True
                result[1] = None
            except Exception as e:
                result[0] = False
                result[1] = str(e)

        t = threading.Thread(target=_attempt, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if not result[0]:
            reason = result[1]
            logging.warning("Connection failed (%s) — offline mode", reason)
            print("   Chat, TTS and web search will still work.")
            self.connected = False
            return False

        time.sleep(1)
        self.connected = True
        print("✅ Connected to Pepper!")
        return True

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
        if not _PARAMIKO_AVAILABLE:
            return False

        if self._ssh_client:
            try:
                transport = self._ssh_client.get_transport()
                if transport and transport.is_active():
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

            if gesture_callback:
                try:
                    gesture_callback()
                except Exception as e:
                    logging.warning("gesture_callback error: %s", e)

            self._is_speaking_hq = True
            anim_thread = threading.Thread(
                target=self._hq_speech_animation_loop,
                daemon=True,
                name="HQSpeechAnim",
            )
            self._anim_thread = anim_thread
            anim_thread.start()

            # Run player.play() in a daemon thread so a hung NAOqi call
            # cannot hold _speech_lock indefinitely.  A 60s timeout is far
            # beyond any realistic TTS clip — if it fires we log and recover.
            play_done = threading.Event()

            def _play_worker():
                try:
                    player.play(file_id)
                except Exception as _e:
                    logging.warning("ALAudioPlayer.play error: %s", _e)
                finally:
                    play_done.set()

            threading.Thread(target=_play_worker, daemon=True, name="AudioPlay").start()
            timed_out = not play_done.wait(timeout=60.0)
            if timed_out:
                logging.warning(
                    "play_audio_file: player.play() timed out after 60s — releasing lock"
                )
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

    EMOTION_COLOUR_MAP = {
        "happy":     "yellow",
        "sad":       "purple",
        "excited":   "green",
        "curious":   "cyan",
        "surprised": "white",
        "neutral":   "blue",
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
        try:
            if not os.path.exists(path) or os.path.getsize(path) < 4:
                return False
            with open(path, "rb") as f:
                header = f.read(12)
            if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                return True
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
        """
        Display a URL in Pepper's tablet browser.

        Correct NAOqi call order: loadUrl() loads the page into the browser
        engine, then showWebview() (no argument) brings the webview panel into
        view.  enableTabletAccess() is called last to unlock touch input so
        the user can interact with the page.
        """
        if not self.tablet:
            return
        try:
            self.tablet.loadUrl(url)
            self.tablet.showWebview()
            self.tablet.enableTabletAccess()
        except Exception as e:
            logging.warning("show_tablet_webview: %s", e)

    def open_browser(self, url: str):
        """
        Open a fully interactive browser on Pepper's tablet.

        Same as show_tablet_webview — loadUrl() + showWebview() +
        enableTabletAccess() — but explicitly named for interactive use.
        The user can tap links, type in the address bar, scroll, and use
        YouTube, Google, or any other site normally.
        """
        if not self.tablet:
            logging.warning("open_browser: tablet service not available")
            return
        try:
            self.tablet.loadUrl(url)
            self.tablet.showWebview()
            self.tablet.enableTabletAccess()
            print(f"🌐 Tablet browser opened: {url}")
        except Exception as e:
            logging.warning("open_browser: %s", e)

    def free_tablet(self):
        """
        Exit the NAOqi webview kiosk and return to Android home screen.

        Calling hideWebview() drops out of the NAOqi-controlled webview
        and lands on the Android launcher, giving access to Chrome and
        any other installed apps. Call open_browser() to return to a
        managed URL.
        """
        if not self.tablet:
            return
        try:
            self.tablet.hideWebview()
            print("🏠 Tablet freed — Android home screen")
        except Exception as e:
            logging.warning("free_tablet: %s", e)

    def clear_tablet(self):
        if not self.tablet:
            return
        for method in [lambda: self.tablet.hideWebview(),
                       lambda: self.tablet.hideImage()]:
            try:
                method()
            except Exception:
                pass

    # ── Camera → Tablet streaming ─────────────────────────────────────────────

    def start_tablet_camera_stream(self) -> bool:
        if not _PARAMIKO_AVAILABLE:
            print("❌ Camera stream: paramiko not installed")
            return False
        if not self.tablet:
            print("❌ Camera stream: tablet service not available")
            return False

        local_script = os.path.join(_SCRIPT_DIR, "camera_stream.py")
        if not os.path.isfile(local_script):
            print("❌ Camera stream: camera_stream.py not found at {0}".format(local_script))
            return False

        if not self._ensure_ssh():
            print("❌ Camera stream: SSH connection failed")
            return False

        remote_script = "/home/nao/camera_stream.py"
        try:
            sftp = self._ssh_client.open_sftp()
            sftp.put(local_script, remote_script)
            sftp.close()
            print("📤 camera_stream.py uploaded to Pepper")
        except Exception as e:
            print("❌ Camera stream: SFTP upload failed: {0}".format(e))
            return False

        launch_cmd = (
            "pkill -f camera_stream.py 2>/dev/null; sleep 0.3; "
            "python {script} > /tmp/camera_stream.log 2>&1 &"
        ).format(script=remote_script)
        try:
            self._ssh_client.exec_command(launch_cmd)
        except Exception as e:
            print("❌ Camera stream: launch command failed: {0}".format(e))
            return False

        print("⏳ Waiting for camera server to start…")
        # Poll the HTTP server port instead of sleeping a fixed 2s.
        # On a loaded Pepper CPU the server can take longer than 2s to bind;
        # on a fast one we waste time waiting.  Try up to 10 times with 0.5s
        # gaps (5s ceiling) — break as soon as a TCP connection succeeds.
        _camera_host = "198.18.0.1"
        _camera_port = 8080
        _ready = False
        for _attempt in range(10):
            try:
                s = socket.create_connection((_camera_host, _camera_port), timeout=0.5)
                s.close()
                _ready = True
                break
            except OSError:
                time.sleep(0.5)
        if not _ready:
            print("⚠️  Camera server did not respond after 5s — attempting webview anyway")

        stream_url = "http://198.18.0.1:8080/stream.html"
        try:
            self.tablet.loadUrl(stream_url)
            self.tablet.showWebview()
            print("📷 Camera stream live → {0}".format(stream_url))
            return True
        except Exception as e:
            logging.warning("Camera stream: tablet webview failed: %s", e)
            return False

    def stop_tablet_camera_stream(self):
        if self._ssh_client:
            try:
                self._ssh_client.exec_command("pkill -f camera_stream.py || true")
                print("📷 Camera stream stopped")
            except Exception as e:
                logging.warning("stop_tablet_camera_stream kill failed: %s", e)

        self.clear_tablet()