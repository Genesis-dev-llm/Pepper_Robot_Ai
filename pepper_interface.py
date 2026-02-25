"""
Pepper Robot Interface — NAOqi hardware control

Key changes from previous version:
- LED priority system: thinking > speaking > idle. No more three systems
  stomping on each other's eye colors from different threads.
- Speech lock now only covers actual NAOqi playback, not SSH file transfer.
  Movement is completely unaffected — it never touched this lock anyway.
- Animation thread is stored and joined after player.play() returns, so
  there's no race between the loop exiting and the next operation touching LEDs.
- speak_hq() always cleans up the local temp file via finally, no leaks.
- speak() now uses a timeout-based lock acquire instead of blocking forever.
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
        self.tablet          = None

        # Speech lock — held ONLY during active NAOqi audio playback.
        # Never held during SSH transfer, TTS generation, or movement.
        self._speech_lock  = threading.Lock()
        self._gesture_lock = threading.Lock()

        # Thinking pulse state
        self._thinking         = False
        self._thinking_thread: Optional[threading.Thread] = None

        # HQ speech animation state
        self._is_speaking_hq   = False
        self._anim_thread: Optional[threading.Thread] = None

        # ── LED priority state ─────────────────────────────────────────
        # Three systems used to write eye colors independently, causing
        # visible flicker. Now all LED writes go through a single state
        # machine with explicit priority levels:
        #   "thinking"  — highest: pulsing while LLM is working
        #   "speaking"  — holds emotion color while audio plays
        #   "idle"      — lowest: steady blue or whatever was last set
        self._led_state         = "idle"   # "idle" | "thinking" | "speaking"
        self._led_emotion_color = "blue"   # retained during speaking state
        self._led_lock          = threading.Lock()

        # Persistent SSH connection — kept alive across utterances
        self._ssh_client: Optional["paramiko.SSHClient"] = None

        # Gesture cooldown
        self._last_gesture_time: float = 0.0
        self._GESTURE_COOLDOWN: float  = 2.5

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

            try:
                self.tablet = self.session.service("ALTabletService")
                print("   ✅ Tablet service available")
            except Exception:
                self.tablet = None
                print("   ⚠️  ALTabletService not available — tablet display disabled")

            try:
                al = self.session.service("ALAutonomousLife")
                al.setState("disabled")
                print("   ✅ Autonomous Life disabled")
            except Exception as e:
                print(f"   ⚠️  Autonomous Life: {e}")

            try:
                self.awareness.stopAwareness()
                print("   ✅ BasicAwareness stopped")
            except Exception as e:
                print(f"   ⚠️  BasicAwareness: {e}")

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

            try:
                self.motion.setStiffnesses("Body", 1.0)
                print("   ✅ Body stiffness set to 1.0")
            except Exception as e:
                print(f"   ⚠️  Stiffness: {e}")

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
    # LED priority state machine
    # ------------------------------------------------------------------
    # All eye color changes flow through here. No external code should
    # call set_eye_color() directly for state-driven purposes — use these
    # enter/exit methods so the priority logic is always respected.
    # Manual key presses (LEDs 5/6/7) still call set_eye_color directly
    # since they're intentional overrides by the operator.

    def _enter_led_thinking(self):
        """Claim LED ownership for the thinking state (highest priority)."""
        with self._led_lock:
            self._led_state = "thinking"
        # Eye will be set by the pulse loop on its next tick

    def _exit_led_thinking(self):
        """Release thinking LED ownership → back to idle, steady blue."""
        with self._led_lock:
            if self._led_state == "thinking":
                self._led_state = "idle"
        self.set_eye_color("blue")

    def _enter_led_speaking(self, emotion_color: str = "blue"):
        """
        Claim LED ownership for speaking state and set the emotion color.
        No-ops if thinking is active (thinking has higher priority).
        """
        with self._led_lock:
            if self._led_state == "thinking":
                return  # don't stomp on thinking state
            self._led_state         = "speaking"
            self._led_emotion_color = emotion_color
        self.set_eye_color(emotion_color)

    def _exit_led_speaking(self):
        """
        Release speaking LED ownership → back to idle, steady blue.
        Only touches the LED hardware if we actually owned the speaking state.
        Safe to call multiple times (idempotent).
        """
        state_was_speaking = False
        with self._led_lock:
            if self._led_state == "speaking":
                self._led_state         = "idle"
                self._led_emotion_color = "blue"
                state_was_speaking = True
        if state_was_speaking:
            self.set_eye_color("blue")

    # ------------------------------------------------------------------
    # Speech — built-in NAOqi TTS
    # ------------------------------------------------------------------

    def speak(self, text: str, use_animation: bool = True):
        """
        NAOqi built-in speech (fallback / offline).
        Uses a timeout-based lock acquire so it never blocks indefinitely.
        """
        acquired = self._speech_lock.acquire(timeout=3.0)
        if not acquired:
            logging.warning("speak(): could not acquire speech lock in 3.0s — skipping")
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

    # ------------------------------------------------------------------
    # Speech — HQ audio pipeline
    # ------------------------------------------------------------------

    def speak_hq(self, text: str, tts_handler: "HybridTTSHandler",
                 emotion: Optional[str] = None,
                 status_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Full HQ pipeline: TTS generation → SSH transfer → NAOqi playback.

        status_callback is called with human-readable progress strings so
        the GUI can show exactly what's happening during the otherwise-silent
        gap between LLM response and audio starting.

        Local temp file is always cleaned up in the finally block regardless
        of which path (success / fallback / exception) is taken.
        """
        audio_path = None
        try:
            if status_callback:
                status_callback("🎙️ Generating voice…")
            audio_path = tts_handler.speak(text, emotion=emotion)

            if audio_path:
                emotion_color = self.EMOTION_COLOUR_MAP.get(emotion or "", "blue")
                if self.play_audio_file(audio_path,
                                        emotion_color    = emotion_color,
                                        status_callback  = status_callback):
                    return True

            print("↩️  Falling back to built-in NAOqi TTS")
            self.speak(text)
            return False

        finally:
            # Always delete the local temp file — covers success, fallback, and exception
            if audio_path:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    def _ensure_ssh(self) -> bool:
        """
        Ensure the persistent SSH connection to Pepper is alive.
        Called from _transfer_to_robot (no lock held).
        """
        if not _PARAMIKO_AVAILABLE:
            return False
        try:
            transport = self._ssh_client.get_transport() if self._ssh_client else None
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
            client.connect(self.ip, username=self.ssh_user,
                           password=self.ssh_password, timeout=5)
            self._ssh_client = client
            print("   ✅ SSH connected")
            return True
        except Exception as e:
            print(f"   ❌ SSH connection failed: {e}")
            self._ssh_client = None
            return False

    def _transfer_to_robot(self, local_path: str) -> Optional[str]:
        """
        Transfer an audio file to Pepper via SFTP.

        No speech lock held — SSH + file transfer is completely independent
        of NAOqi audio. Movement, queued messages, and other operations
        continue uninterrupted while the file is being sent.

        Returns the remote path on success, None on failure.
        """
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
            logging.warning("SFTP transfer failed: %s", e)
            return None

    def play_audio_file(self, file_path: str,
                        emotion_color:   str = "blue",
                        status_callback: Optional[Callable[[str], None]] = None,
                        lock_timeout:    float = 2.0) -> bool:
        """
        Transfer audio to Pepper and play it through ALAudioPlayer.

        Phase 1 — Transfer (no lock):
            SSH health check + SFTP put. Movement and queued messages
            continue freely during this phase.

        Phase 2 — Playback (lock held):
            Speech lock is acquired only for the NAOqi play() call.
            Animation thread is started, runs during playback, then
            joined before the lock is released so LEDs are always in
            a clean state when the next operation begins.
        """
        if not _PARAMIKO_AVAILABLE:
            print("⚠️  paramiko unavailable — cannot transfer audio to robot")
            return False

        # ── Phase 1: Transfer (lock-free) ─────────────────────────────
        if status_callback:
            status_callback("📡 Sending to robot…")
        remote_path = self._transfer_to_robot(file_path)
        if not remote_path:
            return False

        # ── Phase 2: Playback (speech lock only) ──────────────────────
        acquired = self._speech_lock.acquire(timeout=lock_timeout)
        if not acquired:
            print("⚠️  Already speaking — skipping audio playback")
            self._cleanup_remote(remote_path)
            return False

        try:
            if status_callback:
                status_callback("🔊 Speaking…")

            # Set emotion color and mark LED state as speaking
            self._enter_led_speaking(emotion_color)

            player  = self.session.service("ALAudioPlayer")
            file_id = player.loadFile(remote_path)

            # Start background animation loop
            self._is_speaking_hq = True
            anim_thread = threading.Thread(
                target = self._hq_speech_animation_loop,
                daemon = True,
                name   = "HQSpeechAnim",
            )
            self._anim_thread = anim_thread
            anim_thread.start()

            player.play(file_id)  # blocks until audio finishes
            return True

        except Exception as e:
            logging.warning("ALAudioPlayer failed: %s", e)
            return False

        finally:
            # Signal animation to stop, then wait for it to exit cleanly.
            # This guarantees the animation is fully done and LEDs are in
            # a known state before the lock is released and the next
            # operation begins.
            self._is_speaking_hq = False
            if self._anim_thread and self._anim_thread.is_alive():
                self._anim_thread.join(timeout=3.0)
            self._anim_thread = None

            # Reset LED state — safe to call even if anim thread already did it
            self._exit_led_speaking()

            # Clean up remote temp file
            self._cleanup_remote(remote_path)

            self._speech_lock.release()

    def _cleanup_remote(self, remote_path: str):
        """Remove temp file from Pepper's /tmp. Best-effort."""
        if self._ssh_client:
            try:
                self._ssh_client.exec_command(f"rm -f {remote_path}")
            except Exception:
                pass

    def _hq_speech_animation_loop(self):
        """
        Background gestures during HQ audio playback.

        Reads emotion color from the LED state instead of hardcoding blue,
        so the color persists correctly throughout the whole speech.
        Does NOT call _exit_led_speaking — that's handled in play_audio_file's
        finally block after this thread is joined.
        """
        gesture_fns = [self.nod, self.explaining_gesture,
                       self.thinking_gesture, self.look_around]
        while self._is_speaking_hq:
            try:
                # Always reflect the current emotion color, not hardcoded blue
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
        # Loop exits cleanly — play_audio_file's finally handles LED reset

    # ------------------------------------------------------------------
    # Thinking indicator + context manager
    # ------------------------------------------------------------------

    @contextmanager
    def thinking(self):
        """
        Context manager for the thinking indicator.

            with pepper.thinking():
                response = brain.chat(message)
            # eyes reset automatically, even on exception
        """
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
                    target = self._pulse_thinking_loop,
                    daemon = True,
                    name   = "ThinkingPulse",
                )
                self._thinking_thread.start()
        else:
            self._thinking = False
            self._exit_led_thinking()

    def _pulse_thinking_loop(self):
        """
        Alternates eyes blue/off while thinking.
        Exits immediately if LED state is no longer 'thinking' so it
        doesn't conflict with speaking state if the two overlap.
        """
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

    # ------------------------------------------------------------------
    # Gesture dispatcher
    # ------------------------------------------------------------------

    def _run_gesture(self, impl_fn, *args, **kwargs):
        """
        Spawns a daemon thread to run a gesture non-blocking.
        Movement (base/wheels) is on a completely separate motion API
        and is never touched here.
        """
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
    # Movement
    # Movement calls motion.moveToward() directly — completely independent
    # of the speech lock, gesture lock, and LED state. Speaking, thinking,
    # and gesturing never block or delay movement.
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
        if not self.tablet:
            logging.warning("show_tablet_image: ALTabletService not available")
            return
        try:
            self.tablet.showImage(url)
        except Exception as e:
            logging.warning("show_tablet_image failed: %s", e)

    def show_tablet_webview(self, url: str):
        """Display an HTML page on the tablet via ALTabletService.showWebview()."""
        if not self.tablet:
            logging.warning("show_tablet_webview: ALTabletService not available")
            return
        try:
            self.tablet.showWebview(url)
        except Exception as e:
            logging.warning("show_tablet_webview failed: %s", e)

    def clear_tablet(self):
        if not self.tablet:
            return
        try:
            self.tablet.hideWebview()
        except Exception:
            pass
        try:
            self.tablet.hideImage()
        except Exception as e:
            logging.warning("clear_tablet failed: %s", e)