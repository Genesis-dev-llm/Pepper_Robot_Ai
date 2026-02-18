"""
Pepper Robot Interface
Handles all communication and control with Pepper via qi library
"""

import qi
import time
import threading
from typing import Optional

class PepperRobot:
    def __init__(self, ip: str, port: int):
        """Initialize connection to Pepper robot"""
        self.ip = ip
        self.port = port
        self.session = None
        self.tts = None
        self.motion = None
        self.animated_speech = None
        self.audio = None
        self.leds = None
        self.awareness = None
        self._speech_lock = threading.Lock()   # Prevent overlapping speech
        
    def connect(self) -> bool:
        """Establish connection to Pepper"""
        try:
            print(f"🤖 Connecting to Pepper at {self.ip}:{self.port}...")
            self.session = qi.Session()
            self.session.connect(f"tcp://{self.ip}:{self.port}")
            
            # Initialize services
            self.tts = self.session.service("ALTextToSpeech")
            self.motion = self.session.service("ALMotion")
            self.animated_speech = self.session.service("ALAnimatedSpeech")
            self.audio = self.session.service("ALAudioDevice")
            self.leds = self.session.service("ALLeds")
            self.awareness = self.session.service("ALBasicAwareness")
            
            # Wake up and set posture
            self.motion.wakeUp()
            time.sleep(1)
            
            print("✅ Connected to Pepper successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Pepper: {e}")
            return False
    
    def disconnect(self):
        """Safely disconnect from Pepper"""
        try:
            if self.motion:
                self.motion.rest()
            print("👋 Disconnected from Pepper")
        except Exception as e:
            print(f"⚠️ Error during disconnect: {e}")
    
    # ===== SPEECH =====
    
    def speak(self, text: str, use_animation: bool = True):
        """Make Pepper speak text (thread-safe — blocks if already speaking)."""
        with self._speech_lock:
            try:
                if use_animation and self.animated_speech:
                    self.animated_speech.say(text)
                else:
                    self.tts.say(text)
            except Exception as e:
                print(f"❌ Speech error: {e}")
    
    def set_volume(self, volume: int):
        """Set speech volume (0-100)"""
        try:
            self.tts.setVolume(volume / 100.0)
        except Exception as e:
            print(f"❌ Volume error: {e}")
    
    # ===== AUDIO CAPTURE =====
    
    def start_audio_capture(self):
        """Start capturing audio from microphones"""
        # This is handled differently - usually you'd use ALAudioRecorder
        # or subscribe to ALAudioDevice for streaming
        pass

    def play_audio_file(self, file_path: str) -> bool:
        """
        Play a local audio file through Pepper's speakers via ALAudioPlayer.

        Args:
            file_path: Absolute path to a WAV or MP3 file accessible on the robot
                       (or on the machine running this script if using a local qi session).

        Returns:
            True if playback succeeded, False otherwise.
        """
        with self._speech_lock:
            try:
                player = self.session.service("ALAudioPlayer")
                file_id = player.loadFile(file_path)
                player.play(file_id)
                return True
            except Exception as e:
                print(f"⚠️ ALAudioPlayer failed ({e}), falling back to built-in TTS")
                return False
    
    # ===== MOVEMENTS =====
    
    def wave(self):
        """Wave hello/goodbye"""
        try:
            print("👋 Waving...")
            # Right arm wave
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"]
            
            # Wave animation
            angles1 = [-0.5, -0.3, 1.5, 1.2]  # Arm up
            angles2 = [-0.5, -0.3, 1.0, 1.2]  # Wave down
            angles3 = [-0.5, -0.3, 1.5, 1.2]  # Wave up
            
            self.motion.setAngles(names, angles1, 0.2)
            time.sleep(0.3)
            
            # Wave motion
            for _ in range(2):
                self.motion.setAngles(names, angles2, 0.3)
                time.sleep(0.2)
                self.motion.setAngles(names, angles3, 0.3)
                time.sleep(0.2)
            
            # Return to rest
            self.motion.setAngles(names, [1.5, 0.15, 0.5, 1.2], 0.2)
            
        except Exception as e:
            print(f"❌ Wave error: {e}")
    
    def nod(self):
        """Nod head yes"""
        try:
            print("😊 Nodding...")
            # Head pitch control
            names = "HeadPitch"
            
            # Nod down and up
            self.motion.setAngles(names, 0.3, 0.15)  # Down
            time.sleep(0.3)
            self.motion.setAngles(names, -0.1, 0.15)  # Up
            time.sleep(0.3)
            self.motion.setAngles(names, 0.0, 0.15)  # Center
            
        except Exception as e:
            print(f"❌ Nod error: {e}")
    
    def shake_head(self):
        """Shake head no"""
        try:
            print("🙅 Shaking head...")
            names = "HeadYaw"
            
            # Shake left and right
            self.motion.setAngles(names, 0.4, 0.15)  # Right
            time.sleep(0.3)
            self.motion.setAngles(names, -0.4, 0.15)  # Left
            time.sleep(0.3)
            self.motion.setAngles(names, 0.0, 0.15)  # Center
            
        except Exception as e:
            print(f"❌ Shake head error: {e}")
    
    def look_at_sound(self):
        """Turn head toward sound source"""
        try:
            print("👂 Looking at sound source...")
            # Enable awareness to track people
            if self.awareness:
                self.awareness.setEnabled(True)
        except Exception as e:
            print(f"❌ Look at sound error: {e}")
    
    def thinking_gesture(self):
        """Thinking pose - hand to chin"""
        try:
            print("🤔 Thinking gesture...")
            # Right hand to chin area
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw", "RWristYaw"]
            angles = [-0.3, -0.3, 1.2, 1.0, 0.0]
            
            self.motion.setAngles(names, angles, 0.15)
            time.sleep(1.0)
            
            # Return to rest
            self.motion.setAngles(names, [1.5, 0.15, 0.5, 1.2, 0.0], 0.15)
            
        except Exception as e:
            print(f"❌ Thinking gesture error: {e}")
    
    def explaining_gesture(self):
        """Hand gestures while explaining"""
        try:
            print("✋ Explaining gesture...")
            # Both hands move expressively
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                    "LShoulderPitch", "LShoulderRoll", "LElbowRoll"]
            
            # Open gesture
            angles1 = [0.0, -0.3, 1.0, 0.0, 0.3, -1.0]
            self.motion.setAngles(names, angles1, 0.2)
            time.sleep(0.4)
            
            # Close gesture
            angles2 = [0.5, -0.1, 0.5, 0.5, 0.1, -0.5]
            self.motion.setAngles(names, angles2, 0.2)
            time.sleep(0.4)
            
            # Return to rest
            rest = [1.5, 0.15, 0.5, 1.5, -0.15, -0.5]
            self.motion.setAngles(names, rest, 0.2)
            
        except Exception as e:
            print(f"❌ Explaining gesture error: {e}")
    
    def excited_gesture(self):
        """Excited - both arms up"""
        try:
            print("🎉 Excited gesture...")
            # Both arms up
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                    "LShoulderPitch", "LShoulderRoll", "LElbowRoll"]
            
            angles = [-1.0, -0.3, 1.5, -1.0, 0.3, -1.5]
            self.motion.setAngles(names, angles, 0.15)
            time.sleep(0.8)
            
            # Return to rest
            rest = [1.5, 0.15, 0.5, 1.5, -0.15, -0.5]
            self.motion.setAngles(names, rest, 0.15)
            
        except Exception as e:
            print(f"❌ Excited gesture error: {e}")
    
    def point_forward(self):
        """Point forward with right hand"""
        try:
            print("👉 Pointing gesture...")
            # Right arm point
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"]
            angles = [0.0, -0.3, 0.0, 1.5]
            
            self.motion.setAngles(names, angles, 0.15)
            time.sleep(1.0)
            
            # Return to rest
            self.motion.setAngles(names, [1.5, 0.15, 0.5, 1.2], 0.15)
            
        except Exception as e:
            print(f"❌ Point error: {e}")
    
    def shrug(self):
        """Shrug gesture (I don't know)"""
        try:
            print("🤷 Shrug gesture...")
            # Both shoulders up, arms slightly out
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                    "LShoulderPitch", "LShoulderRoll", "LElbowRoll",
                    "HeadPitch"]
            
            # Shrug position
            angles = [0.5, -0.5, 1.2, 0.5, 0.5, -1.2, 0.2]
            self.motion.setAngles(names, angles, 0.15)
            time.sleep(0.8)
            
            # Return to rest
            rest = [1.5, 0.15, 0.5, 1.5, -0.15, -0.5, 0.0]
            self.motion.setAngles(names, rest, 0.15)
            
        except Exception as e:
            print(f"❌ Shrug error: {e}")
    
    def celebrate(self):
        """Celebration gesture - arms wave"""
        try:
            print("🎊 Celebrate gesture...")
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll",
                    "LShoulderPitch", "LShoulderRoll", "LElbowRoll"]
            
            # Wave both arms
            for _ in range(2):
                # Up
                angles1 = [-0.5, -0.3, 1.5, -0.5, 0.3, -1.5]
                self.motion.setAngles(names, angles1, 0.25)
                time.sleep(0.3)
                
                # Down
                angles2 = [0.0, -0.3, 1.0, 0.0, 0.3, -1.0]
                self.motion.setAngles(names, angles2, 0.25)
                time.sleep(0.3)
            
            # Return to rest
            rest = [1.5, 0.15, 0.5, 1.5, -0.15, -0.5]
            self.motion.setAngles(names, rest, 0.2)
            
        except Exception as e:
            print(f"❌ Celebrate error: {e}")
    
    def look_around(self):
        """Look around left and right"""
        try:
            print("👀 Looking around...")
            names = "HeadYaw"
            
            # Look right
            self.motion.setAngles(names, -0.5, 0.15)
            time.sleep(0.5)
            
            # Look left
            self.motion.setAngles(names, 0.5, 0.15)
            time.sleep(0.5)
            
            # Center
            self.motion.setAngles(names, 0.0, 0.15)
            
        except Exception as e:
            print(f"❌ Look around error: {e}")
    
    def bow(self):
        """Bow politely"""
        try:
            print("🙇 Bowing...")
            # Head and torso down
            self.motion.setAngles("HeadPitch", 0.5, 0.1)
            time.sleep(0.5)
            
            # Back up
            self.motion.setAngles("HeadPitch", 0.0, 0.1)
            time.sleep(0.3)
            
        except Exception as e:
            print(f"❌ Bow error: {e}")
    
    # ===== KEYBOARD MOVEMENT CONTROLS =====
    
    def move_forward(self, speed: float = 0.5):
        """Move forward"""
        try:
            self.motion.move(speed, 0, 0)
        except Exception as e:
            print(f"❌ Move forward error: {e}")
    
    def move_backward(self, speed: float = 0.5):
        """Move backward"""
        try:
            self.motion.move(-speed, 0, 0)
        except Exception as e:
            print(f"❌ Move backward error: {e}")
    
    def turn_left(self, speed: float = 0.5):
        """Turn left"""
        try:
            self.motion.move(0, 0, speed)
        except Exception as e:
            print(f"❌ Turn left error: {e}")
    
    def turn_right(self, speed: float = 0.5):
        """Turn right"""
        try:
            self.motion.move(0, 0, -speed)
        except Exception as e:
            print(f"❌ Turn right error: {e}")
    
    def strafe_left(self, speed: float = 0.3):
        """Strafe left"""
        try:
            self.motion.move(0, speed, 0)
        except Exception as e:
            print(f"❌ Strafe left error: {e}")
    
    def strafe_right(self, speed: float = 0.3):
        """Strafe right"""
        try:
            self.motion.move(0, -speed, 0)
        except Exception as e:
            print(f"❌ Strafe right error: {e}")
    
    def stop_movement(self):
        """Stop all movement"""
        try:
            self.motion.stopMove()
        except Exception as e:
            print(f"❌ Stop error: {e}")
    
    # ===== LED CONTROL =====
    
    def set_eye_color(self, color: str):
        """Set eye LED color (blue, green, red, yellow, white)"""
        try:
            color_map = {
                "blue": 0x000000FF,
                "green": 0x0000FF00,
                "red": 0x00FF0000,
                "yellow": 0x00FFFF00,
                "white": 0x00FFFFFF,
                "off": 0x00000000
            }
            
            if color in color_map:
                self.leds.fadeRGB("FaceLeds", color_map[color], 0.5)
        except Exception as e:
            print(f"❌ LED error: {e}")
    
    def pulse_eyes(self, color: str = "blue", duration: float = 2.0):
        """Pulse eye LEDs"""
        try:
            # Simple pulse effect
            self.set_eye_color(color)
            time.sleep(duration / 2)
            self.set_eye_color("off")
            time.sleep(0.2)
            self.set_eye_color(color)
        except Exception as e:
            print(f"❌ Pulse error: {e}")
    
    def thinking_indicator(self, start: bool = True):
        """
        Show thinking indicator (pulsing blue eyes)
        
        Args:
            start: True to start thinking animation, False to stop
        """
        try:
            if start:
                # Pulsing blue = thinking
                self.leds.fadeRGB("FaceLeds", 0x000000FF, 0.5)
            else:
                # Return to steady blue
                self.leds.fadeRGB("FaceLeds", 0x000000FF, 0.2)
        except Exception as e:
            print(f"❌ Thinking indicator error: {e}")