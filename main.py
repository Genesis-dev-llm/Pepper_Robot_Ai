"""
Main Control Script for Pepper AI Robot
- Keyboard controls for movement and gestures
- Push-to-talk voice input (hold R) → Groq Whisper STT
- Text input via DearPyGUI
- AI brain with web search
"""

import time
import threading
from pynput import keyboard
from pepper_interface import PepperRobot
from groq_brain import GroqBrain, test_groq_connection
from hybrid_tts_handler import HybridTTSHandler
from pepper_gui import PepperDearPyGUI
from web_search_handler import WebSearchHandler
from voice_handler import VoiceHandler, list_microphones
import config

# ── Global state ─────────────────────────────────────────────────────────────
robot_active = False
running      = True
pepper       = None
gui          = None
brain        = None
tts          = None
web_searcher = None
voice        = None          # VoiceHandler instance
ptt_active   = False         # Is push-to-talk key currently held?
last_movement_key_time = 0.0 # Watchdog: last time a movement key was seen

# Movement keys — held-state dict
movement_keys = {
    'w': False,  # forward
    's': False,  # backward
    'a': False,  # turn left
    'd': False,  # turn right
    'q': False,  # strafe left
    'e': False,  # strafe right
}

# PTT key (from config, default 'r')
PTT_KEY = config.PTT_KEY


# ── Keyboard handlers ─────────────────────────────────────────────────────────

def on_press(key):
    """Handle key-down events."""
    global robot_active, running, ptt_active

    try:
        k = key.char if hasattr(key, 'char') else None

        # ── PTT (push-to-talk) ────────────────────────────────────────────
        if k == PTT_KEY and config.VOICE_ENABLED:
            # Don't start recording while the user is typing in the GUI text box
            if gui and gui.text_input_focused:
                return  # Let 'r' through as a normal character
            if not ptt_active:
                ptt_active = True
                if voice:
                    started = voice.start_recording()
                    if started and gui:
                        gui.set_recording(True)
                        gui.update_status("🎙️ Recording… release R when done")
            return  # Don't fall through to other handlers

        # ── Movement ─────────────────────────────────────────────────────
        if k in movement_keys:
            movement_keys[k] = True
            global last_movement_key_time
            last_movement_key_time = time.time()

        # ── Gestures ─────────────────────────────────────────────────────
        elif k == '1':
            print("👋 Wave"); pepper.wave()
        elif k == '2':
            print("😊 Nod"); pepper.nod()
        elif k == '3':
            print("🙅 Shake head"); pepper.shake_head()
        elif k == '4':
            print("🤔 Thinking"); pepper.thinking_gesture()
        elif k == '8':
            print("✋ Explaining"); pepper.explaining_gesture()
        elif k == '9':
            print("🎉 Excited"); pepper.excited_gesture()
        elif k == '0':
            print("👉 Point"); pepper.point_forward()

        # ── LEDs ─────────────────────────────────────────────────────────
        elif k == '5':
            pepper.set_eye_color("blue")
        elif k == '6':
            pepper.set_eye_color("green")
        elif k == '7':
            pepper.set_eye_color("red")

        # ── System ───────────────────────────────────────────────────────
        elif k == ' ':
            robot_active = not robot_active
            status = "ACTIVE 🟢" if robot_active else "IDLE 🔴"
            print(f"\n{'='*50}\nPepper is now {status}\n{'='*50}\n")
            pepper.set_eye_color("blue" if robot_active else "white")
            if gui:
                gui.update_status("Active — ready" if robot_active else "Idle")

        elif k == 'x':
            print("\n👋 Shutting down…")
            running = False

    except AttributeError:
        pass  # Special keys (arrows, shift, etc.)


def on_release(key):
    """Handle key-up events."""
    global ptt_active

    try:
        k = key.char if hasattr(key, 'char') else None

        # ── PTT release → trigger transcription ──────────────────────────
        if k == PTT_KEY and config.VOICE_ENABLED:
            if ptt_active:   # Only stop if we actually started
                ptt_active = False
                if voice:
                    voice.stop_recording_and_transcribe()
                if gui:
                    gui.set_recording(False)
            return

        # ── Movement key released → stop that direction ───────────────────
        if k in movement_keys:
            movement_keys[k] = False

    except AttributeError:
        pass


def movement_controller():
    """Continuously check movement keys and move robot.
    Includes a 1-second watchdog: if no key event arrives within that window
    the robot is forced to stop (protects against dropped key-release events)."""
    global running, pepper, movement_keys, last_movement_key_time
    WATCHDOG_TIMEOUT = 1.0  # seconds

    while running:
        try:
            any_pressed = any(movement_keys.values())

            # Watchdog: force-stop if keys appear held but no event in 1s
            if any_pressed and (time.time() - last_movement_key_time > WATCHDOG_TIMEOUT):
                for k in movement_keys:
                    movement_keys[k] = False
                pepper.stop_movement()
                time.sleep(0.1)
                continue

            # Normal movement dispatch
            if movement_keys['w']:
                pepper.move_forward()
            elif movement_keys['s']:
                pepper.move_backward()
            elif movement_keys['a']:
                pepper.turn_left()
            elif movement_keys['d']:
                pepper.turn_right()
            elif movement_keys['q']:
                pepper.strafe_left()
            elif movement_keys['e']:
                pepper.strafe_right()
            else:
                pepper.stop_movement()

            time.sleep(0.1)  # 10Hz update rate
        except Exception as e:
            print(f"❌ Movement controller error: {e}")
            time.sleep(0.5)


def execute_function_calls(function_calls: list):
    """
    Execute robot functions returned by AI
    
    Returns:
        str or None: Search results if web_search was called, None otherwise
    """
    if not function_calls:
        return None
    
    search_results = None
    
    for func in function_calls:
        func_name = func['name']
        func_args = func.get('arguments', {})
        
        try:
            # Handle web search specially - return results
            if func_name == "web_search":
                query = func_args.get('query', '')
                if query:
                    print(f"🔍 AI requested web search: '{query}'")
                    search_results = web_searcher.search(query)
                else:
                    print("⚠️ Web search called without query")
            
            # Handle gesture functions
            elif func_name == "wave":
                pepper.wave()
            elif func_name == "nod":
                pepper.nod()
            elif func_name == "shake_head":
                pepper.shake_head()
            elif func_name == "thinking_gesture":
                pepper.thinking_gesture()
            elif func_name == "explaining_gesture":
                pepper.explaining_gesture()
            elif func_name == "excited_gesture":
                pepper.excited_gesture()
            elif func_name == "point_forward":
                pepper.point_forward()
            elif func_name == "shrug":
                pepper.shrug()
            elif func_name == "celebrate":
                pepper.celebrate()
            elif func_name == "look_around":
                pepper.look_around()
            elif func_name == "bow":
                pepper.bow()
            elif func_name == "look_at_sound":
                pepper.look_at_sound()
            else:
                print(f"⚠️ Unknown function: {func_name}")
        except Exception as e:
            print(f"❌ Error executing {func_name}: {e}")
    
    return search_results


def _speak_hq(text: str):
    """
    Speak using HQ TTS → Pepper speakers pipeline.
    Falls back to Pepper's built-in voice if the HQ pipeline fails.
    """
    try:
        audio_path = tts.speak(text)
        if audio_path and pepper.play_audio_file(audio_path):
            # Clean up temp file
            try:
                import os
                os.remove(audio_path)
            except Exception:
                pass
            return
    except Exception as e:
        print(f"⚠️ HQ TTS pipeline failed: {e}")

    # Fallback: use Pepper's built-in TTS
    pepper.speak(text)


def handle_gui_message(message: str):
    """
    Handle message from GUI
    Called when user sends a message in GUI
    """
    global robot_active, pepper, gui, brain
    
    try:
        if not robot_active:
            gui.update_status("Pepper is idle - Press SPACE to activate")
            gui.add_pepper_message("I'm currently idle. Press SPACE to wake me up!")
            return
        
        gui.update_status("Thinking...")
        
        # Show thinking indicator
        pepper.thinking_indicator(start=True)
        
        # Check for goodbye
        if config.GOODBYE_WORD.lower() in message.lower():
            response = "Goodbye! It was nice talking with you."
            robot_active = False
            pepper.thinking_indicator(start=False)
            pepper.wave()
            pepper.set_eye_color("white")
            gui.add_pepper_message(response)
            gui.update_status("Pepper is idle")
            _speak_hq(response)
            return
        
        # Process with AI
        response_text, function_calls = brain.chat(message)
        
        # Execute any function calls and check for search results
        search_results = None
        if function_calls:
            search_results = execute_function_calls(function_calls)

        # If AI triggered a web search, feed results back WITHOUT polluting history
        if search_results:
            gui.update_status("Processing search results...")
            print("📊 Passing search results back to AI (clean context)...")
            response_text, function_calls = brain.chat_with_context(
                user_message = message,
                context      = search_results,
            )
            # Execute any gesture calls that came with the final answer
            if function_calls:
                execute_function_calls(function_calls)

        # Stop thinking indicator
        pepper.thinking_indicator(start=False)
        
        # Speak and display response
        if response_text:
            gui.add_pepper_message(response_text)
            _speak_hq(response_text)
            gui.update_status("Ready")
        else:
            gui.add_pepper_message("Sorry, I didn't understand that.")
            gui.update_status("Ready")
    
    except Exception as e:
        print(f"❌ Message handling error: {e}")
        import traceback
        traceback.print_exc()
        pepper.thinking_indicator(start=False)
        gui.update_status("Error - Ready")
        gui.add_pepper_message("Sorry, I encountered an error.")


def print_controls():
    """Print control instructions"""
    ptt = PTT_KEY.upper()
    print("\n" + "="*60)
    print("🎮 PEPPER ROBOT CONTROLS")
    print("="*60)
    print(f"\n🎙️ VOICE (Push-to-Talk):")
    print(f"  Hold {ptt}     - Speak  →  release  →  auto-transcribes")
    print("\n💬 TEXT:")
    print("  Type in GUI window + Enter / Send button")
    print("\n🤖 MOVEMENT (hold key):")
    print("  W/S     - Forward / Backward")
    print("  A/D     - Turn Left / Right")
    print("  Q/E     - Strafe Left / Right")
    print("\n✋ MANUAL GESTURES (tap):")
    print("  1=Wave  2=Nod  3=Shake  4=Think  8=Explain  9=Excited  0=Point")
    print("\n💡 LEDs:")
    print("  5=Blue  6=Green  7=Red")
    print("\n⚙️ SYSTEM:")
    print("  SPACE   - Toggle Active / Idle")
    print("  X       - Quit")
    print(f"\n  Wake: '{config.WAKE_WORD}'   Goodbye: '{config.GOODBYE_WORD}'")
    print("\n🧠 AI triggers gestures automatically during conversation!")
    print("="*60 + "\n")


def main():
    """Main program"""
    global pepper, running, brain, tts, gui, web_searcher, voice

    print("\n🤖 PEPPER AI ROBOT — Phase 1 + Web Search + Voice")
    print("=" * 60)

    # 1. Groq API ──────────────────────────────────────────────────────
    print("\n1️⃣  Testing Groq API…")
    if not test_groq_connection(config.GROQ_API_KEY):
        print("❌ Groq API test failed. Check your API key in .env")
        return

    # 2. Pepper ────────────────────────────────────────────────────────
    print("\n2️⃣  Connecting to Pepper…")
    pepper = PepperRobot(config.PEPPER_IP, config.PEPPER_PORT)
    if not pepper.connect():
        print("❌ Failed to connect. Check PEPPER_IP in .env")
        return
    pepper.set_eye_color("white")

    # 3. AI Brain ──────────────────────────────────────────────────────
    print("\n3️⃣  Initialising AI brain…")
    brain = GroqBrain(
        api_key       = config.GROQ_API_KEY,
        llm_model     = config.GROQ_LLM_MODEL,
        whisper_model = config.GROQ_WHISPER_MODEL,
        system_prompt = config.SYSTEM_PROMPT,
        functions     = config.ROBOT_FUNCTIONS,
        use_web_search = config.USE_WEB_SEARCH,
        compound_model = config.GROQ_COMPOUND_MODEL,
    )

    # 4. TTS ───────────────────────────────────────────────────────────
    print("\n4️⃣  Initialising TTS…")
    tts = HybridTTSHandler(
        groq_api_key      = config.GROQ_API_KEY,
        groq_voice        = "hannah",
        elevenlabs_api_key = config.ELEVENLABS_API_KEY,
        elevenlabs_voice  = "Rachel",
        edge_voice        = config.TTS_VOICE,
        edge_rate         = config.TTS_RATE,
    )
    print("   Note: using Pepper's built-in TTS for speech output")

    # 5. Web Search ────────────────────────────────────────────────────
    print("\n5️⃣  Initialising web search…")
    web_searcher = WebSearchHandler(max_results=3)
    print("   ✅ DuckDuckGo search ready (free, unlimited)")

    # 6. Voice Handler ─────────────────────────────────────────────────
    print("\n6️⃣  Initialising voice (STT)…")
    if config.VOICE_ENABLED:
        try:
            VoiceHandler.validate_setup()
        except RuntimeError as e:
            print(f"   ❌ Voice pre-check failed: {e}")
            print("   ⚠️  Voice disabled due to missing dependencies")
            config.VOICE_ENABLED = False

    if config.VOICE_ENABLED:
        list_microphones()

        voice = VoiceHandler(
            transcribe_fn  = brain.transcribe_audio,
            sample_rate    = config.AUDIO_SAMPLE_RATE,
            channels       = config.AUDIO_CHANNELS,
            min_duration   = config.AUDIO_MIN_DURATION,
            max_duration   = config.AUDIO_MAX_DURATION,
        )

        # Wire up callbacks — all thread-safe via GUI queue
        def _on_start():
            gui and gui.set_recording(True)

        def _on_stop():
            gui and gui.set_recording(False)

        def _on_transcribing():
            gui and gui.update_status("🔄 Transcribing…")

        def _on_transcribed(text: str):
            """
            Called from VoiceHandler background thread after Whisper returns.
            We queue the transcribed text into the GUI as a voice message.
            The GUI will render it AND kick off handle_gui_message(text).
            """
            print(f"📝 Transcribed: \"{text}\"")
            if gui:
                gui.add_voice_user_message(text)

        def _on_error(msg: str):
            print(f"🎙️ Voice error: {msg}")
            gui and gui.update_status(f"Voice error: {msg}")
            gui and gui.set_recording(False)

        voice.on_recording_start = _on_start
        voice.on_recording_stop  = _on_stop
        voice.on_transcribing    = _on_transcribing
        voice.on_transcribed     = _on_transcribed
        voice.on_error           = _on_error

        print(f"   ✅ Push-to-talk ready (hold '{PTT_KEY.upper()}' to speak)")
    else:
        print("   ⚠️  Voice disabled in config (VOICE_ENABLED = False)")

    print("\n✅ All systems ready!")

    # 7. DearPyGUI — start keyboard/movement threads first ─────────────
    print(f"\n7️⃣  Starting DearPyGUI…")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    movement_thread = threading.Thread(target=movement_controller, daemon=True)
    movement_thread.start()

    print_controls()

    gui = PepperDearPyGUI(handle_gui_message)
    gui.update_status("Idle — press SPACE to activate Pepper")

    try:
        gui.start()          # Blocks until window closed
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        running = False
        listener.stop()
        if gui:
            gui.stop()
        pepper.disconnect()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()