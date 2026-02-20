"""
Main Control Script — Pepper AI Robot
"""

import threading
import time
from types import SimpleNamespace
from typing import Optional

from pynput import keyboard

import config
from groq_brain import GroqBrain, test_groq_connection
from hybrid_tts_handler import HybridTTSHandler
from pepper_gui import PepperDearPyGUI
from pepper_interface import PepperRobot
from voice_handler import VoiceHandler, list_microphones
from web_search_handler import WebSearchHandler

# ── Shared state ──────────────────────────────────────────────────────────────
state = SimpleNamespace(
    robot_active           = False,
    running                = True,
    ptt_active             = False,
    last_movement_key_time = 0.0,
    message_lock           = threading.Lock(),
    ptt_lock               = threading.Lock(),
)

# Component handles (set in main())
pepper:       Optional[PepperRobot]      = None
gui:          Optional[PepperDearPyGUI]  = None
brain:        Optional[GroqBrain]        = None
tts:          Optional[HybridTTSHandler] = None
web_searcher: Optional[WebSearchHandler] = None
voice:        Optional[VoiceHandler]     = None

movement_keys = {k: False for k in ('w', 's', 'a', 'd', 'q', 'e')}
PTT_KEY = config.PTT_KEY


# ── Pepper guard helper ───────────────────────────────────────────────────────

def _pepper_ok() -> bool:
    return pepper is not None and pepper.connected


# ── Volume callback ───────────────────────────────────────────────────────────

def on_action(action: str):
    """Called from GUI action buttons (e.g. Pulse Eyes)."""
    if action == "pulse_eyes":
        if _pepper_ok():
            pepper.pulse_eyes("blue", duration=2.0)
        else:
            print("⚠️  Pepper not connected — can't pulse eyes")


def on_volume_changed(volume: int):
    """Called from the GUI volume slider. Routes to Pepper hardware if connected."""
    if _pepper_ok():
        pepper.set_volume(volume)


# ── Function-call helpers ─────────────────────────────────────────────────────

def execute_gestures(function_calls: list) -> Optional[str]:
    """Execute gesture function calls. Returns detected emotion string or None."""
    emotion = None
    if not function_calls:
        return emotion

    gesture_map = {
        "wave":               pepper.wave,
        "nod":                pepper.nod,
        "shake_head":         pepper.shake_head,
        "thinking_gesture":   pepper.thinking_gesture,
        "explaining_gesture": pepper.explaining_gesture,
        "excited_gesture":    pepper.excited_gesture,
        "point_forward":      pepper.point_forward,
        "shrug":              pepper.shrug,
        "celebrate":          pepper.celebrate,
        "look_around":        pepper.look_around,
        "bow":                pepper.bow,
        "look_at_sound":      pepper.look_at_sound,
    } if _pepper_ok() else {}

    for fn in function_calls:
        name = fn.get("name", "")
        if name == "express_emotion":
            emotion = fn.get("arguments", {}).get("emotion", None)
        elif name == "web_search":
            pass  # handled by execute_search
        elif name in gesture_map:
            try:
                gesture_map[name]()
            except Exception as e:
                print(f"❌ Gesture '{name}' error: {e}")
        else:
            print(f"⚠️  Unknown function: {name}")

    return emotion


def execute_search(function_calls: list) -> Optional[str]:
    if not function_calls or not config.USE_WEB_SEARCH:
        return None
    for fn in function_calls:
        if fn.get("name") == "web_search":
            query = fn.get("arguments", {}).get("query", "").strip()
            if query:
                print(f"🔍 AI requested web search: '{query}'")
                return web_searcher.search(query)
    return None


# ── Message handler ───────────────────────────────────────────────────────────

def handle_gui_message(message: str):
    if not state.message_lock.acquire(blocking=False):
        if gui:
            gui.update_status("⏳ Still processing — please wait")
            gui.add_system_message("(Busy processing previous message…)")
        return

    _thinking_started = False
    _reset_status     = True

    try:
        if not state.robot_active:
            _reset_status = False
            if gui:
                gui.update_status("Pepper is idle — press SPACE to activate")
                gui.add_pepper_message("I'm currently idle. Press SPACE to wake me up!")
            return

        # Goodbye shortcut — checked before starting thinking indicator
        if config.GOODBYE_WORD.lower() in message.lower():
            _reset_status = False
            _say("Goodbye! It was nice talking with you.")
            state.robot_active = False
            if _pepper_ok():
                pepper.wave()
                pepper.set_eye_color("white")
            if gui:
                gui.update_status("Pepper is idle")
                gui.set_robot_active(False)
            return

        emotion = None   # set by execute_gestures if model calls express_emotion

        if gui:
            gui.update_status("Thinking…")
        if _pepper_ok():
            pepper.thinking_indicator(start=True)
        _thinking_started = True

        if config.USE_WEB_SEARCH and brain.needs_search(message):
            if gui:
                gui.update_status("🔍 Searching web…")
            search_results = web_searcher.search(message)
            response_text, function_calls = brain.chat_with_context(
                user_message=message,
                context=search_results,
            )
            if _pepper_ok():
                pepper.thinking_indicator(start=False)
            _thinking_started = False
            emotion = execute_gestures(function_calls)

        else:
            response_text, function_calls = brain.chat(message)
            search_results = execute_search(function_calls)

            if search_results:
                if gui:
                    gui.update_status("🔍 Processing search results…")
                response_text, function_calls = brain.chat_with_context(
                    user_message=message,
                    context=search_results,
                )
                if _pepper_ok():
                    pepper.thinking_indicator(start=False)
                _thinking_started = False
                emotion = execute_gestures(function_calls)
            else:
                if _pepper_ok():
                    pepper.thinking_indicator(start=False)
                _thinking_started = False
                emotion = execute_gestures(function_calls)

        if response_text:
            if gui:
                gui.add_pepper_message(response_text)
            _say(response_text, emotion=emotion)
        else:
            fallback = "Sorry, I didn't catch that."
            if gui:
                gui.add_pepper_message(fallback)
            _say(fallback)

    except Exception as e:
        print(f"❌ Message handling error: {e}")
        import traceback; traceback.print_exc()
        if _thinking_started and _pepper_ok():
            pepper.thinking_indicator(start=False)
        if gui:
            gui.update_status("Error — Ready")
            gui.add_pepper_message("Sorry, I encountered an error.")
    finally:
        state.message_lock.release()
        if gui and _reset_status:
            gui.update_status("Ready")


def _say(text: str, emotion: Optional[str] = None):
    if gui and gui.is_running:
        gui.update_status("🔊 Speaking…")
    try:
        if _pepper_ok():
            pepper.speak_hq(text, tts, emotion=emotion)
        else:
            if tts:
                tts.speak_and_play(text, emotion=emotion)
    finally:
        if gui and gui.is_running:
            gui.update_status("Ready")


# ── Keyboard handlers ─────────────────────────────────────────────────────────

def on_press(key):
    try:
        # Universal cursor gate — text box focused means all keys go to typing
        if gui and gui.text_input_focused:
            return

        k = key.char if hasattr(key, "char") and key.char else None

        # ESC — quit
        if key == keyboard.Key.esc:
            print("\n👋 Shutting down…")
            state.running = False
            if gui:
                gui.is_running = False
            return

        # SPACE — toggle active/idle
        if key == keyboard.Key.space:
            state.robot_active = not state.robot_active
            label = "ACTIVE 🟢" if state.robot_active else "IDLE 🔴"
            print(f"\n{'='*50}\nPepper is now {label}\n{'='*50}\n")
            if _pepper_ok():
                pepper.set_eye_color("blue" if state.robot_active else "white")
            if gui:
                gui.update_status("Active — ready" if state.robot_active else "Idle")
                gui.set_robot_active(state.robot_active)
            return

        if k is None:
            return

        # PTT
        if k == PTT_KEY and config.VOICE_ENABLED:
            acquired = state.ptt_lock.acquire(blocking=False)
            if not acquired:
                return
            if state.ptt_active:
                state.ptt_lock.release()
                return
            state.ptt_active = True
            if voice:
                started = voice.start_recording()
                if started and gui:
                    gui.set_recording(True)
                    gui.update_status("🎙️ Recording… release R when done")
                else:
                    state.ptt_active = False
                    state.ptt_lock.release()
            return

        # Movement keys
        if k in movement_keys:
            movement_keys[k]             = True
            state.last_movement_key_time = time.time()
            return

        # Gesture / LED keys — silently ignored when not connected
        if not _pepper_ok():
            return

        if   k == '1': pepper.wave()
        elif k == '2': pepper.nod()
        elif k == '3': pepper.shake_head()
        elif k == '4': pepper.thinking_gesture()
        elif k == '8': pepper.explaining_gesture()
        elif k == '9': pepper.excited_gesture()
        elif k == '0': pepper.point_forward()
        elif k == '5': pepper.set_eye_color("blue")
        elif k == '6': pepper.set_eye_color("green")
        elif k == '7': pepper.set_eye_color("red")

    except AttributeError:
        pass


def on_release(key):
    try:
        k = key.char if hasattr(key, "char") and key.char else None

        # PTT release — always processed regardless of text focus
        if k == PTT_KEY and config.VOICE_ENABLED:
            if state.ptt_active:
                state.ptt_active = False
                if voice:
                    voice.stop_recording_and_transcribe()
                if gui:
                    gui.set_recording(False)
                try:
                    state.ptt_lock.release()
                except RuntimeError:
                    pass
            return

        if gui and gui.text_input_focused:
            return

        if k in movement_keys:
            movement_keys[k]             = False
            state.last_movement_key_time = time.time()

    except AttributeError:
        pass


# ── Movement controller ───────────────────────────────────────────────────────

def movement_controller():
    """
    Sends moveToward() continuously at 10 Hz while any key is held.
    NAOqi's internal watchdog stops movement if no command arrives within ~1s,
    so we keep feeding it on every loop iteration.
    """
    WATCHDOG_TIMEOUT = 2.0
    SEND_INTERVAL    = 0.1   # 10 Hz
    prev_any         = False

    while state.running:
        try:
            if not _pepper_ok() or not state.robot_active:
                time.sleep(SEND_INTERVAL)
                continue

            any_pressed = any(movement_keys.values())

            # Watchdog — dropped key-up guard
            if any_pressed and (time.time() - state.last_movement_key_time > WATCHDOG_TIMEOUT):
                for k in movement_keys:
                    movement_keys[k] = False
                pepper.stop_movement()
                prev_any = False
                time.sleep(SEND_INTERVAL)
                continue

            if any_pressed:
                # Additive axes — allows simultaneous W+A (forward+turn left) etc.
                x     =  0.6 if movement_keys['w'] else -0.6 if movement_keys['s'] else 0.0
                theta =  0.5 if movement_keys['a'] else -0.5 if movement_keys['d'] else 0.0
                y     =  0.4 if movement_keys['q'] else -0.4 if movement_keys['e'] else 0.0
                pepper._move(x, y, theta)
            elif prev_any:
                pepper.stop_movement()

            prev_any = any_pressed
            time.sleep(SEND_INTERVAL)

        except Exception as ex:
            print(f"❌ Movement controller error: {ex}")
            time.sleep(0.5)


# ── Controls summary ──────────────────────────────────────────────────────────

def print_controls():
    ptt = PTT_KEY.upper()
    print("\n" + "="*60)
    print("🎮 PEPPER ROBOT CONTROLS")
    print("="*60)
    print(f"\n🎙️ VOICE (Push-to-Talk):")
    print(f"  Hold {ptt}     - Speak → release → auto-transcribes")
    print("\n💬 TEXT:")
    print("  Click the GUI input box to type (robot controls suspended)")
    print("  Press Enter or Send to send (robot controls restored)")
    print("\n🤖 MOVEMENT (input box must NOT be focused):")
    print("  W/S     - Forward / Backward")
    print("  A/D     - Turn Left / Right")
    print("  Q/E     - Strafe Left / Right")
    print("\n✋ GESTURES (tap, input box not focused):")
    print("  1=Wave  2=Nod  3=Shake  4=Think  8=Explain  9=Excited  0=Point")
    print("\n💡 LEDs:")
    print("  5=Blue  6=Green  7=Red")
    print("\n⚙️ SYSTEM:")
    print("  SPACE   - Toggle Active / Idle  (input box must NOT be focused)")
    print("  ESC     - Quit")
    print(f"\n  Wake: '{config.WAKE_WORD}'   Goodbye: '{config.GOODBYE_WORD}'")
    print("\n🧠 AI triggers gestures automatically during conversation!")
    print("="*60 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global pepper, brain, tts, gui, web_searcher, voice

    print("\n🤖 PEPPER AI ROBOT — Phase 2 (Voice + Safety)")
    print("=" * 60)

    print("\n1️⃣  Testing Groq API…")
    if not test_groq_connection(config.GROQ_API_KEY):
        print("❌ Groq API test failed. Check your API key in .env")
        return

    print("\n2️⃣  Connecting to Pepper…")
    pepper = PepperRobot(config.PEPPER_IP, config.PEPPER_PORT,
                         ssh_user=config.PEPPER_SSH_USER,
                         ssh_password=config.PEPPER_SSH_PASS)
    pepper.connect()

    print("\n3️⃣  Initialising AI brain…")
    brain = GroqBrain(
        api_key        = config.GROQ_API_KEY,
        llm_model      = config.GROQ_LLM_MODEL,
        whisper_model  = config.GROQ_WHISPER_MODEL,
        system_prompt  = config.SYSTEM_PROMPT,
        functions      = config.ROBOT_FUNCTIONS,
        use_web_search = config.USE_WEB_SEARCH,
        compound_model = config.GROQ_COMPOUND_MODEL,
    )

    print("\n4️⃣  Initialising TTS…")
    tts = HybridTTSHandler(
        groq_api_key       = config.GROQ_API_KEY,
        groq_voice         = "hannah",
        elevenlabs_api_key = config.ELEVENLABS_API_KEY,
        edge_voice         = config.TTS_VOICE,
        edge_rate          = config.TTS_RATE,
    )

    print("\n5️⃣  Initialising web search…")
    web_searcher = WebSearchHandler(max_results=3, timeout=8.0)
    search_status = "enabled" if config.USE_WEB_SEARCH else "disabled (USE_WEB_SEARCH=False)"
    print(f"   ✅ DuckDuckGo search ready — {search_status}")

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

        def _on_start():
            if gui: gui.set_recording(True)

        def _on_stop():
            if gui: gui.set_recording(False)

        def _on_transcribing():
            if gui: gui.update_status("🔄 Transcribing…")

        def _on_transcribed(text: str):
            print(f"📝 Transcribed: \"{text}\"")
            if gui:
                gui.add_voice_user_message(text)

        def _on_error(msg: str):
            print(f"🎙️ Voice error: {msg}")
            if gui:
                gui.set_recording(False)
                gui.update_status(f"Voice error: {msg}")

        voice.on_recording_start = _on_start
        voice.on_recording_stop  = _on_stop
        voice.on_transcribing    = _on_transcribing
        voice.on_transcribed     = _on_transcribed
        voice.on_error           = _on_error

        print(f"   ✅ Push-to-talk ready (hold '{PTT_KEY.upper()}' to speak)")
    else:
        print("   ⚠️  Voice disabled (VOICE_ENABLED = False in config)")

    print("\n✅ All systems ready!")
    print(f"\n7️⃣  Starting DearPyGUI…")

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    kb_listener.start()

    move_thread = threading.Thread(target=movement_controller, daemon=True)
    move_thread.start()

    print_controls()

    gui = PepperDearPyGUI(
        handle_gui_message,
        volume_callback = on_volume_changed,
        action_callback = on_action,
    )

    if _pepper_ok():
        gui.update_status("Idle — press SPACE to activate Pepper")
    else:
        gui.update_status("⚠️ Pepper offline — chat only mode")

    try:
        gui.start()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        state.running = False
        kb_listener.stop()
        if gui:
            gui.stop()
        if pepper:
            if pepper.connected:
                pepper.set_volume(40)
            pepper.disconnect()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()