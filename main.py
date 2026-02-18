"""
Main Control Script — Pepper AI Robot

Changes from original:
- All mutable globals consolidated into a `state` SimpleNamespace so shared
  state is explicit and easier to reason about across threads.
- message_lock: only one AI request processed at a time. Concurrent calls
  (e.g. voice + text arriving simultaneously) are rejected with a friendly
  "busy" message rather than corrupting conversation history.
- ptt_lock: threading.Lock prevents a double-fire start if the PTT key
  somehow generates two press events before a release.
- execute_function_calls() split into execute_gestures() (fire-and-forget
  robot actions) and execute_search() (returns results string or None).
  Concerns are separated; gestres never accidentally return data.
- Smart search path: brain.needs_search(message) does a local keyword check.
  If True, we run the search first and call chat_with_context() — ONE LLM
  call total instead of two.  The model-driven fallback is still present for
  edge cases not caught by keywords.
- Movement controller: only sends NAOqi commands when the key state CHANGES,
  not every 100ms.  Watchdog safety logic is preserved.
- Speaking status: GUI status shows "🔊 Speaking…" during audio playback and
  resets to "Ready" when done.
- Voice error path now always resets recording indicator and status line.
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
    robot_active          = False,
    running               = True,
    ptt_active            = False,
    last_movement_key_time = 0.0,
    # Prevents concurrent AI/speech calls from corrupting history
    message_lock          = threading.Lock(),
    # Prevents PTT double-fire on key-repeat
    ptt_lock              = threading.Lock(),
)

# Component handles (set in main())
pepper:       PepperRobot      = None
gui:          PepperDearPyGUI  = None
brain:        GroqBrain        = None
tts:          HybridTTSHandler = None
web_searcher: WebSearchHandler = None
voice:        VoiceHandler     = None

# Movement key state
movement_keys = {k: False for k in ('w', 's', 'a', 'd', 'q', 'e')}
PTT_KEY = config.PTT_KEY


# ── Function-call helpers ─────────────────────────────────────────────────────

def execute_gestures(function_calls: list):
    """
    Execute only physical gesture calls returned by the AI.
    Non-blocking: each gesture runs in its own daemon thread (see PepperRobot).
    """
    if not function_calls:
        return
    gesture_map = {
        "wave":              pepper.wave,
        "nod":               pepper.nod,
        "shake_head":        pepper.shake_head,
        "thinking_gesture":  pepper.thinking_gesture,
        "explaining_gesture": pepper.explaining_gesture,
        "excited_gesture":   pepper.excited_gesture,
        "point_forward":     pepper.point_forward,
        "shrug":             pepper.shrug,
        "celebrate":         pepper.celebrate,
        "look_around":       pepper.look_around,
        "bow":               pepper.bow,
        "look_at_sound":     pepper.look_at_sound,
    }
    for fn in function_calls:
        name = fn.get("name", "")
        if name in gesture_map:
            try:
                gesture_map[name]()
            except Exception as e:
                print(f"❌ Gesture '{name}' error: {e}")
        elif name != "web_search":
            print(f"⚠️  Unknown function: {name}")


def execute_search(function_calls: list) -> Optional[str]:
    """
    Execute the web_search call if present in function_calls.
    Returns the formatted search results string, or None if no search was requested.
    """
    if not function_calls:
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
    """
    Process a user message: AI inference → optional web search → speech.

    A non-blocking lock ensures only one message is processed at a time.
    Concurrent calls are dropped with a friendly status message.
    """
    # Reject if we're already processing something
    if not state.message_lock.acquire(blocking=False):
        if gui:
            gui.update_status("⏳ Still processing — please wait")
            gui.add_system_message("(Busy processing previous message…)")
        return

    # Tracks whether the finally block should reset status to "Ready".
    # Set False on paths that leave Pepper in a non-ready state (idle, goodbye).
    _reset_status = True

    try:
        if not state.robot_active:
            _reset_status = False
            if gui:
                gui.update_status("Pepper is idle — press SPACE to activate")
                gui.add_pepper_message("I'm currently idle. Press SPACE to wake me up!")
            return

        if gui:
            gui.update_status("Thinking…")
        pepper.thinking_indicator(start=True)

        # ── Goodbye shortcut ─────────────────────────────────────────
        if config.GOODBYE_WORD.lower() in message.lower():
            _reset_status = False
            _say("Goodbye! It was nice talking with you.")
            state.robot_active = False
            pepper.thinking_indicator(start=False)
            pepper.wave()
            pepper.set_eye_color("white")
            if gui:
                gui.update_status("Pepper is idle")
            return

        # ── Smart search fast path ───────────────────────────────────
        # If the message clearly needs current info, search FIRST so we
        # only need ONE LLM call (with context) instead of two.
        if brain.needs_search(message):
            if gui:
                gui.update_status("🔍 Searching web…")
            search_results = web_searcher.search(message)
            response_text, function_calls = brain.chat_with_context(
                user_message=message,
                context=search_results,
            )
            pepper.thinking_indicator(start=False)
            # Gestures only — search is already done
            execute_gestures(function_calls)

        else:
            # ── Standard LLM call ────────────────────────────────────
            response_text, function_calls = brain.chat(message)

            # Model may have decided to search on its own
            search_results = execute_search(function_calls)

            if search_results:
                # Second call with context (model-driven search path)
                if gui:
                    gui.update_status("🔍 Processing search results…")
                response_text, function_calls = brain.chat_with_context(
                    user_message=message,
                    context=search_results,
                )
                pepper.thinking_indicator(start=False)
                execute_gestures(function_calls)
            else:
                pepper.thinking_indicator(start=False)
                execute_gestures(function_calls)

        # ── Speak & display ──────────────────────────────────────────
        if response_text:
            if gui:
                gui.add_pepper_message(response_text)
            _say(response_text)
        else:
            fallback = "Sorry, I didn't catch that."
            if gui:
                gui.add_pepper_message(fallback)
            _say(fallback)

    except Exception as e:
        print(f"❌ Message handling error: {e}")
        import traceback; traceback.print_exc()
        pepper.thinking_indicator(start=False)
        if gui:
            gui.update_status("Error — Ready")
            gui.add_pepper_message("Sorry, I encountered an error.")
    finally:
        state.message_lock.release()
        if gui and _reset_status:
            gui.update_status("Ready")


def _say(text: str):
    """
    Speak text via HQ pipeline (or NAOqi fallback) and update GUI status.
    Always called from a worker thread — never blocks the GUI.
    """
    if gui:
        gui.update_status("🔊 Speaking…")
    try:
        pepper.speak_hq(text, tts)
    finally:
        if gui:
            gui.update_status("Ready")


# ── Keyboard handlers ─────────────────────────────────────────────────────────

def on_press(key):
    try:
        # ── Typing guard ─────────────────────────────────────────────
        # If the user is typing in the GUI text box, let ALL keys pass
        # through as normal characters — don't trigger any robot commands.
        # Exception: Escape always works so the user can quit even while
        # the text box is focused.
        is_typing = bool(gui and gui.text_input_focused)

        # ── Escape — quit (always active, even while typing) ─────────
        if key == keyboard.Key.esc:
            print("\n👋 Shutting down…")
            state.running = False
            return

        # Block all other robot commands while the text box is focused
        if is_typing:
            return

        # Extract character for char-key branches
        k = key.char if hasattr(key, "char") and key.char else None

        # ── Space — toggle active/idle ────────────────────────────────
        # Space is keyboard.Key.space (special key), NOT key.char == ' '
        if key == keyboard.Key.space:
            state.robot_active = not state.robot_active
            label = "ACTIVE 🟢" if state.robot_active else "IDLE 🔴"
            print(f"\n{'='*50}\nPepper is now {label}\n{'='*50}\n")
            pepper.set_eye_color("blue" if state.robot_active else "white")
            if gui:
                gui.update_status("Active — ready" if state.robot_active else "Idle")
            return

        if k is None:
            return  # Unhandled special key — ignore

        # ── PTT ──────────────────────────────────────────────────────
        if k == PTT_KEY and config.VOICE_ENABLED:
            # Use lock to prevent double-fire on key-repeat
            acquired = state.ptt_lock.acquire(blocking=False)
            if not acquired:
                return  # Already recording

            if not state.ptt_active:
                state.ptt_active = True
                if voice:
                    started = voice.start_recording()
                    if started and gui:
                        gui.set_recording(True)
                        gui.update_status("🎙️ Recording… release R when done")
                    else:
                        # Failed to start — release lock
                        state.ptt_active = False
                        state.ptt_lock.release()
            return

        # ── Movement ─────────────────────────────────────────────────
        if k in movement_keys:
            movement_keys[k]            = True
            state.last_movement_key_time = time.time()

        # ── Gestures ─────────────────────────────────────────────────
        elif k == '1': pepper.wave()
        elif k == '2': pepper.nod()
        elif k == '3': pepper.shake_head()
        elif k == '4': pepper.thinking_gesture()
        elif k == '8': pepper.explaining_gesture()
        elif k == '9': pepper.excited_gesture()
        elif k == '0': pepper.point_forward()

        # ── LEDs ─────────────────────────────────────────────────────
        elif k == '5': pepper.set_eye_color("blue")
        elif k == '6': pepper.set_eye_color("green")
        elif k == '7': pepper.set_eye_color("red")

    except AttributeError:
        pass


def on_release(key):
    try:
        # ── Escape (already handled in on_press, nothing to release) ──
        if key == keyboard.Key.esc:
            return

        k = key.char if hasattr(key, "char") and key.char else None

        # ── PTT release ──────────────────────────────────────────────
        # PTT release always fires regardless of text_input_focused so
        # that recording is never left open if the user clicks into the
        # text box mid-recording and then releases R.
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
                    pass  # Already released somehow
            return

        # ── Movement key released ─────────────────────────────────────
        # Always reset regardless of typing state — prevents a key getting
        # "stuck" held if the user clicks into the text box while moving.
        if k in movement_keys:
            movement_keys[k] = False

    except AttributeError:
        pass


# ── Movement controller ───────────────────────────────────────────────────────

def movement_controller():
    """
    Sends NAOqi move commands only when the active-key state CHANGES.
    The watchdog still fires after 1s of silence to handle dropped releases.
    """
    WATCHDOG_TIMEOUT = 1.0
    prev_state       = None

    while state.running:
        try:
            any_pressed = any(movement_keys.values())

            # Watchdog: force-stop if keys look held but no recent event
            if any_pressed and (time.time() - state.last_movement_key_time > WATCHDOG_TIMEOUT):
                for k in movement_keys:
                    movement_keys[k] = False
                pepper.stop_movement()
                prev_state = None
                time.sleep(0.1)
                continue

            # Snapshot current state
            current = (
                movement_keys['w'], movement_keys['s'],
                movement_keys['a'], movement_keys['d'],
                movement_keys['q'], movement_keys['e'],
            )

            # Only dispatch when something changed
            if current != prev_state:
                w, s, a, d, q, e = current
                if   w: pepper.move_forward()
                elif s: pepper.move_backward()
                elif a: pepper.turn_left()
                elif d: pepper.turn_right()
                elif q: pepper.strafe_left()
                elif e: pepper.strafe_right()
                else:   pepper.stop_movement()
                prev_state = current

            time.sleep(0.05)   # 20 Hz polling (only to catch watchdog)

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
    print(f"  Hold {ptt}     - Speak  →  release  →  auto-transcribes")
    print("\n💬 TEXT:")
    print("  Click the GUI text box and type, then Enter / Send")
    print("  (robot controls are disabled while typing in the box)")
    print("\n🤖 MOVEMENT (hold key — focus must be outside text box):")
    print("  W/S     - Forward / Backward")
    print("  A/D     - Turn Left / Right")
    print("  Q/E     - Strafe Left / Right")
    print("\n✋ MANUAL GESTURES (tap):")
    print("  1=Wave  2=Nod  3=Shake  4=Think  8=Explain  9=Excited  0=Point")
    print("\n💡 LEDs:")
    print("  5=Blue  6=Green  7=Red")
    print("\n⚙️ SYSTEM:")
    print("  SPACE   - Toggle Active / Idle  (click outside text box first)")
    print("  ESC     - Quit  (works even while typing)")
    print(f"\n  Wake: '{config.WAKE_WORD}'   Goodbye: '{config.GOODBYE_WORD}'")
    print("\n🧠 AI triggers gestures automatically during conversation!")
    print("="*60 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global pepper, brain, tts, gui, web_searcher, voice

    print("\n🤖 PEPPER AI ROBOT — Phase 2 (Voice + Safety)")
    print("=" * 60)

    # 1. Groq API ──────────────────────────────────────────────────────
    print("\n1️⃣  Testing Groq API…")
    if not test_groq_connection(config.GROQ_API_KEY):
        print("❌ Groq API test failed. Check your API key in .env")
        return

    # 2. Pepper ────────────────────────────────────────────────────────
    print("\n2️⃣  Connecting to Pepper…")
    pepper = PepperRobot(config.PEPPER_IP, config.PEPPER_PORT,
                         ssh_user=config.PEPPER_SSH_USER,
                         ssh_password=config.PEPPER_SSH_PASS)
    if not pepper.connect():
        print("❌ Failed to connect. Check PEPPER_IP in .env")
        return
    pepper.set_eye_color("white")

    # 3. AI Brain ──────────────────────────────────────────────────────
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

    # 4. TTS ───────────────────────────────────────────────────────────
    print("\n4️⃣  Initialising TTS…")
    tts = HybridTTSHandler(
        groq_api_key       = config.GROQ_API_KEY,
        groq_voice         = "hannah",
        elevenlabs_api_key = config.ELEVENLABS_API_KEY,
        edge_voice         = config.TTS_VOICE,
        edge_rate          = config.TTS_RATE,
    )

    # 5. Web Search ────────────────────────────────────────────────────
    print("\n5️⃣  Initialising web search…")
    web_searcher = WebSearchHandler(max_results=3, timeout=8.0)
    print("   ✅ DuckDuckGo search ready (free, unlimited, 8s timeout)")

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
                # add_voice_user_message queues the message_callback call

        def _on_error(msg: str):
            print(f"🎙️ Voice error: {msg}")
            if gui:
                # Always reset both status and recording indicator on error
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

    # 7. Start keyboard + movement threads, then GUI ───────────────────
    print(f"\n7️⃣  Starting DearPyGUI…")

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    kb_listener.start()

    move_thread = threading.Thread(target=movement_controller, daemon=True)
    move_thread.start()

    print_controls()

    gui = PepperDearPyGUI(handle_gui_message)
    gui.update_status("Idle — press SPACE to activate Pepper")

    try:
        gui.start()         # Blocks until window closed
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        state.running = False
        kb_listener.stop()
        if gui:
            gui.stop()
        if pepper:
            pepper.disconnect()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()