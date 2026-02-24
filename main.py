"""
Main Control Script — Pepper AI Robot
"""

import logging
import queue
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Optional

from pynput import keyboard

import config
from groq_brain import GroqBrain, test_groq_connection
from hybrid_tts_handler import HybridTTSHandler
from pepper_display import PepperDisplayManager
from pepper_gui import PepperDearPyGUI
from pepper_interface import PepperRobot
from voice_handler import VoiceHandler, list_microphones
from web_search_handler import WebSearchHandler

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt= "%H:%M:%S",
)

# ── Shared state ──────────────────────────────────────────────────────────────
state = SimpleNamespace(
    robot_active           = False,
    running                = True,
    ptt_active             = False,
    last_movement_key_time = 0.0,
    message_lock           = threading.Lock(),
    message_queue          = queue.Queue(maxsize=3),
    ptt_lock               = threading.Lock(),
)

# Component handles (set in main())
pepper:          Optional[PepperRobot]         = None
gui:             Optional[PepperDearPyGUI]     = None
brain:           Optional[GroqBrain]           = None
tts:             Optional[HybridTTSHandler]    = None
web_searcher:    Optional[WebSearchHandler]    = None
voice:           Optional[VoiceHandler]        = None
display_manager: Optional[PepperDisplayManager] = None

movement_keys = {k: False for k in ('w', 's', 'a', 'd', 'q', 'e')}
PTT_KEY = config.PTT_KEY


# ── Pepper guard helper ───────────────────────────────────────────────────────

def _pepper_ok() -> bool:
    return pepper is not None and pepper.connected


def _retry(fn, *args, attempts: int = 2, delay: float = 0.5, **kwargs):
    """
    Call fn(*args, **kwargs) up to `attempts` times.

    On transient failures (network hiccup, Groq 500) the pipeline retries
    once after `delay` seconds before falling through to the error handler.
    Two attempts is intentional — if it fails twice the error message to
    the user is appropriate and we don't want silent retry loops.

    Only used for LLM calls (brain.chat / brain.chat_with_context). TTS
    has its own 3-tier fallback chain. SSH has its own reconnect logic.
    """
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < attempts - 1:
                logging.warning("LLM attempt %d/%d failed: %s — retrying in %.1fs",
                                attempt + 1, attempts, e, delay)
                time.sleep(delay)
    raise last_exc


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
                logging.error("Gesture '%s' error: %s", name, e)
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
    """
    Entry point for all incoming messages (text or voice).

    If the pipeline is free, processing starts immediately on the calling
    thread. If busy (Pepper is thinking or speaking), the message is queued
    instead of dropped. The queue is bounded at 3 — if it's full, the oldest
    message is evicted to make room so the most recent message always lands.
    """
    if not state.message_lock.acquire(blocking=False):
        # Pipeline is busy — queue instead of dropping.
        if state.message_queue.full():
            # Evict the oldest to make room for the newest.
            try:
                dropped = state.message_queue.get_nowait()
                logging.warning("Queue full — dropped oldest message: '%s'", dropped[:60])
            except queue.Empty:
                pass
        try:
            state.message_queue.put_nowait(message)
            n = state.message_queue.qsize()
            if gui:
                gui.update_status(f"⏳ Queued — {n} waiting")
        except queue.Full:
            # Shouldn't happen since we just made room, but be safe.
            if gui:
                gui.add_system_message("⚠️ Queue full — message dropped")
        return

    # Lock acquired — process immediately on this thread.
    _process_message(message)


def _process_message(message: str):
    """
    Run one message through the full LLM → gesture → TTS pipeline.

    Must always be called with state.message_lock already held.
    Releases the lock in its finally block, then calls _drain_message_queue
    so any queued messages are picked up without needing a separate thread.
    """
    _reset_status = True

    try:
        if not state.robot_active:
            _reset_status = False
            if gui:
                gui.update_status("Pepper is idle — press SPACE to activate")
                gui.add_pepper_message("I'm currently idle. Press SPACE to wake me up!")
            return

        # Goodbye shortcut — checked before starting thinking indicator.
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

        response_text  = None
        function_calls = None

        if gui:
            gui.update_status("Thinking…")

        # thinking() context manager guarantees eyes stop pulsing even if
        # an exception fires mid-LLM. nullcontext used in offline mode so
        # the logic path is identical either way.
        _think_ctx = pepper.thinking() if _pepper_ok() else nullcontext()

        with _think_ctx:
            if config.USE_WEB_SEARCH and brain.needs_search(message):
                if gui:
                    gui.update_status("🔍 Searching web…")
                search_results = web_searcher.search(message)
                response_text, function_calls = _retry(
                    brain.chat_with_context,
                    user_message=message,
                    context=search_results,
                )
            else:
                response_text, function_calls = _retry(brain.chat, message)

                # Only run the search + follow-up LLM call if the model
                # returned no text — i.e. it called web_search instead of
                # answering directly. If it already gave us a response,
                # trust that and skip the second call entirely.
                search_results = execute_search(function_calls) if not response_text else None

                if search_results:
                    if gui:
                        gui.update_status("🔍 Processing search results…")
                    response_text, function_calls = _retry(
                        brain.chat_with_context,
                        user_message=message,
                        context=search_results,
                    )

        # Thinking done. Gestures and speech happen outside the context
        # manager — eyes stop pulsing before Pepper starts speaking.
        emotion = execute_gestures(function_calls)

        # Set eye colour to match the emotion before speaking so the colour
        # is visible while Pepper talks. Eyes revert to blue automatically
        # after speech ends (handled in _hq_speech_animation_loop).
        if emotion and _pepper_ok():
            eye_color = pepper.EMOTION_COLOUR_MAP.get(emotion, "blue")
            pepper.set_eye_color(eye_color)

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
        logging.error("Message handling error: %s", e, exc_info=True)
        if gui:
            gui.update_status("Error — Ready")
            gui.add_pepper_message("Sorry, I encountered an error.")
    finally:
        state.message_lock.release()
        if gui and _reset_status:
            n = state.message_queue.qsize()
            gui.update_status("Ready" if n == 0 else f"Ready — {n} queued")
        # Always drain — runs on normal exit, early returns, AND exceptions.
        # Lock is released above before this line, so _drain_message_queue
        # can re-acquire it safely.
        _drain_message_queue()


def _drain_message_queue():
    """
    Process the next queued message if one is waiting.

    Uses a check-then-acquire pattern:
      1. Bail early if the queue is empty (fast path, no lock needed).
      2. Try to acquire the lock non-blocking.
      3. Only pop from the queue AFTER holding the lock.

    This guarantees a message is never popped and then lost because the
    lock acquire failed. If another thread grabbed the lock first (e.g. a
    new message arrived at exactly this moment), that thread will drain the
    queue at the end of its own run.
    """
    if state.message_queue.empty():
        return

    if not state.message_lock.acquire(blocking=False):
        # Another active processing thread will drain the queue itself.
        return

    # We hold the lock — now safely pop.
    try:
        next_msg = state.message_queue.get_nowait()
    except queue.Empty:
        state.message_lock.release()
        return

    # Process — lock is held, same contract as always.
    _process_message(next_msg)


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

        # Movement keys — update timestamp on PRESS only.
        # The watchdog in movement_controller is specifically for the case
        # where a key-up event is dropped (network/focus issue). Updating
        # on release would defeat the watchdog by resetting it after every
        # normal key-up.
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
            # Clear the key state only — do NOT update last_movement_key_time.
            # The movement_controller loop feeds a heartbeat to the watchdog
            # while keys are held, so it only fires in genuinely stuck situations.
            movement_keys[k] = False

    except AttributeError:
        pass


# ── Movement controller ───────────────────────────────────────────────────────

def movement_controller():
    """
    Sends moveToward() continuously at 20 Hz while any key is held.

    Heartbeat: while movement is active, last_movement_key_time is updated
    every loop tick. This means the watchdog only fires if the movement loop
    itself stops feeding updates — i.e. something has genuinely gone wrong —
    not during normal held-key movement. This eliminates the brief pause/stop
    that occurred when holding a key longer than the old 2s timeout.

    NAOqi's internal watchdog stops movement if no moveToward() arrives within
    ~1s, so we keep feeding it every 50ms loop iteration.
    """
    WATCHDOG_TIMEOUT = 1.0   # seconds — must match documented intent
    SEND_INTERVAL    = 0.05  # 20 Hz — tighter than the old 10 Hz, less stutter
    prev_any         = False

    while state.running:
        try:
            if not _pepper_ok() or not state.robot_active:
                time.sleep(SEND_INTERVAL)
                continue

            any_pressed = any(movement_keys.values())

            # Watchdog — catches a genuinely stuck key state (e.g. OS/focus
            # swallowed the key-up event). Under normal held-key movement this
            # never fires because we refresh the timestamp below every tick.
            if any_pressed and (time.time() - state.last_movement_key_time > WATCHDOG_TIMEOUT):
                print("⚠️  Movement watchdog fired — clearing stuck keys")
                for k in movement_keys:
                    movement_keys[k] = False
                pepper.stop_movement()
                prev_any = False
                time.sleep(SEND_INTERVAL)
                continue

            if any_pressed:
                # Additive axes — allows simultaneous W+A (forward+turn) etc.
                x     =  0.6 if movement_keys['w'] else -0.6 if movement_keys['s'] else 0.0
                theta =  0.5 if movement_keys['a'] else -0.5 if movement_keys['d'] else 0.0
                y     =  0.4 if movement_keys['q'] else -0.4 if movement_keys['e'] else 0.0
                pepper._move(x, y, theta)
                # Feed the watchdog — as long as the loop is running and keys
                # are held this stays fresh, so the watchdog never triggers.
                state.last_movement_key_time = time.time()
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
    global pepper, brain, tts, gui, web_searcher, voice, display_manager

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
    # Queued — processed on first GUI frame (DPG context isn't ready yet here)
    # gui.set_connection_status() is called below after gui is created.

    # Tablet display manager — always created so the GUI panel is available
    # even in offline mode. Tablet calls are guarded inside PepperRobot so
    # they silently no-op if the robot isn't connected or lacks the service.
    display_manager = PepperDisplayManager(pepper_ip=config.PEPPER_IP, port=8765)
    display_manager.set_tablet_fns(
        show_fn  = pepper.show_tablet_image,
        clear_fn = pepper.clear_tablet,
    )
    display_manager.start()
    print("   ✅ Tablet display manager started")

    # HQ audio depends on paramiko to transfer files to Pepper over SSH.
    # Warn loudly here rather than letting it silently fall back to robotic TTS.
    from pepper_interface import _PARAMIKO_AVAILABLE
    if not _PARAMIKO_AVAILABLE:
        print("\n   ⚠️  ══════════════════════════════════════════════════")
        print("   ⚠️  paramiko is NOT installed — HQ audio is DISABLED")
        print("   ⚠️  Pepper will use her robotic built-in voice instead")
        print("   ⚠️  Fix: pip install paramiko --break-system-packages")
        print("   ⚠️  ══════════════════════════════════════════════════\n")

    print("\n3️⃣  Initialising AI brain…")
    brain = GroqBrain(
        api_key        = config.GROQ_API_KEY,
        llm_model      = config.GROQ_LLM_MODEL,
        whisper_model  = config.GROQ_WHISPER_MODEL,
        system_prompt  = config.build_system_prompt(),   # fresh date every startup
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
        volume_callback        = on_volume_changed,
        action_callback        = on_action,
        display_callback       = display_manager.show_image if display_manager else None,
        clear_display_callback = display_manager.clear_display if display_manager else None,
    )

    if _pepper_ok():
        gui.update_status("Idle — press SPACE to activate Pepper")
    else:
        gui.update_status("⚠️ Pepper offline — chat only mode")

    # Connection dot — cyan if connected, grey if offline.
    # Queued here, rendered on first frame.
    gui.set_connection_status(pepper.connected if pepper else False)

    try:
        gui.start()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        state.running = False
        kb_listener.stop()
        if gui:
            gui.set_connection_status(False)
            gui.stop()
        if display_manager:
            display_manager.stop()
        if pepper:
            if pepper.connected:
                pepper.set_volume(40)
            pepper.disconnect()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()