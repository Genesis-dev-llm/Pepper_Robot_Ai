"""
Main Control Script — Pepper AI Robot
"""

import logging
import queue
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Callable, List, Optional

from pynput import keyboard

import config
from groq_brain import GroqBrain, test_groq_connection
from hybrid_tts_handler import HybridTTSHandler
from pepper_display import PepperDisplayManager
from pepper_gui import PepperDearPyGUI
from pepper_interface import PepperRobot
from voice_handler import VoiceHandler, list_microphones
from wake_word_handler import WakeWordHandler, _PORCUPINE_AVAILABLE
from web_search_handler import WebSearchHandler


# ── Logging setup ──────────────────────────────────────────────────────────────

def _setup_logging():
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s [%(levelname)s] %(message)s",
        datefmt  = "%H:%M:%S",
        handlers = [logging.StreamHandler()],
    )


# ── Shared state ───────────────────────────────────────────────────────────────

state = SimpleNamespace(
    robot_active           = False,
    running                = True,
    ptt_active             = False,
    last_movement_key_time = 0.0,
    message_lock           = threading.Lock(),
    message_queue          = queue.Queue(maxsize=config.MSG_QUEUE_SIZE),
    ptt_lock               = threading.Lock(),
)

# ── Component handles ──────────────────────────────────────────────────────────
pepper:            Optional[PepperRobot]          = None
gui:               Optional[PepperDearPyGUI]      = None
brain:             Optional[GroqBrain]            = None
tts:               Optional[HybridTTSHandler]     = None
web_searcher:      Optional[WebSearchHandler]     = None
voice:             Optional[VoiceHandler]         = None
display_manager:   Optional[PepperDisplayManager] = None
wake_word_handler: Optional[WakeWordHandler]      = None

# ── Movement keys — snapshot approach for thread safety ───────────────────────
_movement_keys      = {k: False for k in ('w', 's', 'a', 'd', 'q', 'e')}
_movement_keys_lock = threading.Lock()

PTT_KEY = config.PTT_KEY


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pepper_ok() -> bool:
    return pepper is not None and pepper.connected


def _retry(fn, *args, attempts: int = 2, delay: float = 0.5, **kwargs):
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < attempts - 1:
                logging.warning("Attempt %d/%d failed: %s — retrying", attempt + 1, attempts, e)
                time.sleep(delay)
    raise last_exc


# ── Gesture parsing ────────────────────────────────────────────────────────────

def _parse_function_calls(function_calls: Optional[List]) -> tuple:
    """
    Split function calls into:
      - gesture_callback: closure that executes all gesture calls (or None)
      - emotion: string from express_emotion (or None)
    Returns (gesture_callback, emotion).
    """
    if not function_calls:
        return None, None

    gesture_calls = [
        fn for fn in function_calls
        if fn.get("name") in config.GESTURE_NAMES
    ]
    emotion_calls = [
        fn for fn in function_calls
        if fn.get("name") == "express_emotion"
    ]

    emotion = emotion_calls[0]["arguments"].get("emotion") if emotion_calls else None

    gesture_map = {
        "wave":               lambda: pepper.wave(),
        "nod":                lambda: pepper.nod(),
        "shake_head":         lambda: pepper.shake_head(),
        "thinking_gesture":   lambda: pepper.thinking_gesture(),
        "explaining_gesture": lambda: pepper.explaining_gesture(),
        "excited_gesture":    lambda: pepper.excited_gesture(),
        "point_forward":      lambda: pepper.point_forward(),
        "shrug":              lambda: pepper.shrug(),
        "celebrate":          lambda: pepper.celebrate(),
        "look_around":        lambda: pepper.look_around(),
        "bow":                lambda: pepper.bow(),
        "look_at_sound":      lambda: pepper.look_at_sound(),
    }

    if not gesture_calls:
        return None, emotion

    def _gesture_callback():
        if not _pepper_ok():
            return
        for fn in gesture_calls:
            name = fn.get("name", "")
            action = gesture_map.get(name)
            if action:
                try:
                    action()
                except Exception as e:
                    logging.error("Gesture '%s' error: %s", name, e)

    return _gesture_callback, emotion


def _extract_search_call(function_calls: Optional[List]) -> Optional[str]:
    """Return the query from a web_search function call, or None."""
    if not function_calls or not config.USE_WEB_SEARCH:
        return None
    for fn in function_calls:
        if fn.get("name") == "web_search":
            q = fn.get("arguments", {}).get("query", "").strip()
            if q:
                return q
    return None


# ── Callbacks ──────────────────────────────────────────────────────────────────

def on_action(action: str):
    if action == "pulse_eyes":
        if _pepper_ok():
            threading.Thread(
                target=pepper.pulse_eyes,
                args=("blue", 2.0),
                daemon=True,
                name="PulseEyes",
            ).start()
    elif action == "clear_conversation":
        if brain:
            brain.reset_conversation()
        if gui:
            gui.add_system_message("🔄 Conversation cleared")
    elif action == "reconnect":
        threading.Thread(target=_attempt_reconnect, daemon=True, name="Reconnect").start()


def on_volume_changed(volume: int):
    if _pepper_ok():
        pepper.set_volume(volume)


def on_tts_tier(tier_label: str):
    """Called by HybridTTSHandler when a tier is selected."""
    if gui:
        gui.update_tts_tier(tier_label)


# ── Reconnection ───────────────────────────────────────────────────────────────

def _attempt_reconnect():
    global pepper
    if gui:
        gui.update_status("🔄 Reconnecting to Pepper…")
        gui.set_connection_status(False)
    print("🔄 Attempting NAOqi reconnect…")
    if pepper:
        try:
            pepper.disconnect()
        except Exception:
            pass
    pepper = PepperRobot(
        config.PEPPER_IP, config.PEPPER_PORT,
        ssh_user=config.PEPPER_SSH_USER,
        ssh_password=config.PEPPER_SSH_PASS,
    )
    success = pepper.connect()
    if gui:
        gui.set_connection_status(success)
        gui.update_status("Reconnected ✅" if success else "Reconnect failed ❌")
    if success:
        print("✅ Reconnected to Pepper")
    else:
        print("❌ Reconnect failed — still in offline mode")


# ── Wake word callback ─────────────────────────────────────────────────────────

def _on_wake_word():
    """
    Fired by WakeWordHandler from its background thread when the wake word
    is detected.  Must be thread-safe — all GUI calls go through message_queue.

    Flow:
      1. If PTT recording is already active, do nothing (lock would fail anyway).
      2. Activate Pepper if currently idle.
      3. Acquire ptt_lock (non-blocking) — abort if PTT is already in progress.
      4. Start a recording window via voice.start_recording().
      5. Spawn WakeAutoStop thread to stop recording after WAKE_WORD_LISTEN_SECONDS.
         The auto-stop thread mirrors on_release() in the PTT path: it calls
         stop_recording_and_transcribe() which fires the on_transcribed callback
         asynchronously, which in turn spawns a thread → handle_message().
    """
    # Fast-path: bail out if a recording is already underway
    if state.ptt_active:
        return

    # Activate Pepper if currently idle
    if not state.robot_active:
        state.robot_active = True
        if _pepper_ok():
            pepper.set_eye_color("blue")
        if gui:
            gui.set_robot_active(True)

    if gui:
        gui.update_status("👂 Wake word detected — listening…")

    # Try to acquire the PTT lock — if PTT is already in progress, give up
    if not state.ptt_lock.acquire(blocking=False):
        return

    state.ptt_active = True
    if gui:
        gui.set_recording(True)

    # Guard: voice may be None if VOICE_ENABLED is False
    if not voice:
        state.ptt_active = False
        state.ptt_lock.release()
        if gui:
            gui.set_recording(False)
        return

    started = voice.start_recording()
    if not started:
        state.ptt_active = False
        state.ptt_lock.release()
        if gui:
            gui.set_recording(False)
        return

    # Auto-stop thread: releases lock after stopping, mirroring on_release()
    def _auto_stop():
        time.sleep(config.WAKE_WORD_LISTEN_SECONDS)
        if state.ptt_active:
            state.ptt_active = False
            if voice:
                voice.stop_recording_and_transcribe()
            if gui:
                gui.set_recording(False)
            try:
                state.ptt_lock.release()
            except RuntimeError:
                # Lock already released (shouldn't happen, but be defensive)
                pass

    threading.Thread(target=_auto_stop, daemon=True, name="WakeAutoStop").start()


# ── Message handler ────────────────────────────────────────────────────────────

def handle_message(message: str):
    """
    Entry point for all messages (text or voice).
    If pipeline is busy, queues the message. Drops oldest if queue is full.
    """
    if not state.message_lock.acquire(blocking=False):
        if state.message_queue.full():
            try:
                dropped = state.message_queue.get_nowait()
                logging.warning("Queue full — dropped: '%s'", dropped[:60])
                if gui:
                    gui.add_system_message("⚠️ Queue full — dropped older message")
            except queue.Empty:
                pass
        try:
            state.message_queue.put_nowait(message)
            n = state.message_queue.qsize()
            if gui:
                gui.update_status(f"⏳ Queued — {n} waiting")
        except queue.Full:
            if gui:
                gui.add_system_message("⚠️ Queue full — message dropped")
        return

    _process_message(message)


def _process_message(message: str):
    """
    Run one message through the LLM → gesture → TTS pipeline.
    Lock is already held on entry; released in finally.
    """
    _reset_status = True
    try:
        if not state.robot_active:
            _reset_status = False
            if gui:
                gui.update_status("Pepper is idle — press SPACE to activate")
                gui.add_pepper_message("I'm idle right now. Press SPACE to wake me up!")
            return

        if config.GOODBYE_WORD.lower() in message.lower():
            _reset_status = False
            _say("Goodbye! It was nice talking with you.")
            state.robot_active = False
            if _pepper_ok():
                pepper.wave()
                pepper.set_eye_color("white")
            if gui:
                gui.update_status("Idle")
                gui.set_robot_active(False)
            return

        if gui:
            gui.update_status("Thinking…")

        response_text  = None
        function_calls = None

        _think_ctx = pepper.thinking() if _pepper_ok() else nullcontext()

        with _think_ctx:
            # Keyword fast-path: search before LLM if obvious
            if config.USE_WEB_SEARCH and brain.needs_search(message):
                if gui:
                    gui.update_status("🔍 Searching…")
                search_results = web_searcher.search(message)
                response_text, function_calls = _retry(
                    brain.chat_with_context,
                    user_message=message,
                    context=search_results,
                )
            else:
                response_text, function_calls = _retry(brain.chat, message)

        # Check if the model requested a web search as a function call
        search_query = _extract_search_call(function_calls)
        if search_query:
            if gui:
                gui.update_status("🔍 Searching…")
            print(f"🔍 Model requested search: '{search_query}'")
            search_results = web_searcher.search(search_query)
            response_text, function_calls = _retry(
                brain.chat_with_context,
                user_message=message,
                context=search_results,
            )

        gesture_callback, emotion = _parse_function_calls(function_calls)

        if response_text:
            if gui:
                gui.add_pepper_message(response_text)
            _say(response_text, emotion=emotion, gesture_callback=gesture_callback)
        else:
            fallback = "Sorry, I didn't catch that."
            if gui:
                gui.add_pepper_message(fallback)
            _say(fallback)

    except Exception as e:
        logging.error("Message handling error: %s", e, exc_info=True)
        if gui:
            gui.update_status("Error — Ready")
            gui.add_pepper_message("Sorry, I hit an error.")
    finally:
        state.message_lock.release()
        if gui and _reset_status:
            n = state.message_queue.qsize()
            gui.update_status("Ready" if n == 0 else f"Ready — {n} queued")
        _drain_queue()


def _drain_queue():
    if state.message_queue.empty():
        return
    if not state.message_lock.acquire(blocking=False):
        return
    try:
        next_msg = state.message_queue.get_nowait()
    except queue.Empty:
        state.message_lock.release()
        return
    _process_message(next_msg)


def _say(
    text: str,
    emotion:          Optional[str]      = None,
    gesture_callback: Optional[Callable] = None,
):
    try:
        if _pepper_ok():
            def _status_cb(msg: str):
                if gui and gui.is_running:
                    gui.update_status(msg)
            pepper.speak_hq(
                text,
                tts,
                emotion          = emotion,
                status_callback  = _status_cb,
                gesture_callback = gesture_callback,
            )
        else:
            if gui and gui.is_running:
                gui.update_status("🎙️ Generating voice…")
            if tts:
                if gesture_callback:
                    try:
                        gesture_callback()
                    except Exception:
                        pass
                tts.speak_and_play(text, emotion=emotion)
    finally:
        if gui and gui.is_running:
            gui.update_status("Ready")


# ── Camera stream helpers ─────────────────────────────────────────────────────

def _start_camera_stream():
    """
    Blocking — runs in a daemon thread spawned by main.
    Uploads camera_stream.py, starts the server on Pepper, waits for it to
    bind, then points the tablet at the stream page.
    Calls gui.update_camera_status() with the result when done.
    """
    if not _pepper_ok():
        if gui:
            gui.add_system_message("⚠️ Pepper not connected — can't start camera stream")
            gui.update_camera_status(False)
        return
    success = pepper.start_tablet_camera_stream()
    if gui:
        gui.update_camera_status(success)
        if not success:
            gui.add_system_message(
                "❌ Camera stream failed — check /tmp/camera_stream.log on Pepper"
            )


def _stop_camera_stream():
    """Instant — kills server process and clears tablet."""
    if _pepper_ok():
        pepper.stop_tablet_camera_stream()
    if gui:
        gui.update_camera_status(False)


# ── Keyboard handlers ──────────────────────────────────────────────────────────

def on_press(key):
    try:
        if gui and gui.text_input_focused:
            return

        k = key.char if hasattr(key, "char") and key.char else None

        if key == keyboard.Key.esc:
            print("\n👋 Shutting down…")
            state.running = False
            if gui:
                gui.is_running = False
            return

        if key == keyboard.Key.space:
            state.robot_active = not state.robot_active
            label = "ACTIVE 🟢" if state.robot_active else "IDLE 🔴"
            print(f"\n{'='*50}\nPepper is now {label}\n{'='*50}\n")
            if _pepper_ok():
                pepper.set_eye_color("blue" if state.robot_active else "white")
            if gui:
                gui.set_robot_active(state.robot_active)
                gui.update_status("Active — ready" if state.robot_active else "Idle")
            return

        if k is None:
            return

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
            else:
                # voice is None — release lock immediately so PTT isn't stuck
                state.ptt_active = False
                state.ptt_lock.release()
            return

        with _movement_keys_lock:
            if k in _movement_keys:
                _movement_keys[k]              = True
                state.last_movement_key_time   = time.time()
                return

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

        with _movement_keys_lock:
            if k in _movement_keys:
                _movement_keys[k] = False

    except AttributeError:
        pass


# ── Movement controller ────────────────────────────────────────────────────────

def movement_controller():
    """
    20Hz movement loop. Takes a brief lock snapshot at the top of each tick
    (~5µs) to safely read movement key state across threads.
    """
    WATCHDOG_TIMEOUT = 1.0
    SEND_INTERVAL    = 0.05
    prev_any         = False

    while state.running:
        try:
            if not _pepper_ok() or not state.robot_active:
                time.sleep(SEND_INTERVAL)
                continue

            with _movement_keys_lock:
                keys = dict(_movement_keys)

            any_pressed = any(keys.values())

            if any_pressed and (time.time() - state.last_movement_key_time > WATCHDOG_TIMEOUT):
                print("⚠️  Movement watchdog — clearing stuck keys")
                with _movement_keys_lock:
                    for k in _movement_keys:
                        _movement_keys[k] = False
                pepper.stop_movement()
                prev_any = False
                time.sleep(SEND_INTERVAL)
                continue

            if any_pressed:
                x     =  config.MOVE_SPEED_FWD    if keys['w'] else -config.MOVE_SPEED_FWD    if keys['s'] else 0.0
                theta =  config.MOVE_SPEED_TURN   if keys['a'] else -config.MOVE_SPEED_TURN   if keys['d'] else 0.0
                y     =  config.MOVE_SPEED_STRAFE if keys['q'] else -config.MOVE_SPEED_STRAFE if keys['e'] else 0.0
                pepper._move(x, y, theta)
                state.last_movement_key_time = time.time()
            elif prev_any:
                pepper.stop_movement()

            prev_any = any_pressed
            time.sleep(SEND_INTERVAL)

        except Exception as ex:
            print(f"❌ Movement controller: {ex}")
            time.sleep(0.5)


# ── Controls summary ───────────────────────────────────────────────────────────

def print_controls():
    ptt = PTT_KEY.upper()
    print("\n" + "="*60)
    print("🎮 PEPPER ROBOT CONTROLS")
    print("="*60)
    print(f"\n🎙️ VOICE:  Hold {ptt} → speak → release → auto-transcribes")
    if config.WAKE_WORD_ENABLED and wake_word_handler is not None:
        print(f"👂 WAKE:   Say '{config.WAKE_WORD}' → Pepper activates + listens")
    print("\n💬 TEXT:   Click GUI input → type → Enter or Send")
    print("\n🤖 MOVEMENT (input box NOT focused):")
    print("  W/S=Forward/Back  A/D=Turn  Q/E=Strafe")
    print(f"  Speeds: fwd={config.MOVE_SPEED_FWD} turn={config.MOVE_SPEED_TURN} strafe={config.MOVE_SPEED_STRAFE}")
    print("\n✋ GESTURES:  1=Wave  2=Nod  3=Shake  4=Think  8=Explain  9=Excited  0=Point")
    print("💡 LEDs:     5=Blue  6=Green  7=Red")
    print("\n⚙️ SYSTEM:  SPACE=Wake/Sleep  ESC=Quit")
    print("="*60 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    global pepper, brain, tts, gui, web_searcher, voice, display_manager, wake_word_handler

    _setup_logging()

    print("\n🤖 PEPPER AI ROBOT")
    print("=" * 60)

    print("\n1️⃣  Testing Groq API…")
    if not test_groq_connection(config.GROQ_API_KEY):
        print("❌ Groq API test failed. Check your API key in .env")
        return

    print("\n2️⃣  Connecting to Pepper…")
    pepper = PepperRobot(
        config.PEPPER_IP, config.PEPPER_PORT,
        ssh_user=config.PEPPER_SSH_USER,
        ssh_password=config.PEPPER_SSH_PASS,
    )
    pepper.connect()

    display_manager = PepperDisplayManager(pepper_ip=config.PEPPER_IP, port=8765)
    display_manager.set_tablet_fns(
        show_fn    = pepper.show_tablet_image,
        webview_fn = pepper.show_tablet_webview,
        clear_fn   = pepper.clear_tablet,
    )
    display_manager.start()
    print("   ✅ Tablet display manager started")

    from pepper_interface import _PARAMIKO_AVAILABLE
    if not _PARAMIKO_AVAILABLE:
        print("\n   ⚠️  paramiko not installed — HQ audio DISABLED")
        print("   Fix: pip install paramiko --break-system-packages\n")

    print("\n3️⃣  Initialising AI brain…")
    brain = GroqBrain(
        api_key        = config.GROQ_API_KEY,
        llm_model      = config.GROQ_LLM_MODEL,
        whisper_model  = config.GROQ_WHISPER_MODEL,
        system_prompt  = config.build_system_prompt(),
        functions      = config.ROBOT_FUNCTIONS,
        use_web_search = config.USE_WEB_SEARCH,
    )

    print("\n4️⃣  Initialising TTS…")
    tts = HybridTTSHandler(
        groq_api_key       = config.GROQ_API_KEY,
        groq_voice         = config.GROQ_VOICE,
        elevenlabs_api_key = config.ELEVENLABS_API_KEY,
        edge_voice         = config.TTS_VOICE,
        edge_rate          = config.TTS_RATE,
        tier_callback      = on_tts_tier,
    )

    print("\n5️⃣  Initialising web search…")
    web_searcher = WebSearchHandler(max_results=3, timeout=8.0)
    print(f"   ✅ DuckDuckGo ready — {'enabled' if config.USE_WEB_SEARCH else 'disabled'}")

    print("\n6️⃣  Initialising voice (STT)…")
    if config.VOICE_ENABLED:
        try:
            VoiceHandler.validate_setup()
        except RuntimeError as e:
            print(f"   ❌ Voice pre-check failed: {e}")
            config.VOICE_ENABLED = False

    if config.VOICE_ENABLED:
        list_microphones()
        voice = VoiceHandler(
            transcribe_fn = brain.transcribe_audio,
            sample_rate   = config.AUDIO_SAMPLE_RATE,
            channels      = config.AUDIO_CHANNELS,
            min_duration  = config.AUDIO_MIN_DURATION,
            max_duration  = config.AUDIO_MAX_DURATION,
        )

        def _on_recording_start():
            if gui:
                gui.set_recording(True)

        def _on_recording_stop():
            if gui:
                gui.set_recording(False)

        def _on_transcribing():
            if gui:
                gui.update_status("🔄 Transcribing…")

        def _on_transcribed(text: str):
            print(f"📝 Transcribed: \"{text}\"")
            if gui:
                gui.add_chat_message(text, source="voice")
            threading.Thread(
                target=handle_message,
                args=(text,),
                daemon=True,
                name="VoiceMessageHandler",
            ).start()

        def _on_error(msg: str):
            print(f"🎙️ Voice error: {msg}")
            if gui:
                gui.set_recording(False)
                gui.update_status(f"Voice: {msg}")

        def _on_audio_level(level: float):
            if gui:
                gui.update_audio_level(level)

        voice.on_recording_start = _on_recording_start
        voice.on_recording_stop  = _on_recording_stop
        voice.on_transcribing    = _on_transcribing
        voice.on_transcribed     = _on_transcribed
        voice.on_error           = _on_error
        voice.on_audio_level     = _on_audio_level

        print(f"   ✅ Push-to-talk ready (hold '{PTT_KEY.upper()}' to speak)")
    else:
        print("   ⚠️  Voice disabled")

    # ── Step 7: Wake word ──────────────────────────────────────────────────────
    print("\n7️⃣  Initialising wake word…")
    if config.WAKE_WORD_ENABLED and _PORCUPINE_AVAILABLE:
        if not config.PICOVOICE_ACCESS_KEY:
            print("   ⚠️  PICOVOICE_ACCESS_KEY not set — wake word disabled")
            print("   Get a free key at https://console.picovoice.ai/")
            print("   Then add to .env:  export PICOVOICE_ACCESS_KEY=\"your-key-here\"")
        else:
            try:
                wake_word_handler = WakeWordHandler(
                    keyword     = config.WAKE_WORD,
                    access_key  = config.PICOVOICE_ACCESS_KEY,
                    sensitivity = config.WAKE_WORD_SENSITIVITY,
                    on_wake     = _on_wake_word,
                )
                wake_word_handler.start()
                print(f"   ✅ Wake word ready — say '{config.WAKE_WORD}' to activate")
            except Exception as e:
                print(f"   ❌ Wake word init failed: {e}")
                wake_word_handler = None
    elif not _PORCUPINE_AVAILABLE:
        print("   ⚠️  pvporcupine not installed — wake word disabled")
        print("   Install: pip install pvporcupine pvrecorder")
    else:
        print("   ⚠️  Wake word disabled in config (WAKE_WORD_ENABLED = False)")

    print("\n✅ All systems ready!")

    # ── Start background threads ───────────────────────────────────────────────
    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    kb_listener.start()

    threading.Thread(target=movement_controller, daemon=True, name="MovementController").start()

    print_controls()

    # ── GUI (blocks until window is closed) ───────────────────────────────────
    gui = PepperDearPyGUI(
        message_callback       = lambda msg: threading.Thread(
            target=handle_message, args=(msg,), daemon=True
        ).start(),
        volume_callback        = on_volume_changed,
        action_callback        = on_action,
        display_callback       = display_manager.show_image if display_manager else None,
        clear_display_callback = display_manager.clear_display if display_manager else None,
        webview_callback       = lambda url: pepper.show_tablet_webview(url) if _pepper_ok() else None,
        start_camera_callback  = lambda: threading.Thread(
            target=_start_camera_stream, daemon=True, name="CameraStart"
        ).start(),
        stop_camera_callback   = _stop_camera_stream,
    )

    if _pepper_ok():
        gui.update_status("Idle — press SPACE to activate Pepper")
    else:
        gui.update_status("⚠️ Pepper offline — chat only mode")

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
        # Stop wake word listener before SSH is closed
        if wake_word_handler:
            wake_word_handler.stop()
        if display_manager:
            display_manager.stop()
        if pepper:
            # Stop camera stream BEFORE disconnect — disconnect() closes the
            # SSH client, so the pkill command must go out while it's still open.
            pepper.stop_tablet_camera_stream()
            if pepper.connected:
                pepper.set_volume(40)
            pepper.disconnect()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()