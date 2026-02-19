"""
DearPyGUI Interface for Pepper AI Chat
"""

import queue
import threading
import time

import dearpygui.dearpygui as dpg


MAX_CHAT_MESSAGES = 60


class PepperDearPyGUI:
    def __init__(self, message_callback, volume_callback=None):
        self.message_callback  = message_callback
        self.volume_callback   = volume_callback   # fn(int 0–100) — wired to pepper.set_volume
        self.is_running        = False
        self.message_queue     = queue.Queue()
        self.status_queue      = queue.Queue()

        # Written only from DPG callbacks (main thread) — read from pynput thread.
        # No lock needed: bool assignment is atomic in CPython.
        self.text_input_focused = False

        self._msg_tag_counter      = 0
        self._msg_tags: list       = []
        self._scroll_pending: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self.is_running = True
        dpg.create_context()
        self._setup_window()
        dpg.create_viewport(
            title      = "🤖 Pepper AI Control Dashboard",
            width      = 900,
            height     = 780,
            min_width  = 600,
            min_height = 480,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

        while dpg.is_dearpygui_running() and self.is_running:
            self._process_queues()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()
        self.is_running = False

    def stop(self):
        self.is_running = False
        try:
            dpg.stop_dearpygui()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Window layout
    # ------------------------------------------------------------------

    def _setup_window(self):
        with dpg.window(label="Pepper AI Chat", tag="main_window",
                        no_close=True, no_collapse=True):

            with dpg.group(horizontal=True):
                dpg.add_text("🤖 Pepper AI Dashboard",
                             tag="header_text", color=(100, 149, 237))
                dpg.add_spacer(width=20)
                dpg.add_text("Status: Starting…",
                             tag="status_text", color=(150, 150, 150))

            dpg.add_separator()

            with dpg.group(tag="recording_indicator", show=False):
                dpg.add_text("🔴  RECORDING — Release R to send",
                             tag="recording_label", color=(255, 80, 80))
                dpg.add_separator()

            with dpg.collapsing_header(label="💡 Controls & Instructions",
                                       default_open=True):
                dpg.add_text("Text mode:  click the input box → type → Enter / Send")
                dpg.add_text("           (click outside or send to return to robot controls)")
                dpg.add_text("Voice mode: hold R → speak → release R")
                dpg.add_spacer(height=4)
                dpg.add_text("Robot controls (when input box is NOT focused):")
                dpg.add_text("  SPACE=Wake/Sleep  |  WASD=Move  |  1-9=Gestures  |  5-7=LED")

            dpg.add_separator()

            dpg.add_text("Chat History:", color=(200, 200, 200))
            dpg.add_child_window(tag="chat_window", height=350, border=True)

            dpg.add_separator()

            dpg.add_text("Your Message:", color=(200, 200, 200))
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    tag      = "message_input",
                    hint     = "Click here to type… or hold R to speak",
                    width    = -100,
                    on_enter = True,
                    callback = self._send_text_message,
                )
                dpg.add_button(label="Send", width=90,
                               callback=self._send_text_message)

            # Cursor-based keyboard gate
            with dpg.item_handler_registry(tag="input_focus_handler"):
                dpg.add_item_activated_handler(callback=self._on_input_activated)
                dpg.add_item_deactivated_handler(callback=self._on_input_deactivated)
            dpg.bind_item_handler_registry("message_input", "input_focus_handler")

            dpg.add_separator()

            # ── Volume slider ─────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("🔊 Volume:", color=(200, 200, 200))
                dpg.add_slider_int(
                    tag            = "volume_slider",
                    default_value  = 100,
                    min_value      = 0,
                    max_value      = 100,
                    width          = 300,
                    callback       = self._on_volume_changed,
                    format         = "%d%%",
                )

            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_text("Terminal:", color=(150, 150, 150))
                dpg.add_text(" SPACE=Wake", color=(100, 200, 100))
                dpg.add_text(" R=Voice",    color=(255, 150, 50))
                dpg.add_text(" WASD=Move",  color=(100, 200, 100))
                dpg.add_text(" 1-9=Gesture", color=(100, 200, 100))

        dpg.set_primary_window("main_window", True)

        self._add_system_message("🤖 Pepper AI Control Dashboard started")
        self._add_system_message("Click the input box below to type, or hold R to speak")
        self._add_system_message("Press SPACE (outside input box) to wake Pepper")

    # ------------------------------------------------------------------
    # Keyboard gate callbacks
    # ------------------------------------------------------------------

    def _on_input_activated(self):
        self.text_input_focused = True

    def _on_input_deactivated(self):
        self.text_input_focused = False

    # ------------------------------------------------------------------
    # Volume callback
    # ------------------------------------------------------------------

    def _on_volume_changed(self, sender, app_data):
        if self.volume_callback:
            self.volume_callback(int(app_data))

    # ------------------------------------------------------------------
    # Send callback
    # ------------------------------------------------------------------

    def _send_text_message(self, sender=None, app_data=None):
        message = dpg.get_value("message_input").strip()
        if not message:
            return
        dpg.set_value("message_input", "")
        dpg.focus_item("main_window")
        self._add_user_message(message, voice=False)
        threading.Thread(
            target=self.message_callback, args=(message,), daemon=True
        ).start()

    # ------------------------------------------------------------------
    # Thread-safe public methods
    # ------------------------------------------------------------------

    def add_pepper_message(self, message: str):
        self.message_queue.put(("pepper", message))

    def add_system_message(self, message: str):
        self.message_queue.put(("system", message))

    def add_voice_user_message(self, text: str):
        self.message_queue.put(("user_voice", text))

    def update_status(self, status: str):
        self.status_queue.put(status)

    def set_recording(self, recording: bool):
        self.message_queue.put(("recording_state", recording))

    # ------------------------------------------------------------------
    # Main-thread renderers
    # ------------------------------------------------------------------

    def _next_tag(self) -> int:
        self._msg_tag_counter += 1
        return self._msg_tag_counter

    def _register_message_tag(self, tag: int):
        self._msg_tags.append(tag)
        if len(self._msg_tags) > MAX_CHAT_MESSAGES:
            old_tag = self._msg_tags.pop(0)
            try:
                dpg.delete_item(old_tag)
            except Exception:
                pass
        self._scroll_pending = True

    def _add_user_message(self, message: str, voice: bool = False):
        tag    = self._next_tag()
        prefix = "🎙️ You:" if voice else "You:"
        color  = (255, 180, 50) if voice else (66, 135, 245)
        with dpg.group(tag=tag, parent="chat_window"):
            with dpg.group(horizontal=True):
                dpg.add_text(prefix, color=color)
                dpg.add_text(message, wrap=620)
        self._register_message_tag(tag)

    def _add_pepper_message_internal(self, message: str):
        tag = self._next_tag()
        with dpg.group(tag=tag, parent="chat_window"):
            with dpg.group(horizontal=True):
                dpg.add_text("Pepper:", color=(76, 175, 80))
                dpg.add_text(message, wrap=620)
        self._register_message_tag(tag)

    def _add_system_message(self, message: str):
        tag = self._next_tag()
        with dpg.group(tag=tag, parent="chat_window"):
            dpg.add_text(f"• {message}", color=(150, 150, 150))
        self._register_message_tag(tag)

    def _set_recording_internal(self, recording: bool):
        dpg.configure_item("recording_indicator", show=recording)
        if recording:
            dpg.set_value("status_text", "Status: 🔴 RECORDING — release R when done")

    # ------------------------------------------------------------------
    # Frame-loop queue drain
    # ------------------------------------------------------------------

    def _process_queues(self):
        if self._scroll_pending:
            dpg.set_y_scroll("chat_window", dpg.get_y_scroll_max("chat_window"))
            self._scroll_pending = False

        while not self.message_queue.empty():
            try:
                kind, data = self.message_queue.get_nowait()
                if kind == "pepper":
                    self._add_pepper_message_internal(data)
                elif kind == "system":
                    self._add_system_message(data)
                elif kind == "user_voice":
                    self._add_user_message(data, voice=True)
                    threading.Thread(
                        target=self.message_callback, args=(data,), daemon=True
                    ).start()
                elif kind == "recording_state":
                    self._set_recording_internal(data)
            except queue.Empty:
                break

        last_status = None
        while not self.status_queue.empty():
            try:
                last_status = self.status_queue.get_nowait()
            except queue.Empty:
                break
        if last_status is not None:
            try:
                dpg.set_value("status_text", f"Status: {last_status}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def test_callback(message):
        print(f"Received: {message}")
        time.sleep(0.8)
        gui.add_pepper_message(f"Echo: {message}")

    def test_volume(vol):
        print(f"Volume: {vol}")

    gui = PepperDearPyGUI(test_callback, volume_callback=test_volume)
    print("Starting DearPyGUI test…")
    gui.start()
    print("GUI closed.")