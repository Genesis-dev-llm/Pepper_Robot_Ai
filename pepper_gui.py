"""
DearPyGUI Interface for Pepper AI Chat
GPU-accelerated, 60fps, perfect for future video streaming
"""

import dearpygui.dearpygui as dpg
import threading
import queue
import time


class PepperDearPyGUI:
    def __init__(self, message_callback):
        """
        Initialize DearPyGUI interface

        Args:
            message_callback: Called when the user sends a message (text or voice)
        """
        self.message_callback    = message_callback
        self.is_running          = False
        self.message_queue       = queue.Queue()   # Thread-safe updates
        self.status_queue        = queue.Queue()
        # Set True while the text input field has keyboard focus — used by
        # on_press() to prevent PTT from firing while the user is typing.
        self.text_input_focused  = False

    # ------------------------------------------------------------------ #
    #  Public lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def start(self):
        """Start GUI — blocks until the window is closed."""
        self.is_running = True

        dpg.create_context()
        self._setup_window()

        dpg.create_viewport(
            title    = "🤖 Pepper AI Control Dashboard",
            width    = 900,
            height   = 740,
            min_width  = 600,
            min_height = 450,
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

    # ------------------------------------------------------------------ #
    #  Window layout                                                       #
    # ------------------------------------------------------------------ #

    def _setup_window(self):
        with dpg.window(label="Pepper AI Chat", tag="main_window",
                        no_close=True, no_collapse=True):

            # ── Header ──────────────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("🤖 Pepper AI Dashboard",
                             tag="header_text", color=(100, 149, 237))
                dpg.add_spacer(width=20)
                dpg.add_text("Status: Starting...",
                             tag="status_text", color=(150, 150, 150))

            dpg.add_separator()

            # ── Recording indicator (hidden by default) ──────────────────
            with dpg.group(tag="recording_indicator", show=False):
                dpg.add_text("🔴  RECORDING — Release R to send",
                             tag="recording_label", color=(255, 80, 80))
                dpg.add_separator()

            # ── Instructions ─────────────────────────────────────────────
            with dpg.collapsing_header(label="💡 Controls & Instructions",
                                       default_open=True):
                dpg.add_text("Text mode:  type below → Enter / Send")
                dpg.add_text("Voice mode: hold R → speak → release R")
                dpg.add_spacer(height=4)
                dpg.add_text("Terminal — movement & gesture shortcuts:")
                dpg.add_text("  SPACE=Wake/Sleep  |  WASD=Move  |  1-9=Gestures")
                dpg.add_text("  5-7=LED colour    |  X=Quit")

            dpg.add_separator()

            # ── Chat area ────────────────────────────────────────────────
            dpg.add_text("Chat History:", color=(200, 200, 200))
            dpg.add_child_window(tag="chat_window", height=350, border=True)

            dpg.add_separator()

            # ── Text input ───────────────────────────────────────────────
            dpg.add_text("Your Message:", color=(200, 200, 200))
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    tag      = "message_input",
                    hint     = "Type here… or hold R in terminal to speak",
                    width    = -100,
                    on_enter = True,
                    callback = self._send_text_message,
                )
                dpg.add_button(label="Send", width=90,
                               callback=self._send_text_message)

            dpg.add_separator()

            # ── Footer reminder ──────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("Terminal:", color=(150, 150, 150))
                dpg.add_text("SPACE=Wake", color=(100, 200, 100))
                dpg.add_text(" R=Voice",   color=(255, 150, 50))
                dpg.add_text(" WASD=Move", color=(100, 200, 100))
                dpg.add_text(" 1-9=Gesture", color=(100, 200, 100))

        dpg.set_primary_window("main_window", True)

        # Welcome messages
        self._add_system_message("🤖 Pepper AI Control Dashboard started")
        self._add_system_message("Type below, or hold R in the terminal to speak")
        self._add_system_message("Press SPACE in terminal to wake Pepper")

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _send_text_message(self, sender=None, app_data=None):
        message = dpg.get_value("message_input").strip()
        if not message:
            return
        dpg.set_value("message_input", "")
        self._add_user_message(message, voice=False)
        threading.Thread(
            target=self.message_callback, args=(message,), daemon=True
        ).start()

    # ------------------------------------------------------------------ #
    #  Thread-safe public methods                                          #
    # ------------------------------------------------------------------ #

    def add_pepper_message(self, message: str):
        """Queue a Pepper response bubble."""
        self.message_queue.put(("pepper", message))

    def add_system_message(self, message: str):
        """Queue a system/info message."""
        self.message_queue.put(("system", message))

    def add_voice_user_message(self, text: str):
        """Show transcribed voice message as user bubble."""
        self.message_queue.put(("user_voice", text))

    def update_status(self, status: str):
        """Update the status line at the top."""
        self.status_queue.put(status)

    def set_recording(self, recording: bool):
        """Show/hide the recording indicator banner."""
        self.message_queue.put(("recording_state", recording))

    # ------------------------------------------------------------------ #
    #  Main-thread internal renderers                                      #
    # ------------------------------------------------------------------ #

    def _add_user_message(self, message: str, voice: bool = False):
        prefix = "🎙️ You:" if voice else "You:"
        color  = (255, 180, 50) if voice else (66, 135, 245)
        with dpg.group(parent="chat_window"):
            with dpg.group(horizontal=True):
                dpg.add_text(prefix, color=color)
                dpg.add_text(message, wrap=620)
        dpg.set_y_scroll("chat_window", dpg.get_y_scroll_max("chat_window"))

    def _add_pepper_message_internal(self, message: str):
        with dpg.group(parent="chat_window"):
            with dpg.group(horizontal=True):
                dpg.add_text("Pepper:", color=(76, 175, 80))
                dpg.add_text(message, wrap=620)
        dpg.set_y_scroll("chat_window", dpg.get_y_scroll_max("chat_window"))

    def _add_system_message(self, message: str):
        with dpg.group(parent="chat_window"):
            dpg.add_text(f"• {message}", color=(150, 150, 150))
        dpg.set_y_scroll("chat_window", dpg.get_y_scroll_max("chat_window"))

    def _set_recording_internal(self, recording: bool):
        dpg.configure_item("recording_indicator", show=recording)
        if recording:
            dpg.set_value("status_text",
                          "Status: 🔴 RECORDING — release R when done")
        # Status will be updated via status_queue when recording stops

    def _process_queues(self):
        """Drain both queues once per frame (called from main thread)."""
        # Track whether the text input field currently has keyboard focus.
        # on_press() reads this to suppress PTT while the user is typing.
        try:
            self.text_input_focused = dpg.is_item_focused("message_input")
        except Exception:
            self.text_input_focused = False

        while not self.message_queue.empty():
            try:
                kind, data = self.message_queue.get_nowait()
                if kind == "pepper":
                    self._add_pepper_message_internal(data)
                elif kind == "system":
                    self._add_system_message(data)
                elif kind == "user_voice":
                    self._add_user_message(data, voice=True)
                    # Also kick off the message callback
                    threading.Thread(
                        target=self.message_callback, args=(data,), daemon=True
                    ).start()
                elif kind == "recording_state":
                    self._set_recording_internal(data)
            except queue.Empty:
                break

        while not self.status_queue.empty():
            try:
                status = self.status_queue.get_nowait()
                dpg.set_value("status_text", f"Status: {status}")
            except queue.Empty:
                break


# ------------------------------------------------------------------ #
#  Standalone test                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import time

    def test_callback(message):
        print(f"Received: {message}")
        time.sleep(0.8)
        gui.add_pepper_message(f"Echo: {message}")

    gui = PepperDearPyGUI(test_callback)
    print("Starting DearPyGUI test…")
    gui.start()
    print("GUI closed.")