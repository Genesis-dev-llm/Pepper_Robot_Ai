"""
DearPyGUI Interface for Pepper AI Chat
"""

import os
import queue
import threading
import time

import dearpygui.dearpygui as dpg


MAX_CHAT_MESSAGES = 60


class PepperDearPyGUI:
    def __init__(self, message_callback, volume_callback=None, action_callback=None,
                 display_callback=None, clear_display_callback=None):
        self.message_callback       = message_callback
        self.volume_callback        = volume_callback          # fn(int 0–100)
        self.action_callback        = action_callback          # fn(str action_name)
        self.display_callback       = display_callback         # fn(path, sharpen: bool)
        self.clear_display_callback = clear_display_callback   # fn()
        self.is_running             = False
        self.message_queue          = queue.Queue()
        self.status_queue           = queue.Queue()

        # threading.Event is explicitly thread-safe for cross-thread reads.
        # Written by DearPyGUI's main thread (activated/deactivated callbacks).
        # Read by the pynput listener thread on every keypress.
        # .is_set() is a proper memory barrier — no stale-read risk.
        self._input_focused_event = threading.Event()

        # Cached status before entering text mode so we can restore it on exit.
        self._pre_focus_status: str = "Ready"

        self._msg_tag_counter      = 0
        self._msg_tags: list       = []
        self._scroll_pending: bool = False

        # Volume debounce — avoids spamming NAOqi on every pixel of slider drag.
        # At 150ms debounce the user can still scrub smoothly (~7 calls/s max).
        self._volume_last_sent:  float = 0.0
        self._VOLUME_DEBOUNCE:   float = 0.15   # seconds

    # ------------------------------------------------------------------
    # text_input_focused — property backed by threading.Event
    # ------------------------------------------------------------------

    @property
    def text_input_focused(self) -> bool:
        """
        True while the text input box has keyboard focus.

        Backed by a threading.Event so reads from the pynput listener thread
        are properly synchronised with writes from the DearPyGUI main thread.
        Exposed as a bool property so all callers (main.py etc.) are unchanged.
        """
        return self._input_focused_event.is_set()

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

        # Register OS-level file drop — available in DearPyGUI 1.11+.
        # Gracefully skipped on older versions so the button still works.
        self._drop_supported = False
        try:
            dpg.set_viewport_drop_callback(self._on_file_drop)
            self._drop_supported = True
        except AttributeError:
            pass  # Older DPG version — drag-drop unavailable, button still works

        # Update the drag-drop hint text now that we know if it's supported
        try:
            if self._drop_supported:
                dpg.configure_item("display_drag_hint",
                                   default_value="💡 Or drag & drop an image anywhere onto this window")
            else:
                dpg.configure_item("display_drag_hint",
                                   default_value="💡 Drag & drop not available — use the Load Image button")
        except Exception:
            pass

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
                dpg.add_spacer(width=8)
                # Active/idle dot — green=active, red=idle, never overwritten by status
                dpg.add_text("●", tag="active_dot", color=(120, 120, 120))
                dpg.add_spacer(width=4)
                # Connection dot — cyan=connected, grey=offline, set once on connect/disconnect
                dpg.add_text("●", tag="connection_dot", color=(120, 120, 120))
                dpg.add_spacer(width=12)
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

            # ── Volume + quick actions ────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("🔊 Volume:", color=(200, 200, 200))
                dpg.add_slider_int(
                    tag            = "volume_slider",
                    default_value  = 100,
                    min_value      = 0,
                    max_value      = 100,
                    width          = 240,
                    callback       = self._on_volume_changed,
                    format         = "%d%%",
                )
                dpg.add_spacer(width=16)
                dpg.add_button(
                    label    = "💡 Pulse Eyes",
                    width    = 110,
                    callback = lambda: self._on_action("pulse_eyes"),
                )

            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_text("Terminal:", color=(150, 150, 150))
                dpg.add_text(" SPACE=Wake", color=(100, 200, 100))
                dpg.add_text(" R=Voice",    color=(255, 150, 50))
                dpg.add_text(" WASD=Move",  color=(100, 200, 100))
                dpg.add_text(" 1-9=Gesture", color=(100, 200, 100))

            dpg.add_separator()

            # ── Tablet display ────────────────────────────────────────
            with dpg.collapsing_header(label="🖼️ Tablet Display", default_open=False):
                dpg.add_text(
                    "Send an image to Pepper's chest tablet.",
                    color=(180, 180, 180)
                )
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label    = "📂 Load Image",
                        width    = 120,
                        callback = self._open_image_dialog,
                    )
                    dpg.add_button(
                        label    = "🗑️ Clear Display",
                        width    = 120,
                        callback = self._on_clear_display,
                    )
                    dpg.add_spacer(width=12)
                    dpg.add_checkbox(
                        label        = "Sharpen (logos/icons)",
                        tag          = "display_sharpen_checkbox",
                        default_value= False,
                    )
                dpg.add_spacer(height=4)
                # Drag-drop target hint — updated in start() once we know if it's supported
                dpg.add_text(
                    "💡 Checking drag & drop support…",
                    tag   = "display_drag_hint",
                    color = (120, 120, 120)
                )
                dpg.add_spacer(height=2)
                dpg.add_text("No image loaded", tag="display_status_text",
                             color=(150, 150, 150))

        dpg.set_primary_window("main_window", True)

        self._add_system_message("🤖 Pepper AI Control Dashboard started")
        self._add_system_message("Click the input box below to type, or hold R to speak")
        self._add_system_message("Press SPACE (outside input box) to wake Pepper")

    # ------------------------------------------------------------------
    # Tablet display callbacks
    # ------------------------------------------------------------------

    def _on_file_drop(self, sender, app_data):
        """
        Called by DearPyGUI when one or more files are dragged onto the window.

        app_data is a list of dropped file paths. We take the first image file
        found (by extension) and treat it exactly like a file-dialog selection.
        Non-image files are silently ignored so accidentally dropping a .txt
        or folder doesn't do anything unexpected.
        """
        IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
        for path in app_data:
            ext = os.path.splitext(path)[1].lower()
            if ext in IMAGE_EXTS and os.path.isfile(path):
                # Reuse the existing selection handler — same processing pipeline
                self._on_image_selected(sender, {"file_path_name": path})
                return

    def _open_image_dialog(self):
        """Open DearPyGUI's built-in file dialog filtered to image types."""
        # Delete any existing dialog first — clicking Load Image twice
        # would otherwise throw a duplicate-tag error from DPG.
        try:
            dpg.delete_item("image_file_dialog")
        except Exception:
            pass

        dpg.add_file_dialog(
            label            = "Select Image for Pepper's Tablet",
            default_path     = os.path.expanduser("~"),
            callback         = self._on_image_selected,
            cancel_callback  = lambda s, a: None,
            width            = 700,
            height           = 450,
            modal            = True,
            tag              = "image_file_dialog",
            file_count       = 1,
        )
        dpg.add_file_extension(".png",  parent="image_file_dialog", color=(100, 220, 100))
        dpg.add_file_extension(".jpg",  parent="image_file_dialog", color=(100, 220, 100))
        dpg.add_file_extension(".jpeg", parent="image_file_dialog", color=(100, 220, 100))
        dpg.add_file_extension(".bmp",  parent="image_file_dialog", color=(100, 220, 100))
        dpg.add_file_extension(".webp", parent="image_file_dialog", color=(100, 220, 100))
        dpg.add_file_extension(".gif",  parent="image_file_dialog", color=(100, 220, 100))

    def _on_image_selected(self, sender, app_data):
        """Called by DPG when the user confirms a file selection."""
        path = app_data.get("file_path_name", "")
        if not path or not os.path.isfile(path):
            return
        sharpen = dpg.get_value("display_sharpen_checkbox")
        filename = os.path.basename(path)
        try:
            dpg.set_value("display_status_text",
                          f"⏳ Processing: {filename}…")
        except Exception:
            pass
        if self.display_callback:
            self.display_callback(path, sharpen)
        # Update status after callback is fired (callback runs worker thread,
        # so "Sent" appears immediately while processing continues in background)
        try:
            dpg.set_value("display_status_text",
                          f"✅ Sent: {filename}")
        except Exception:
            pass

    def _on_clear_display(self):
        """Called when the user clicks Clear Display."""
        if self.clear_display_callback:
            self.clear_display_callback()
        try:
            dpg.set_value("display_status_text", "No image loaded")
        except Exception:
            pass

    def _on_input_activated(self):
        """
        Called by DearPyGUI when the text input receives focus.

        Sets the event so the pynput thread stops routing keypresses to
        robot controls, and updates the status bar so the user knows they're
        in text-input mode rather than wondering why WASD isn't working.
        """
        self._input_focused_event.set()
        # Cache whatever the status was before entering text mode so we can
        # restore it cleanly when the user finishes typing.
        try:
            current = dpg.get_value("status_text")
            self._pre_focus_status = current.replace("Status: ", "", 1)
        except Exception:
            self._pre_focus_status = "Ready"
        try:
            dpg.set_value("status_text", "Status: ✏️ Text mode — robot controls paused")
        except Exception:
            pass

    def _on_input_deactivated(self):
        """
        Called by DearPyGUI when the text input loses focus.

        Clears the event and restores the previous status so the user
        can see robot controls are active again.
        """
        self._input_focused_event.clear()
        try:
            dpg.set_value("status_text", f"Status: {self._pre_focus_status}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Action callback
    # ------------------------------------------------------------------

    def _on_action(self, action: str):
        if self.action_callback:
            self.action_callback(action)

    # ------------------------------------------------------------------
    # Active/idle dot
    # ------------------------------------------------------------------

    def set_robot_active(self, active: bool):
        """Update the persistent dot in the header. Green=active, red=idle."""
        try:
            color = (80, 200, 80) if active else (200, 60, 60)
            dpg.configure_item("active_dot", color=color)
        except Exception:
            pass

    def set_connection_status(self, connected: bool):
        """
        Update the connection dot in the header.

        Cyan = connected to Pepper hardware.
        Grey = offline / not reachable.

        Thread-safe — queued so it can be called before the DPG render loop
        starts (e.g. right after pepper.connect() during initialisation).
        """
        self.message_queue.put(("connection_status", connected))

    # ------------------------------------------------------------------
    # Volume callback
    # ------------------------------------------------------------------

    def _on_volume_changed(self, sender, app_data):
        """
        Debounced volume callback — fires at most every 150ms while dragging.

        Without debouncing, sliding from 0→100 fires ~100 consecutive NAOqi
        calls (setVolume + setOutputVolume each), which stacks up and causes
        audio glitches. At 150ms we cap it at ~7 calls/s which feels responsive
        but doesn't overload the NAOqi service.
        """
        now = time.time()
        if now - self._volume_last_sent < self._VOLUME_DEBOUNCE:
            return
        self._volume_last_sent = now
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
                elif kind == "connection_status":
                    try:
                        color = (0, 200, 200) if data else (120, 120, 120)
                        dpg.configure_item("connection_dot", color=color)
                    except Exception:
                        pass
            except queue.Empty:
                break

        last_status = None
        while not self.status_queue.empty():
            try:
                last_status = self.status_queue.get_nowait()
            except queue.Empty:
                break
        if last_status is not None:
            # Don't overwrite the text-mode indicator while the input is focused
            if not self.text_input_focused:
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