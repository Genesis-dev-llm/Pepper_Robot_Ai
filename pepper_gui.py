"""
DearPyGUI Interface for Pepper AI Chat

Changes in this version (Group 4 — GUI State & Input Handling):
- Native OS file picker (zenity → kdialog → DPG fallback).
  The DPG built-in rendered its own folder tree widget; now the system's
  real file manager opens exactly like any other professional app.
  The subprocess runs in a daemon thread so DPG's render loop never blocks.
- Dedicated drag-and-drop zone with proper visual styling.
  The drop zone shows clearly what to do instead of a single hint line.
  Also detects Wayland and shows a warning if drops won't work.
- Volume slider final-value fix.
  The 150ms debounce meant stopping within that window silently dropped
  the last position. An item_deactivated_handler now always sends the
  final value on mouse release regardless of debounce state.
- Pre-focus status restoration: guard against overwriting "RECORDING"
  indicator if the user somehow clicks the input while recording is active.
"""

import os
import queue
import shutil
import subprocess
import threading
import time

import dearpygui.dearpygui as dpg


MAX_CHAT_MESSAGES = 60

# Image extensions accepted by the file picker and drag-drop handler
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}


def _detect_wayland() -> bool:
    """Return True if the session appears to be running under Wayland."""
    return bool(
        os.environ.get("WAYLAND_DISPLAY") or
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    )


def _pick_file_native(title: str = "Select Image") -> str | None:
    """
    Open the OS-native file picker and return the chosen path, or None.

    Try order on Linux:
      1. zenity  (GTK, ships with GNOME — most ThinkPads on Ubuntu/Fedora)
      2. kdialog (Qt, ships with KDE Plasma)
      3. None    — caller falls back to DPG built-in

    Runs synchronously; always call from a worker thread.
    """
    filter_arg = "image/png image/jpeg image/bmp image/webp image/gif"

    # ── zenity ──────────────────────────────────────────────────────────
    if shutil.which("zenity"):
        try:
            result = subprocess.run(
                [
                    "zenity",
                    "--file-selection",
                    f"--title={title}",
                    "--file-filter=Images (png jpg jpeg bmp webp gif)|*.png *.jpg *.jpeg *.bmp *.webp *.gif",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            path = result.stdout.strip()
            return path if path else None
        except Exception:
            pass

    # ── kdialog ─────────────────────────────────────────────────────────
    if shutil.which("kdialog"):
        try:
            result = subprocess.run(
                [
                    "kdialog",
                    "--getopenfilename",
                    os.path.expanduser("~"),
                    "*.png *.jpg *.jpeg *.bmp *.webp *.gif|Images",
                    "--title", title,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            path = result.stdout.strip()
            return path if path else None
        except Exception:
            pass

    # ── nothing found ───────────────────────────────────────────────────
    return None


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
        # The item_deactivated_handler on the slider always sends the final
        # value on mouse release to cover the within-debounce-window edge case.
        self._volume_last_sent:  float = 0.0
        self._VOLUME_DEBOUNCE:   float = 0.15   # seconds

        # Wayland detection — drag-drop via XDrop doesn't work under Wayland
        self._is_wayland      = _detect_wayland()
        self._drop_supported  = False

        # True while the native file picker thread is open — prevents double-launch
        self._picker_open = False

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

        # ── OS-level file drop ──────────────────────────────────────────
        # DPG 1.11+ exposes set_viewport_drop_callback for whole-window drops.
        # Under Wayland, the X11 XDND protocol doesn't fire so we warn the user.
        # The Load Image button always works regardless.
        if self._is_wayland:
            self._update_drop_zone_hint(
                "⚠️  Wayland detected — drag & drop may not work. "
                "Use the Load Image button instead.",
                color=(255, 180, 50),
            )
        else:
            try:
                dpg.set_viewport_drop_callback(self._on_file_drop)
                self._drop_supported = True
                self._update_drop_zone_hint(
                    "💡 Or drag & drop an image anywhere onto this window",
                    color=(120, 120, 120),
                )
            except AttributeError:
                # Older DPG build — no viewport drop API
                self._update_drop_zone_hint(
                    "💡 Drag & drop unavailable on this DPG version — "
                    "use the Load Image button",
                    color=(120, 120, 120),
                )

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

            # Keyboard gate — blocks robot controls while the input is focused
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

            # Volume final-value fix: always send the exact resting position
            # when the user lifts the mouse, regardless of debounce state.
            with dpg.item_handler_registry(tag="volume_release_handler"):
                dpg.add_item_deactivated_handler(callback=self._on_volume_released)
            dpg.bind_item_handler_registry("volume_slider", "volume_release_handler")

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
                    color=(180, 180, 180),
                )
                dpg.add_spacer(height=6)

                # ── Drop zone ─────────────────────────────────────────
                # Visually distinct panel styled like a real file drop target.
                # The actual drop handling is registered on the whole viewport
                # (set_viewport_drop_callback in start()), so dropping anywhere
                # on the window works — this panel is the visual affordance.
                with dpg.child_window(
                    tag    = "drop_zone_panel",
                    height = 72,
                    border = True,
                ):
                    dpg.add_spacer(height=8)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=20)
                        with dpg.group():
                            dpg.add_text(
                                "📂  Drop an image file here",
                                tag   = "drop_zone_title",
                                color = (160, 160, 180),
                            )
                            dpg.add_text(
                                "PNG · JPG · JPEG · BMP · WEBP · GIF",
                                tag   = "drop_zone_types",
                                color = (100, 100, 120),
                            )

                dpg.add_spacer(height=6)

                # ── Buttons row ───────────────────────────────────────
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label    = "📂 Load Image…",
                        width    = 130,
                        callback = self._open_image_dialog,
                    )
                    dpg.add_button(
                        label    = "🗑️ Clear Display",
                        width    = 120,
                        callback = self._on_clear_display,
                    )
                    dpg.add_spacer(width=12)
                    dpg.add_checkbox(
                        label         = "Sharpen (logos / icons)",
                        tag           = "display_sharpen_checkbox",
                        default_value = False,
                    )

                dpg.add_spacer(height=4)

                # Drag-drop support hint — updated in start() once we know the
                # display server and DPG version
                dpg.add_text(
                    "💡 Checking drag & drop support…",
                    tag   = "display_drag_hint",
                    color = (120, 120, 120),
                )
                dpg.add_spacer(height=2)
                dpg.add_text(
                    "No image loaded",
                    tag   = "display_status_text",
                    color = (150, 150, 150),
                )

        dpg.set_primary_window("main_window", True)

        self._add_system_message("🤖 Pepper AI Control Dashboard started")
        self._add_system_message("Click the input box below to type, or hold R to speak")
        self._add_system_message("Press SPACE (outside input box) to wake Pepper")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_drop_zone_hint(self, text: str, color=(120, 120, 120)):
        """Update the drag-drop hint text below the drop zone buttons."""
        try:
            dpg.configure_item("display_drag_hint", default_value=text, color=color)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Tablet display callbacks
    # ------------------------------------------------------------------

    def _on_file_drop(self, sender, app_data):
        """
        Called by DearPyGUI when one or more files are dragged onto the window.

        app_data is a list of dropped file paths (DPG 1.11+).
        We take the first image file found and treat it exactly like a
        file-dialog selection. Non-image files are silently ignored.
        """
        for path in (app_data or []):
            ext = os.path.splitext(path)[1].lower()
            if ext in _IMAGE_EXTS and os.path.isfile(path):
                self._on_image_selected(sender, {"file_path_name": path})
                return

    def _open_image_dialog(self):
        """
        Open a file picker to select an image for the tablet.

        On Linux: launches the OS-native file picker (zenity or kdialog)
        in a daemon thread so the DPG render loop never blocks. The result
        is queued back to the main thread via message_queue.

        If neither zenity nor kdialog is installed, falls back to DPG's
        built-in file dialog (which renders its own folder tree — functional
        but not native-looking).
        """
        # Prevent double-launch if the user clicks Load Image twice quickly
        if self._picker_open:
            return

        # Check whether a native picker is available before spawning a thread
        has_native = bool(shutil.which("zenity") or shutil.which("kdialog"))

        if has_native:
            self._picker_open = True
            threading.Thread(
                target = self._native_picker_thread,
                daemon = True,
                name   = "FilePicker",
            ).start()
        else:
            # Fallback: DPG built-in (must run on main thread — call directly)
            self._open_dpg_file_dialog()

    def _native_picker_thread(self):
        """
        Worker thread: open native picker, queue result back to main thread.
        Runs entirely off the DPG thread so the UI stays responsive.
        """
        try:
            path = _pick_file_native(title="Select Image for Pepper's Tablet")
            if path:
                self.message_queue.put(("file_selected", path))
        finally:
            self._picker_open = False

    def _open_dpg_file_dialog(self):
        """
        Fallback: DPG's built-in file dialog.
        Only used when zenity/kdialog are not installed.
        """
        try:
            dpg.delete_item("image_file_dialog")
        except Exception:
            pass

        dpg.add_file_dialog(
            label           = "Select Image for Pepper's Tablet",
            default_path    = os.path.expanduser("~"),
            callback        = self._on_image_selected,
            cancel_callback = lambda s, a: None,
            width           = 700,
            height          = 450,
            modal           = True,
            tag             = "image_file_dialog",
            file_count      = 1,
        )
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"):
            dpg.add_file_extension(ext, parent="image_file_dialog",
                                   color=(100, 220, 100))

    def _on_image_selected(self, sender, app_data):
        """
        Called when a file is confirmed — either from DPG dialog, native picker
        (via message_queue), or drag-drop. Single processing pipeline for all
        three entry points.
        """
        path = app_data.get("file_path_name", "")
        if not path or not os.path.isfile(path):
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in _IMAGE_EXTS:
            try:
                dpg.set_value("display_status_text",
                              f"⚠️  Not an image file: {os.path.basename(path)}")
            except Exception:
                pass
            return

        sharpen  = dpg.get_value("display_sharpen_checkbox")
        filename = os.path.basename(path)
        try:
            dpg.set_value("display_status_text", f"⏳ Processing: {filename}…")
        except Exception:
            pass
        if self.display_callback:
            self.display_callback(path, sharpen)
        # Callback fires a worker thread, so "Sent" appears immediately
        try:
            dpg.set_value("display_status_text", f"✅ Sent: {filename}")
        except Exception:
            pass

    def _on_clear_display(self):
        if self.clear_display_callback:
            self.clear_display_callback()
        try:
            dpg.set_value("display_status_text", "No image loaded")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Input focus handlers
    # ------------------------------------------------------------------

    def _on_input_activated(self):
        """
        Called when the text input receives focus.

        Blocks robot controls via the threading.Event and updates the status
        bar. Does not cache "RECORDING" as the pre-focus status — if Pepper
        is actively recording when the user clicks the input box, we want
        the recording indicator to stay visible when they unfocus.
        """
        self._input_focused_event.set()
        try:
            current = dpg.get_value("status_text").replace("Status: ", "", 1)
            # Don't overwrite a recording status — it's shown separately but
            # we still preserve it so defocus restores the right message.
            if "RECORDING" not in current:
                self._pre_focus_status = current
        except Exception:
            self._pre_focus_status = "Ready"
        try:
            dpg.set_value("status_text", "Status: ✏️ Text mode — robot controls paused")
        except Exception:
            pass

    def _on_input_deactivated(self):
        """
        Called when the text input loses focus.
        Restores the pre-focus status unless Pepper is actively recording,
        in which case the recording status takes priority.
        """
        self._input_focused_event.clear()
        try:
            # Don't restore over an active recording indicator
            current = dpg.get_value("status_text")
            if "RECORDING" not in current:
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
    # Active/idle and connection dots
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
    # Volume callbacks
    # ------------------------------------------------------------------

    def _on_volume_changed(self, sender, app_data):
        """
        Debounced volume callback — fires at most every 150ms while dragging.

        Without debouncing, sliding from 0→100 fires ~100 consecutive NAOqi
        calls (setVolume + setOutputVolume each), which stacks up and causes
        audio glitches. At 150ms we cap it at ~7 calls/s which feels responsive
        but doesn't overload the NAOqi service.

        The item_deactivated_handler (_on_volume_released) always sends the
        final value on mouse release to cover the within-window edge case.
        """
        now = time.time()
        if now - self._volume_last_sent < self._VOLUME_DEBOUNCE:
            return
        self._volume_last_sent = now
        if self.volume_callback:
            self.volume_callback(int(app_data))

    def _on_volume_released(self):
        """
        Called when the user releases the volume slider (item_deactivated).

        Guarantees the final resting value is always sent to NAOqi even if
        the user stopped dragging within the 150ms debounce window.
        Resets the debounce clock so the next drag starts fresh.
        """
        try:
            val = dpg.get_value("volume_slider")
        except Exception:
            return
        self._volume_last_sent = time.time()
        if self.volume_callback:
            self.volume_callback(int(val))

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
                elif kind == "file_selected":
                    # Result from the native file picker thread.
                    # Must be handled here (main DPG thread) because
                    # _on_image_selected calls dpg.set_value/get_value.
                    self._on_image_selected(None, {"file_path_name": data})
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