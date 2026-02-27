"""
DearPyGUI Interface for Pepper AI Chat

Changes from previous version:
- set_robot_active() now routes through message_queue (thread-safe from pynput).
- add_voice_user_message() replaced by add_chat_message(text, source) — no
  thread spawning from inside the DPG render loop.
- "Clear Conversation" and "Reconnect" buttons added (→ action_callback).
- TTS tier label displayed in status area.
- Audio level meter shown during recording, hidden otherwise.
- MAX_CHAT_MESSAGES raised to 100.
"""

import os
import queue
import shutil
import subprocess
import threading
import time

import dearpygui.dearpygui as dpg


MAX_CHAT_MESSAGES = 100

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}


def _detect_wayland() -> bool:
    return bool(
        os.environ.get("WAYLAND_DISPLAY") or
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    )


def _pick_file_native(title: str = "Select Image") -> str | None:
    if shutil.which("zenity"):
        try:
            result = subprocess.run(
                ["zenity", "--file-selection", f"--title={title}",
                 "--file-filter=Images (png jpg jpeg bmp webp gif)|*.png *.jpg *.jpeg *.bmp *.webp *.gif"],
                capture_output=True, text=True, timeout=120,
            )
            path = result.stdout.strip()
            return path if path else None
        except Exception:
            pass
    if shutil.which("kdialog"):
        try:
            result = subprocess.run(
                ["kdialog", "--getopenfilename", os.path.expanduser("~"),
                 "*.png *.jpg *.jpeg *.bmp *.webp *.gif|Images", "--title", title],
                capture_output=True, text=True, timeout=120,
            )
            path = result.stdout.strip()
            return path if path else None
        except Exception:
            pass
    return None


class PepperDearPyGUI:
    def __init__(self, message_callback, volume_callback=None, action_callback=None,
                 display_callback=None, clear_display_callback=None):
        self.message_callback       = message_callback
        self.volume_callback        = volume_callback
        self.action_callback        = action_callback
        self.display_callback       = display_callback
        self.clear_display_callback = clear_display_callback
        self.is_running             = False

        self.message_queue = queue.Queue()
        self.status_queue  = queue.Queue()

        self._input_focused_event = threading.Event()
        self._pre_focus_status    = "Ready"

        self._msg_tag_counter = 0
        self._msg_tags: list  = []
        self._scroll_pending  = False

        self._volume_last_sent = 0.0
        self._VOLUME_DEBOUNCE  = 0.15

        self._is_wayland     = _detect_wayland()
        self._drop_supported = False
        self._picker_open    = False

    # ── text_input_focused ─────────────────────────────────────────────────────

    @property
    def text_input_focused(self) -> bool:
        return self._input_focused_event.is_set()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        self.is_running = True
        dpg.create_context()
        self._setup_window()
        dpg.create_viewport(
            title="🤖 Pepper AI Control Dashboard",
            width=900, height=820, min_width=600, min_height=500,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

        if self._is_wayland:
            self._update_drop_zone_hint(
                "⚠️  Wayland detected — drag & drop may not work. Use Load Image.",
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
                self._update_drop_zone_hint(
                    "💡 Drag & drop unavailable on this DPG version — use Load Image",
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

    # ── Window layout ──────────────────────────────────────────────────────────

    def _setup_window(self):
        with dpg.window(label="Pepper AI Chat", tag="main_window",
                        no_close=True, no_collapse=True):

            # Header row
            with dpg.group(horizontal=True):
                dpg.add_text("🤖 Pepper AI Dashboard", color=(100, 149, 237))
                dpg.add_spacer(width=8)
                dpg.add_text("●", tag="active_dot",     color=(120, 120, 120))
                dpg.add_spacer(width=4)
                dpg.add_text("●", tag="connection_dot", color=(120, 120, 120))
                dpg.add_spacer(width=12)
                dpg.add_text("Status: Starting…", tag="status_text", color=(150, 150, 150))
                dpg.add_spacer(width=12)
                dpg.add_text("", tag="tts_tier_text", color=(100, 180, 100))

            dpg.add_separator()

            # Recording indicator
            with dpg.group(tag="recording_indicator", show=False):
                dpg.add_text("🔴  RECORDING — Release R to send",
                             tag="recording_label", color=(255, 80, 80))
                dpg.add_progress_bar(
                    tag=    "audio_level_bar",
                    default_value=0.0,
                    width=-1,
                    overlay="",
                )
                dpg.add_separator()

            with dpg.collapsing_header(label="💡 Controls", default_open=True):
                dpg.add_text("Text: click input → type → Enter/Send")
                dpg.add_text("Voice: hold R → speak → release")
                dpg.add_text("SPACE=Wake/Sleep  |  WASD=Move  |  1-9=Gestures  |  5-7=LED")

            dpg.add_separator()

            dpg.add_text("Chat History:", color=(200, 200, 200))
            dpg.add_child_window(tag="chat_window", height=320, border=True)

            dpg.add_separator()

            dpg.add_text("Your Message:", color=(200, 200, 200))
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    tag="message_input",
                    hint="Click here to type… or hold R to speak",
                    width=-100,
                    on_enter=True,
                    callback=self._send_text_message,
                )
                dpg.add_button(label="Send", width=90, callback=self._send_text_message)

            with dpg.item_handler_registry(tag="input_focus_handler"):
                dpg.add_item_activated_handler(callback=self._on_input_activated)
                dpg.add_item_deactivated_handler(callback=self._on_input_deactivated)
            dpg.bind_item_handler_registry("message_input", "input_focus_handler")

            dpg.add_separator()

            # Volume + quick actions
            with dpg.group(horizontal=True):
                dpg.add_text("🔊 Volume:", color=(200, 200, 200))
                dpg.add_slider_int(
                    tag="volume_slider", default_value=100,
                    min_value=0, max_value=100, width=200,
                    callback=self._on_volume_changed, format="%d%%",
                )
                dpg.add_spacer(width=10)
                dpg.add_button(label="💡 Pulse Eyes", width=110,
                               callback=lambda: self._on_action("pulse_eyes"))
                dpg.add_spacer(width=6)
                dpg.add_button(label="🔄 Clear Chat", width=110,
                               callback=lambda: self._on_action("clear_conversation"))
                dpg.add_spacer(width=6)
                dpg.add_button(label="📡 Reconnect", width=110,
                               callback=lambda: self._on_action("reconnect"))

            with dpg.item_handler_registry(tag="volume_release_handler"):
                dpg.add_item_deactivated_handler(callback=self._on_volume_released)
            dpg.bind_item_handler_registry("volume_slider", "volume_release_handler")

            dpg.add_separator()

            # Tablet display (collapsible)
            with dpg.collapsing_header(label="🖼️ Tablet Display", default_open=False):
                dpg.add_text("Send an image to Pepper's chest tablet.", color=(180, 180, 180))
                dpg.add_spacer(height=6)

                with dpg.child_window(tag="drop_zone_panel", height=72, border=True):
                    dpg.add_spacer(height=8)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=20)
                        with dpg.group():
                            dpg.add_text("📂  Drop an image file here",
                                         tag="drop_zone_title", color=(160, 160, 180))
                            dpg.add_text("PNG · JPG · JPEG · BMP · WEBP · GIF",
                                         tag="drop_zone_types", color=(100, 100, 120))

                dpg.add_spacer(height=6)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="📂 Load Image…", width=130,
                                   callback=self._open_image_dialog)
                    dpg.add_button(label="🗑️ Clear Display", width=120,
                                   callback=self._on_clear_display)
                    dpg.add_spacer(width=12)
                    dpg.add_checkbox(label="Sharpen (logos / icons)",
                                     tag="display_sharpen_checkbox", default_value=False)

                dpg.add_spacer(height=4)
                dpg.add_text("💡 Checking drag & drop support…",
                             tag="display_drag_hint", color=(120, 120, 120))
                dpg.add_spacer(height=2)
                dpg.add_text("No image loaded", tag="display_status_text", color=(150, 150, 150))

        dpg.set_primary_window("main_window", True)
        self._add_system_message("🤖 Pepper AI Control Dashboard started")
        self._add_system_message("Click the input box to type, or hold R to speak")
        self._add_system_message("Press SPACE (outside input) to wake Pepper")

    # ── Thread-safe public methods ─────────────────────────────────────────────

    def add_pepper_message(self, message: str):
        self.message_queue.put(("pepper", message))

    def add_system_message(self, message: str):
        self.message_queue.put(("system", message))

    def add_chat_message(self, text: str, source: str = "text"):
        """
        Display a user message. source='voice' for transcribed speech,
        source='text' for typed input. No thread spawning — that's main.py's job.
        """
        self.message_queue.put(("user_display", (text, source)))

    def update_status(self, status: str):
        self.status_queue.put(status)

    def set_recording(self, recording: bool):
        self.message_queue.put(("recording_state", recording))

    def set_robot_active(self, active: bool):
        """Thread-safe — routes through queue instead of direct DPG call."""
        self.message_queue.put(("robot_active", active))

    def set_connection_status(self, connected: bool):
        self.message_queue.put(("connection_status", connected))

    def update_tts_tier(self, label: str):
        """Show which TTS tier fired (e.g. 'Tier 2 (Edge)')."""
        self.message_queue.put(("tts_tier", label))

    def update_audio_level(self, level: float):
        """Update the audio level bar (0.0–1.0). Called ~10x/sec during recording."""
        self.message_queue.put(("audio_level", level))

    # ── Input focus handlers ───────────────────────────────────────────────────

    def _on_input_activated(self):
        self._input_focused_event.set()
        try:
            current = dpg.get_value("status_text").replace("Status: ", "", 1)
            if "RECORDING" not in current:
                self._pre_focus_status = current
        except Exception:
            self._pre_focus_status = "Ready"
        try:
            dpg.set_value("status_text", "Status: ✏️ Text mode — robot controls paused")
        except Exception:
            pass

    def _on_input_deactivated(self):
        self._input_focused_event.clear()
        try:
            current = dpg.get_value("status_text")
            if "RECORDING" not in current:
                dpg.set_value("status_text", f"Status: {self._pre_focus_status}")
        except Exception:
            pass

    # ── Action callback ────────────────────────────────────────────────────────

    def _on_action(self, action: str):
        if self.action_callback:
            self.action_callback(action)

    # ── Volume callbacks ───────────────────────────────────────────────────────

    def _on_volume_changed(self, sender, app_data):
        now = time.time()
        if now - self._volume_last_sent < self._VOLUME_DEBOUNCE:
            return
        self._volume_last_sent = now
        if self.volume_callback:
            self.volume_callback(int(app_data))

    def _on_volume_released(self):
        try:
            val = dpg.get_value("volume_slider")
        except Exception:
            return
        self._volume_last_sent = time.time()
        if self.volume_callback:
            self.volume_callback(int(val))

    # ── Send callback ──────────────────────────────────────────────────────────

    def _send_text_message(self, sender=None, app_data=None):
        message = dpg.get_value("message_input").strip()
        if not message:
            return
        dpg.set_value("message_input", "")
        dpg.focus_item("main_window")
        self._add_user_message(message, voice=False)
        if self.message_callback:
            self.message_callback(message)

    # ── Tablet display callbacks ───────────────────────────────────────────────

    def _on_file_drop(self, sender, app_data):
        for path in (app_data or []):
            ext = os.path.splitext(path)[1].lower()
            if ext in _IMAGE_EXTS and os.path.isfile(path):
                self._on_image_selected(sender, {"file_path_name": path})
                return

    def _open_image_dialog(self):
        if self._picker_open:
            return
        has_native = bool(shutil.which("zenity") or shutil.which("kdialog"))
        if has_native:
            self._picker_open = True
            threading.Thread(target=self._native_picker_thread, daemon=True, name="FilePicker").start()
        else:
            self._open_dpg_file_dialog()

    def _native_picker_thread(self):
        try:
            path = _pick_file_native(title="Select Image for Pepper's Tablet")
            if path:
                self.message_queue.put(("file_selected", path))
        finally:
            self._picker_open = False

    def _open_dpg_file_dialog(self):
        try:
            dpg.delete_item("image_file_dialog")
        except Exception:
            pass
        dpg.add_file_dialog(
            label="Select Image", default_path=os.path.expanduser("~"),
            callback=self._on_image_selected, cancel_callback=lambda s, a: None,
            width=700, height=450, modal=True, tag="image_file_dialog", file_count=1,
        )
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"):
            dpg.add_file_extension(ext, parent="image_file_dialog", color=(100, 220, 100))

    def _on_image_selected(self, sender, app_data):
        path = app_data.get("file_path_name", "")
        if not path or not os.path.isfile(path):
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in _IMAGE_EXTS:
            try:
                dpg.set_value("display_status_text",
                              f"⚠️  Not an image: {os.path.basename(path)}")
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

    def _update_drop_zone_hint(self, text: str, color=(120, 120, 120)):
        try:
            dpg.configure_item("display_drag_hint", default_value=text, color=color)
        except Exception:
            pass

    # ── Main-thread renderers ──────────────────────────────────────────────────

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
        else:
            try:
                dpg.set_value("audio_level_bar", 0.0)
            except Exception:
                pass

    # ── Frame-loop queue drain ─────────────────────────────────────────────────

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
                elif kind == "user_display":
                    text, source = data
                    self._add_user_message(text, voice=(source == "voice"))
                elif kind == "recording_state":
                    self._set_recording_internal(data)
                elif kind == "robot_active":
                    # Thread-safe DPG configure call — from render thread
                    try:
                        color = (80, 200, 80) if data else (200, 60, 60)
                        dpg.configure_item("active_dot", color=color)
                    except Exception:
                        pass
                elif kind == "connection_status":
                    try:
                        color = (0, 200, 200) if data else (120, 120, 120)
                        dpg.configure_item("connection_dot", color=color)
                    except Exception:
                        pass
                elif kind == "tts_tier":
                    try:
                        dpg.set_value("tts_tier_text", f"[{data}]")
                    except Exception:
                        pass
                elif kind == "audio_level":
                    try:
                        dpg.set_value("audio_level_bar", float(data))
                    except Exception:
                        pass
                elif kind == "file_selected":
                    self._on_image_selected(None, {"file_path_name": data})
            except queue.Empty:
                break

        last_status = None
        while not self.status_queue.empty():
            try:
                last_status = self.status_queue.get_nowait()
            except queue.Empty:
                break
        if last_status is not None and not self.text_input_focused:
            try:
                dpg.set_value("status_text", f"Status: {last_status}")
            except Exception:
                pass


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def test_callback(message):
        print(f"Received: {message}")
        time.sleep(0.8)
        gui.add_pepper_message(f"Echo: {message}")

    gui = PepperDearPyGUI(test_callback, volume_callback=lambda v: print(f"Vol: {v}"))
    gui.start()