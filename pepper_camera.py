"""
Pepper Camera — NAOqi ALVideoDevice streaming for DPG display

Architecture:
  PepperCamera.start()
    └─ _capture_loop() [daemon thread @ ~15fps]
         └─ ALVideoDevice.getImageRemote()
              └─ RGB bytes → numpy RGBA float32
                   └─ _frame_buffer + _dirty flag

DPG integration (in pepper_gui.py or standalone):
  1. Register a raw_texture: 320 × 240 × 4 floats (all zeros)
  2. Each render frame, call camera.get_frame() — returns array only if dirty
  3. If array returned: dpg.set_value(texture_tag, array.tolist())

Resolution: QVGA (320×240) default — ~307k floats per frame, ~2-3ms conversion.
Bumping to VGA (640×480) works but conversion takes ~10ms.

The module is fully optional — if NAOqi is unavailable or the camera
fails to subscribe, PepperCamera.start() returns False and the caller
should simply not show the camera section in the GUI.
"""

import threading
import time
from typing import Optional

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import qi as _qi
    _QI_AVAILABLE = True
except ImportError:
    _qi = None
    _QI_AVAILABLE = False


# NAOqi camera constants
_CAMERA_TOP          = 0   # front/top camera
_RESOLUTION_QVGA     = 1   # 320×240
_RESOLUTION_VGA      = 2   # 640×480
_COLOR_SPACE_RGB     = 11  # 3-channel RGB
_FPS_15              = 15


class PepperCamera:
    """
    Manages a background capture loop from Pepper's top camera.

    Usage:
        cam = PepperCamera(session)
        if cam.start():
            # in render loop:
            frame = cam.get_frame()   # None if no new frame
            if frame is not None:
                dpg.set_value(texture_tag, frame.tolist())
        cam.stop()
    """

    def __init__(
        self,
        session,
        resolution: int = _RESOLUTION_QVGA,
        fps:        int = _FPS_15,
    ):
        self._session    = session
        self._resolution = resolution
        self._fps        = fps

        # Dimensions are set after subscribe
        self.width  = 320 if resolution == _RESOLUTION_QVGA else 640
        self.height = 240 if resolution == _RESOLUTION_QVGA else 480

        self._video_device = None
        self._client_name  = "PepperCamClient"

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.connected = False

        self._frame_lock  = threading.Lock()
        self._frame_buffer: Optional["np.ndarray"] = None  # float32 RGBA, shape (H*W*4,)
        self._dirty = False  # True when a new frame is available

    def start(self) -> bool:
        """Subscribe to ALVideoDevice and start the capture thread."""
        if not _QI_AVAILABLE or not _NUMPY_AVAILABLE:
            print("⚠️  PepperCamera: qi or numpy unavailable — camera disabled")
            return False
        try:
            self._video_device = self._session.service("ALVideoDevice")
            self._client_name  = self._video_device.subscribeCamera(
                "PepperCam",
                _CAMERA_TOP,
                self._resolution,
                _COLOR_SPACE_RGB,
                self._fps,
            )
            self.connected = True
            print(f"📷 Camera subscribed ({self.width}×{self.height} @ {self._fps}fps)")
        except Exception as e:
            print(f"⚠️  Camera subscribe failed: {e}")
            return False

        # Pre-allocate frame buffer (black RGBA)
        self._frame_buffer = np.zeros(self.width * self.height * 4, dtype=np.float32)

        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraCapture",
        )
        self._thread.start()
        return True

    def stop(self):
        """Stop capture and unsubscribe from ALVideoDevice."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._video_device and self.connected:
            try:
                self._video_device.unsubscribe(self._client_name)
            except Exception:
                pass
        self.connected = False
        print("📷 Camera stopped")

    def get_frame(self) -> Optional["np.ndarray"]:
        """
        Returns the latest frame as a flat float32 RGBA array (H*W*4,)
        if a new frame is available since the last call, else None.
        Thread-safe.
        """
        with self._frame_lock:
            if not self._dirty:
                return None
            self._dirty = False
            return self._frame_buffer.copy()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _capture_loop(self):
        interval = 1.0 / self._fps
        while self._running:
            t0 = time.monotonic()
            try:
                image = self._video_device.getImageRemote(self._client_name)
                if image is not None:
                    self._process_frame(image)
            except Exception as e:
                print(f"⚠️  Camera capture error: {e}")
                time.sleep(1.0)
                continue
            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, interval - elapsed)
            time.sleep(sleep_for)

    def _process_frame(self, image):
        """
        Convert NAOqi image tuple to flat float32 RGBA numpy array.
        image[6] contains raw RGB bytes (width × height × 3 bytes).
        """
        try:
            raw_bytes = image[6]
            # bytes → uint8 array → float32 [0,1]
            rgb = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) / 255.0
            # RGB → RGBA by inserting alpha=1.0 channel
            rgba = np.ones(self.width * self.height * 4, dtype=np.float32)
            rgba[0::4] = rgb[0::3]  # R
            rgba[1::4] = rgb[1::3]  # G
            rgba[2::4] = rgb[2::3]  # B
            # rgba[3::4] stays 1.0 (alpha)
            with self._frame_lock:
                self._frame_buffer[:] = rgba
                self._dirty = True
        except Exception as e:
            print(f"⚠️  Frame processing error: {e}")

    @property
    def blank_frame(self) -> list:
        """Return a black RGBA frame as a list, suitable for DPG texture init."""
        return [0.0] * (self.width * self.height * 4)