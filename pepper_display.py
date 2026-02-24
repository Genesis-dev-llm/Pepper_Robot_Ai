"""
Pepper Tablet Display Manager

Serves images to Pepper's chest tablet (1280×800 Android display) over HTTP.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  GUI thread  →  user picks file  →  display_callback()  │
  │                                          ↓               │
  │  Worker thread  →  process image  →  swap buffer        │
  │                                          ↓               │
  │  HTTP server thread  ← always running, serves /image    │
  │                                          ↓               │
  │  ALTabletService.showImage(url)  ← one-shot per image   │
  └─────────────────────────────────────────────────────────┘

The HTTP server serves a single in-memory PNG buffer. When a new image
arrives the buffer is swapped atomically, and the tablet is told to
reload the URL. No file system writes are needed.

ALTabletService uses HTTP — the tablet's Android browser fetches the
image from the control PC over the local network. This is the standard
approach for all Pepper tablet display work.
"""

import logging
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from typing import Optional, TYPE_CHECKING

try:
    from PIL import Image, ImageFilter
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logging.warning("Pillow not installed — tablet display will be disabled. "
                    "Install with: pip install Pillow --break-system-packages")

if TYPE_CHECKING:
    pass

# Pepper's tablet resolution
TABLET_WIDTH  = 1280
TABLET_HEIGHT = 800

# Unique colour threshold for deciding whether to apply sharpening.
# Images with fewer unique colours than this are treated as logos/diagrams
# and benefit from the sharpen pass. Images above it are photos and don't.
_PHOTO_COLOUR_THRESHOLD = 50_000


def _get_local_ip(peer_ip: str) -> str:
    """
    Discover which local IP address reaches peer_ip.

    Opens a UDP socket (no actual data sent), connects it to peer_ip, and
    reads back the local address the OS chose. Falls back to hostname
    resolution if that fails, and '0.0.0.0' as last resort.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((peer_ip, 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "0.0.0.0"


class _ImageRequestHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves the in-memory image buffer."""

    # Injected by PepperDisplayManager after class creation
    manager: "PepperDisplayManager" = None

    def do_GET(self):
        if self.path not in ("/image", "/image/"):
            self.send_response(404)
            self.end_headers()
            return

        with self.manager._buffer_lock:
            data = self.manager._image_buffer

        if data is None:
            self.send_response(204)   # No Content
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        # No-cache so the tablet always fetches the latest image
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        # Suppress per-request HTTP logs — they're noise in the terminal
        pass


class PepperDisplayManager:
    """
    Manages image display on Pepper's chest tablet.

    Usage:
        manager = PepperDisplayManager(pepper_ip="10.0.0.1", port=8765)
        manager.start()                        # starts HTTP server thread
        manager.show_image("/path/to/img.png") # process + display
        manager.clear_display()                # blank the tablet
        manager.stop()                         # on shutdown
    """

    def __init__(self, pepper_ip: str, port: int = 8765):
        self.pepper_ip  = pepper_ip
        self.port       = port
        self._local_ip  = _get_local_ip(pepper_ip)
        self.image_url  = f"http://{self._local_ip}:{port}/image"

        self._image_buffer: Optional[bytes] = None
        self._buffer_lock   = threading.Lock()
        self._http_server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

        # Injected by main() after PepperRobot is connected
        self._tablet_show_fn  = None   # callable(url: str) → None
        self._tablet_clear_fn = None   # callable() → None

        logging.info("PepperDisplayManager: local IP %s, serving on port %d",
                     self._local_ip, port)

    def set_tablet_fns(self, show_fn, clear_fn):
        """Inject the ALTabletService wrappers from PepperRobot."""
        self._tablet_show_fn  = show_fn
        self._tablet_clear_fn = clear_fn

    def start(self):
        """Start the HTTP image server in a daemon thread."""
        if not _PIL_AVAILABLE:
            logging.warning("PepperDisplayManager.start(): Pillow not available — skipping")
            return

        # Create a handler class with manager injected so it's accessible in do_GET
        handler = type("Handler", (_ImageRequestHandler,), {"manager": self})

        self._http_server = HTTPServer(("0.0.0.0", self.port), handler)
        self._server_thread = threading.Thread(
            target=self._http_server.serve_forever,
            daemon=True,
            name="TabletHTTPServer",
        )
        self._server_thread.start()
        logging.info("Tablet HTTP server started on %s", self.image_url)

    def stop(self):
        """Shut down the HTTP server cleanly."""
        if self._http_server:
            self._http_server.shutdown()
            self._http_server = None

    def show_image(self, path: str, sharpen: bool = False):
        """
        Process and display an image on Pepper's tablet.

        Processing runs in a worker thread so the GUI stays responsive.
        Steps:
          1. Open with Pillow (any format)
          2. Letterbox to 1280×800 (LANCZOS resize, black bars if needed)
          3. Optional sharpening for logos/icons (skipped for photos)
          4. Encode to PNG bytes → swap in-memory buffer
          5. Tell tablet to (re)load the URL — one HTTP request

        Args:
            path:    Absolute path to the image file.
            sharpen: If True AND the image looks like a logo (< 50k unique
                     colours), apply UnsharpMask for extra crispness.
                     For photos this flag is ignored to avoid artefacts.
        """
        if not _PIL_AVAILABLE:
            logging.warning("show_image: Pillow not available")
            return

        threading.Thread(
            target=self._process_and_display,
            args=(path, sharpen),
            daemon=True,
            name="TabletImageWorker",
        ).start()

    def clear_display(self):
        """Clear the tablet display."""
        with self._buffer_lock:
            self._image_buffer = None
        if self._tablet_clear_fn:
            try:
                self._tablet_clear_fn()
            except Exception as e:
                logging.warning("clear_display tablet call failed: %s", e)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_and_display(self, path: str, sharpen: bool):
        """Worker: process image, swap buffer, notify tablet."""
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logging.error("Failed to open image '%s': %s", path, e)
            return

        # ── Letterbox to TABLET_WIDTH × TABLET_HEIGHT ──────────────────
        # Fit the image inside 1280×800 preserving aspect ratio.
        # Black bars are added on the sides/top so Pepper's display is
        # never stretched or cropped.
        img.thumbnail((TABLET_WIDTH, TABLET_HEIGHT), Image.LANCZOS)
        canvas = Image.new("RGB", (TABLET_WIDTH, TABLET_HEIGHT), (0, 0, 0))
        x_off = (TABLET_WIDTH  - img.width)  // 2
        y_off = (TABLET_HEIGHT - img.height) // 2
        canvas.paste(img, (x_off, y_off))
        img = canvas

        # ── Optional sharpening (logos/icons only) ──────────────────────
        if sharpen:
            # Count unique colours to decide if this is a photo.
            # Sampling is cheap: convert to paletted 256-colour image and
            # check — if dithering was needed (many colours → many unique)
            # we treat it as a photo.
            sample = img.copy().quantize(colors=256, dither=Image.Dither.NONE)
            unique = len(set(sample.getdata()))
            is_photo = unique >= 200   # quantized to 256, so ≥200 distinct = very colourful

            if not is_photo:
                img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
                logging.info("Tablet: applied sharpening (logo/diagram detected, %d unique colours after quant)", unique)
            else:
                logging.info("Tablet: skipped sharpening (photo detected, %d unique colours after quant)", unique)

        # ── Encode to PNG bytes ─────────────────────────────────────────
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=False)
        png_bytes = buf.getvalue()

        # ── Swap buffer (atomic) ────────────────────────────────────────
        with self._buffer_lock:
            self._image_buffer = png_bytes

        logging.info("Tablet: image ready (%d bytes, %s)",
                     len(png_bytes), os.path.basename(path))

        # ── Tell tablet to fetch the new image ──────────────────────────
        if self._tablet_show_fn:
            try:
                self._tablet_show_fn(self.image_url)
                logging.info("Tablet: showImage(%s)", self.image_url)
            except Exception as e:
                logging.warning("Tablet showImage failed: %s", e)
        else:
            logging.warning("Tablet: no show_fn set — image buffered but not sent to robot")