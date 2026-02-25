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

Static images (PNG, JPG, BMP, WebP) are letter-boxed to 1280×800, encoded to
PNG in memory, and served at /image.  The tablet fetches them via showImage().

Animated GIFs are served as raw file bytes at /gif.  The tablet loads an HTML
wrapper page at /gifpage (via showWebview) that uses CSS to centre and scale
the animation.  This avoids per-frame re-encoding and keeps file sizes small.

ALTabletService uses HTTP — the tablet's Android browser fetches the content
from the control PC over the local network.
"""

import logging
import os
import time
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

# HTML template for animated GIF display — CSS handles scaling/centering
# so we never need to re-encode frames.  The timestamp query param on the
# <img> src forces the WebView to re-fetch when a new GIF is loaded.
_GIF_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: 100vw; height: 100vh; overflow: hidden;
               background: #000;
               display: flex; align-items: center; justify-content: center; }}
  img {{ max-width: 100vw; max-height: 100vh;
        display: block;
        object-fit: contain; }}
</style>
</head>
<body>
<img src="/gif?t={cache_bust}" alt="">
</body>
</html>"""


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
    """Minimal HTTP handler that serves images and GIF HTML pages."""

    # Injected by PepperDisplayManager after class creation
    manager: "PepperDisplayManager" = None

    def do_GET(self):
        # Strip query string (cache-buster) before matching
        clean_path = self.path.split("?")[0].rstrip("/")

        if clean_path == "/image":
            self._serve_image_buffer()
        elif clean_path == "/gif":
            self._serve_gif_buffer()
        elif clean_path == "/gifpage":
            self._serve_gif_html()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_image_buffer(self):
        """Serve the processed PNG image buffer."""
        with self.manager._buffer_lock:
            data = self.manager._image_buffer

        if data is None:
            self.send_response(204)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_gif_buffer(self):
        """Serve the raw animated GIF bytes."""
        with self.manager._buffer_lock:
            data = self.manager._gif_buffer

        if data is None:
            self.send_response(204)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/gif")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            pass  # tablet disconnected mid-download (large GIF)

    def _serve_gif_html(self):
        """Serve the HTML wrapper page that displays the GIF."""
        cache_bust = int(time.time() * 1000)
        html = _GIF_HTML_TEMPLATE.format(cache_bust=cache_bust).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html)

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
        self._base_url  = f"http://{self._local_ip}:{port}"

        self._image_buffer: Optional[bytes] = None
        self._gif_buffer:   Optional[bytes] = None
        self._buffer_lock   = threading.Lock()
        self._http_server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

        # Injected by main() after PepperRobot is connected
        self._tablet_show_fn    = None   # callable(url: str) → None  (showImage)
        self._tablet_webview_fn = None   # callable(url: str) → None  (showWebview)
        self._tablet_clear_fn   = None   # callable() → None

        logging.info("PepperDisplayManager: local IP %s, serving on port %d",
                     self._local_ip, port)

    def set_tablet_fns(self, show_fn, clear_fn, webview_fn=None):
        """Inject the ALTabletService wrappers from PepperRobot."""
        self._tablet_show_fn    = show_fn
        self._tablet_webview_fn = webview_fn
        self._tablet_clear_fn   = clear_fn

    def start(self):
        """Start the HTTP image server in a daemon thread."""
        if not _PIL_AVAILABLE:
            logging.warning("PepperDisplayManager.start(): Pillow not available — skipping")
            return

        handler = type("Handler", (_ImageRequestHandler,), {"manager": self})

        self._http_server = HTTPServer(("0.0.0.0", self.port), handler)
        self._server_thread = threading.Thread(
            target=self._http_server.serve_forever,
            daemon=True,
            name="TabletHTTPServer",
        )
        self._server_thread.start()
        logging.info("Tablet HTTP server started on %s", self._base_url)

    def stop(self):
        """Shut down the HTTP server cleanly."""
        if self._http_server:
            self._http_server.shutdown()
            self._http_server = None

    def show_image(self, path: str, sharpen: bool = False):
        """
        Process and display an image on Pepper's tablet.

        Static images (PNG, JPG, BMP, WebP):
          Letterboxed to 1280×800, encoded as PNG, served at /image,
          displayed via showImage().

        Animated GIFs:
          Served as raw file bytes at /gif, displayed via showWebview()
          loading an HTML wrapper at /gifpage.  CSS handles scaling.

        Args:
            path:    Absolute path to the image file.
            sharpen: If True AND the image looks like a logo (< 50k unique
                     colours), apply UnsharpMask for extra crispness.
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
            self._gif_buffer   = None
        if self._tablet_clear_fn:
            try:
                self._tablet_clear_fn()
            except Exception as e:
                logging.warning("clear_display tablet call failed: %s", e)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _is_animated_gif(path: str) -> bool:
        """Return True if the file is an animated GIF with more than one frame."""
        try:
            img = Image.open(path)
            try:
                img.seek(1)
                return True
            except EOFError:
                return False
            finally:
                img.close()
        except Exception:
            return False

    def _process_and_display(self, path: str, sharpen: bool):
        """Worker: process image, swap buffer, notify tablet."""

        ext = os.path.splitext(path)[1].lower()

        # ── Animated GIF path ──────────────────────────────────────────
        # Serve the original file bytes directly — no re-encoding.
        # The tablet's Android WebView handles GIF animation natively.
        if ext == ".gif" and self._is_animated_gif(path):
            try:
                with open(path, "rb") as f:
                    gif_bytes = f.read()
            except Exception as e:
                logging.error("Failed to read GIF '%s': %s", path, e)
                return

            with self._buffer_lock:
                self._gif_buffer = gif_bytes

            file_size_kb = len(gif_bytes) / 1024
            logging.info("Tablet: animated GIF ready (%.0f KB, %s)",
                         file_size_kb, os.path.basename(path))

            # Use showWebview with the HTML wrapper page
            cache_bust = int(time.time() * 1000)
            webview_url = f"{self._base_url}/gifpage?t={cache_bust}"

            if self._tablet_webview_fn:
                try:
                    self._tablet_webview_fn(webview_url)
                    logging.info("Tablet: showWebview(%s)", webview_url)
                except Exception as e:
                    logging.warning("Tablet showWebview failed: %s", e)
            elif self._tablet_show_fn:
                # Fallback: try showImage with direct GIF URL
                gif_url = f"{self._base_url}/gif?t={cache_bust}"
                try:
                    self._tablet_show_fn(gif_url)
                    logging.info("Tablet: showImage fallback (%s)", gif_url)
                except Exception as e:
                    logging.warning("Tablet showImage(gif) failed: %s", e)
            else:
                logging.warning("Tablet: no show_fn set — GIF buffered but not sent")
            return

        # ── Static image path (PNG, JPG, BMP, WebP, static GIF) ───────
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logging.error("Failed to open image '%s': %s", path, e)
            return

        # Letterbox to TABLET_WIDTH × TABLET_HEIGHT
        img.thumbnail((TABLET_WIDTH, TABLET_HEIGHT), Image.LANCZOS)
        canvas = Image.new("RGB", (TABLET_WIDTH, TABLET_HEIGHT), (0, 0, 0))
        x_off = (TABLET_WIDTH  - img.width)  // 2
        y_off = (TABLET_HEIGHT - img.height) // 2
        canvas.paste(img, (x_off, y_off))
        img = canvas

        # Optional sharpening (logos/icons only)
        if sharpen:
            sample = img.copy().quantize(colors=256, dither=Image.Dither.NONE)
            unique = len(set(sample.getdata()))
            is_photo = unique >= 200

            if not is_photo:
                img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
                logging.info("Tablet: applied sharpening (%d unique colours after quant)", unique)
            else:
                logging.info("Tablet: skipped sharpening (photo, %d unique colours after quant)", unique)

        # Encode to PNG bytes
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=False)
        png_bytes = buf.getvalue()

        # Swap buffer
        with self._buffer_lock:
            self._image_buffer = png_bytes

        logging.info("Tablet: image ready (%d bytes, %s)",
                     len(png_bytes), os.path.basename(path))

        # Tell tablet to fetch the new image
        if self._tablet_show_fn:
            try:
                cache_bust_url = f"{self.image_url}?t={int(time.time() * 1000)}"
                self._tablet_show_fn(cache_bust_url)
                logging.info("Tablet: showImage(%s)", cache_bust_url)
            except Exception as e:
                logging.warning("Tablet showImage failed: %s", e)
        else:
            logging.warning("Tablet: no show_fn set — image buffered but not sent")