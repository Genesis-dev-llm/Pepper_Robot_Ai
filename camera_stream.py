#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pepper Camera → Tablet MJPEG Server
Runs ON Pepper's head CPU (Python 2.7).

Architecture:
  ALVideoDevice (127.0.0.1:9559)
    └─ capture_loop [background thread, 15fps]
         └─ RGB bytes → PIL JPEG → _frame_buffer
  ThreadedHTTPServer on 0.0.0.0:8080
    ├─ GET /stream.html  → fullscreen HTML page pointing at /stream.mjpeg
    └─ GET /stream.mjpeg → multipart MJPEG response (streams frames)

The tablet accesses this over the internal USB link at 198.18.0.1:8080.
No WiFi hop — USB 2.0 bandwidth is far more than enough for QVGA MJPEG.

Usage (called via SSH from the laptop):
  python /home/nao/camera_stream.py

Kill:
  pkill -f camera_stream.py
"""

from __future__ import print_function

import sys
import time
import threading
import SocketServer
import BaseHTTPServer
from io import BytesIO

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    print("[camera_stream] WARNING: PIL not available — stream will produce no frames")

# ── NAOqi ──────────────────────────────────────────────────────────────────────

try:
    from naoqi import ALProxy
    _NAOQI_AVAILABLE = True
except ImportError:
    _NAOQI_AVAILABLE = False
    print("[camera_stream] ERROR: naoqi not available — exiting")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

PEPPER_IP      = "127.0.0.1"
PEPPER_PORT    = 9559
SERVER_PORT    = 8080

CAMERA_TOP     = 0   # front/top camera
RESOLUTION     = 1   # QVGA 320x240
COLOR_SPACE    = 11  # RGB
FPS            = 15

IMG_WIDTH      = 320
IMG_HEIGHT     = 240

BOUNDARY       = "mjpegboundary"

# ── Shared frame buffer ────────────────────────────────────────────────────────

_frame_lock   = threading.Lock()
_frame_jpeg   = None   # bytes: latest JPEG frame
_frame_event  = threading.Event()  # set whenever a new frame is ready

# ── HTML page served at /stream.html ─────────────────────────────────────────

_HTML = b"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100vw;height:100vh;overflow:hidden;background:#000;
  display:flex;align-items:center;justify-content:center}
img{max-width:100vw;max-height:100vh;display:block;object-fit:contain}
</style>
</head>
<body>
<img src="/stream.mjpeg" alt="Camera">
</body>
</html>"""

# ── HTTP handler ──────────────────────────────────────────────────────────────

class StreamHandler(BaseHTTPServer.BaseHTTPRequestHandler):

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/stream.html":
            self._serve_html()
        elif path == "/stream.mjpeg":
            self._serve_mjpeg()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(_HTML)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(_HTML)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=" + BOUNDARY
        )
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            while True:
                # Block until a new frame is available (max 1s timeout)
                _frame_event.wait(timeout=1.0)
                _frame_event.clear()

                with _frame_lock:
                    jpeg = _frame_jpeg

                if jpeg is None:
                    continue

                # Write MJPEG boundary + frame
                header = (
                    "--" + BOUNDARY + "\r\n"
                    "Content-Type: image/jpeg\r\n"
                    "Content-Length: " + str(len(jpeg)) + "\r\n"
                    "\r\n"
                )
                self.wfile.write(header.encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

        except Exception:
            # Client disconnected — normal, not an error
            pass

    def log_message(self, fmt, *args):
        # Suppress per-request logs — noisy on a 15fps stream
        pass


class ThreadedHTTPServer(SocketServer.ThreadingMixIn, BaseHTTPServer.HTTPServer):
    daemon_threads = True


# ── Camera capture loop ────────────────────────────────────────────────────────

def capture_loop():
    global _frame_jpeg

    video = ALProxy("ALVideoDevice", PEPPER_IP, PEPPER_PORT)
    client_name = video.subscribeCamera(
        "CameraStream",
        CAMERA_TOP,
        RESOLUTION,
        COLOR_SPACE,
        FPS,
    )
    print("[camera_stream] Camera subscribed ({0}x{1} @ {2}fps)".format(
        IMG_WIDTH, IMG_HEIGHT, FPS))

    interval = 1.0 / FPS

    try:
        while True:
            t0 = time.time()

            image = video.getImageRemote(client_name)
            if image is not None:
                raw = image[6]  # raw RGB bytes (str in Python 2)
                jpeg = _encode_jpeg(raw)
                if jpeg is not None:
                    with _frame_lock:
                        _frame_jpeg = jpeg
                    _frame_event.set()

            elapsed = time.time() - t0
            sleep_for = max(0.0, interval - elapsed)
            time.sleep(sleep_for)

    except Exception as e:
        print("[camera_stream] Capture loop error: {0}".format(e))
    finally:
        try:
            video.unsubscribe(client_name)
        except Exception:
            pass
        print("[camera_stream] Camera unsubscribed")


def _encode_jpeg(raw_rgb):
    """
    Encode raw RGB bytes (320x240x3) to JPEG bytes.

    PIL path: Image.frombytes (Pillow) or Image.fromstring (old PIL).
    Fallback: use NAOqi's own JPEG encoding via ALVideoDevice.
    Returns bytes or None on failure.
    """
    if _PIL_AVAILABLE:
        try:
            try:
                # Pillow >= 2.0
                img = Image.frombytes("RGB", (IMG_WIDTH, IMG_HEIGHT), bytes(raw_rgb))
            except AttributeError:
                # Old PIL
                img = Image.fromstring("RGB", (IMG_WIDTH, IMG_HEIGHT), str(raw_rgb))  # noqa
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=70)
            return buf.getvalue()
        except Exception as e:
            print("[camera_stream] PIL encode error: {0}".format(e))
            return None
    else:
        # PIL not available — can't encode, return None
        return None


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[camera_stream] Starting camera capture thread…")
    t = threading.Thread(target=capture_loop)
    t.daemon = True
    t.start()

    # Give capture loop a moment to subscribe before the server starts
    time.sleep(0.5)

    print("[camera_stream] Starting HTTP server on port {0}…".format(SERVER_PORT))
    server = ThreadedHTTPServer(("0.0.0.0", SERVER_PORT), StreamHandler)

    try:
        print("[camera_stream] Serving. Tablet URL: http://198.18.0.1:{0}/stream.html".format(
            SERVER_PORT))
        server.serve_forever()
    except KeyboardInterrupt:
        print("[camera_stream] Shutting down")
        server.shutdown()