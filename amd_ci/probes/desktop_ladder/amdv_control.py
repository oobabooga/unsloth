#!/usr/bin/env python3
"""The control channel between the harness and a page inside the Unsloth DESKTOP webview.

Why this exists at all. In the WEB UI ladder the harness OWNED the browser: it was a PyGObject
WebKitGTK window, so it could inject a document-start user script, evaluate JavaScript after
load, and receive results through `window.webkit.messageHandlers.bench`. None of that is
available for Desktop. The Tauri app loads its BUNDLED frontend at `tauri://localhost` from
`frontendDist`, so there is no URL to point anywhere, no `?thread=` query string, no external
eval channel and no message handler of ours.

What there IS: the app's own CSP (studio/src-tauri/tauri.conf.json, `app.security.csp`) allows
`connect-src ... http://localhost:* http://127.0.0.1:*`. So the page can talk to a loopback
HTTP server, and that is the whole mechanism. The harness serves a CONFIG the page polls for,
and the page POSTs back its events and its final payload.

This deliberately does not drive the app through an automation protocol. `tauri-driver` +
`WebKitWebDriver` exist and the repo uses them in tests/studio/appimage_model_download_webdriver.py,
but a WebDriver session is a second consumer of the main thread and its `execute/sync` round
trips land in exactly the phases being measured. The instrument would then be part of the load.
A poll every 250 ms against a loopback socket is not free either, and it is recorded and held
identical across every rung so it cannot produce a rung-dependent difference.

CORS matters and is not optional: the document origin is `tauri://localhost`, which is not the
server's origin, so a POST of `application/json` is preflighted. The repo's own WebDriver
fixture answers `Access-Control-Allow-Origin: tauri://localhost` for the same reason
(tests/studio/appimage_model_download_webdriver.py). This echoes the request Origin rather than
hard-coding one, so it works for a dev-server run too, and it binds 127.0.0.1 only.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Store:
    def __init__(self, events_path: Path, result_path: Path):
        self.lock = threading.Lock()
        self.config: dict = {"ready": False}
        self.result: dict | None = None
        self.events_path = events_path
        self.result_path = result_path
        self.event_count = 0
        self.first_contact: float | None = None
        self.last_contact: float | None = None

    def set_config(self, cfg: dict) -> None:
        with self.lock:
            self.config = cfg

    def note_contact(self) -> None:
        with self.lock:
            now = time.time()
            if self.first_contact is None:
                self.first_contact = now
            self.last_contact = now

    def add_event(self, obj) -> None:
        with self.lock:
            self.event_count += 1
            with open(self.events_path, "a", encoding = "utf-8") as fh:
                fh.write(json.dumps({"t": time.time(), "v": obj}) + "\n")

    def set_result(self, obj) -> None:
        with self.lock:
            self.result = obj
            self.result_path.write_text(json.dumps(obj))


def make_handler(store: Store):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # noqa: A003
            # Silence. Every request is a poll and the access log would be the largest file in
            # the artifact without carrying anything the events file does not.
            return

        def _cors(self):
            origin = self.headers.get("Origin") or "*"
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Cache-Control", "no-store")

        def _send(self, code: int, body: bytes, ctype = "application/json"):
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # noqa: N802
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            store.note_contact()
            if self.path.startswith("/amdv/config"):
                with store.lock:
                    body = json.dumps(store.config).encode()
                self._send(200, body)
            elif self.path.startswith("/amdv/status"):
                with store.lock:
                    body = json.dumps({
                        "events": store.event_count,
                        "have_result": store.result is not None,
                        "first_contact": store.first_contact,
                        "last_contact": store.last_contact,
                    }).encode()
                self._send(200, body)
            else:
                self._send(404, b'{"error":"no"}')

        def do_POST(self):  # noqa: N802
            store.note_contact()
            try:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                obj = json.loads(raw.decode("utf-8", "replace")) if raw else None
            except Exception as e:  # noqa: BLE001
                self._send(400, json.dumps({"error": repr(e)}).encode())
                return
            if self.path.startswith("/amdv/result"):
                store.set_result(obj)
                self._send(200, b'{"ok":true}')
            elif self.path.startswith("/amdv/event"):
                store.add_event(obj)
                self._send(200, b'{"ok":true}')
            else:
                self._send(404, b'{"error":"no"}')

    return H


class Control:
    """Start/stop helper for use from a probe, rather than as a separate process."""

    def __init__(self, port: int, out_dir: Path):
        out_dir.mkdir(parents = True, exist_ok = True)
        self.port = port
        self.store = Store(out_dir / "page_events.jsonl", out_dir / "page_result.json")
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(self.store))
        self.thread = threading.Thread(target = self.httpd.serve_forever, daemon = True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:  # noqa: BLE001
            pass

    def wait_for_result(self, timeout: float) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.store.lock:
                if self.store.result is not None:
                    return self.store.result
            time.sleep(1.0)
        return None

    def wait_for_contact(self, timeout: float) -> bool:
        """Did the PAGE ever reach us? Distinguishes 'the app never painted our script' from
        'the scene ran and failed', which otherwise both present as a missing result."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.store.lock:
                if self.store.first_contact is not None:
                    return True
            time.sleep(0.5)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type = int, required = True)
    ap.add_argument("--out-dir", required = True, type = Path)
    ap.add_argument("--seconds", type = float, default = 600)
    args = ap.parse_args()
    c = Control(args.port, args.out_dir).start()
    print(json.dumps({"status": "READY", "port": args.port}), flush = True)
    try:
        time.sleep(args.seconds)
    finally:
        c.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
