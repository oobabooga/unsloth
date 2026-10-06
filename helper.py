import ctypes
import http.server
import pathlib
import subprocess
import sys
import time
import urllib.parse

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)


def windows_shot(path):
    """The Unsloth window's rectangle, grabbed from the composited screen (so the native page is in it)."""
    import ctypes
    from ctypes import wintypes

    from PIL import ImageGrab

    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        length = user32.GetWindowTextLengthW(hwnd)
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title, length + 1)
        if user32.IsWindowVisible(hwnd) and title.value == "Unsloth":
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            found.append((rect.left, rect.top, rect.right, rect.bottom))
        return True

    user32.EnumWindows(visit, 0)
    box = max(found, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), default=None)
    if box:
        user32.SetForegroundWindow(user32.FindWindowW(None, "Unsloth"))
    ImageGrab.grab(bbox=box, all_screens=True).save(path)
    return f"window={box or 'screen'}"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)
        if url.path == "/shot":
            name = "".join(c for c in query.get("name", ["x"])[0] if c.isalnum() or c in "-_")
            if sys.platform == "win32":
                print(f"[shot] {name} {windows_shot(out / f'{name}.png')}", flush=True)
            else:
                # The app window alone, so system prompts over the desktop stay out of the picture.
                window = subprocess.run([sys.argv[2]], capture_output=True, text=True).stdout.strip()
                target = ["-o", f"-l{window}"] if window else []
                r = subprocess.run(["screencapture", "-x", *target, str(out / f"{name}.png")], capture_output=True, text=True)
                print(f"[shot] {name} window={window or 'screen'} rc={r.returncode} {r.stderr.strip()}", flush=True)
        elif url.path == "/log":
            print(f"[log] {time.strftime('%H:%M:%S')} {query.get('m', [''])[0]}", flush=True)
        elif url.path == "/done":
            (out / "DONE").write_text("ok")
            print("[done]", flush=True)
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


http.server.ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
