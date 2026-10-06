import http.server
import pathlib
import subprocess
import sys
import time
import urllib.parse

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)
        if url.path == "/shot":
            name = "".join(c for c in query.get("name", ["x"])[0] if c.isalnum() or c in "-_")
            r = subprocess.run(["screencapture", "-x", str(out / f"{name}.png")], capture_output=True, text=True)
            print(f"[shot] {name} rc={r.returncode} {r.stderr.strip()}", flush=True)
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
