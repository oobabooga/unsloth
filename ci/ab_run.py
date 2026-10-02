"""Live A/B of NPU downloads in Studio: main vs this branch, each on a fresh Studio home.

env: W (work root), OUT (artifact dir), PY (Studio venv python), PW_PY (Playwright python).
"""

import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

W = Path(os.environ["W"])
OUT = Path(os.environ["OUT"])
PY = os.environ["PY"]
PW_PY = os.environ["PW_PY"]
CI = Path(__file__).resolve().parent
UI_MODEL = "qwen3-4b-FLM"
HUB_MODEL = "llama3.2-1b-FLM"
RESUME_MODEL = "gemma3-1b-FLM"
T0 = time.monotonic()
OUT.mkdir(parents = True, exist_ok = True)
summary: dict = {}


def log(*parts):
    print(f"[{time.monotonic() - T0:7.1f}s]", *parts, flush = True)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Studio:
    def __init__(self, variant):
        self.variant = variant
        self.home = W / f"home-{variant}"
        self.backend = W / variant / "studio" / "backend"
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log_path = OUT / f"studio-{variant}.log"
        self.proc = None
        self.token = None

    def start(self):
        env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(self.home)}
        handle = open(self.log_path, "a")
        self.proc = subprocess.Popen(
            [PY, "run.py", "--host", "127.0.0.1", "--port", str(self.port),
             "--frontend", str(W / f"dist-{self.variant}")],
            cwd = self.backend, env = env, stdout = handle, stderr = subprocess.STDOUT,
            start_new_session = True,
        )
        for _ in range(300):
            try:
                urllib.request.urlopen(self.base + "/api/liveness", timeout = 2)
                return
            except Exception:  # noqa: BLE001
                time.sleep(1)
        raise RuntimeError(f"{self.variant} Studio did not start")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        subprocess.run(["pkill", "-9", "-f", str(self.home)], check = False)

    def mint(self):
        out = subprocess.check_output(
            [PY, str(CI / "mint.py"), str(self.backend), str(self.home), str(self.port)],
            env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(self.home)},
        )
        tokens = json.loads(out)
        self.token = tokens["access_token"]
        path = OUT / f"tokens-{self.variant}.json"
        path.write_text(json.dumps(tokens))
        return path

    def call(self, path, body = None, method = None, timeout = 900):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path, data = data, method = method or ("POST" if data else "GET"),
            headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout = timeout) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode()[:1000]

    def pull(self, model, on_event = None):
        """Follow a download stream; return (events, final event)."""
        req = urllib.request.Request(
            f"{self.base}/api/npu/models/{model}/download", data = b"", method = "POST",
            headers = {"Authorization": f"Bearer {self.token}"},
        )
        events = []
        try:
            with urllib.request.urlopen(req, timeout = 3600) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:])
                    events.append(event)
                    if on_event:
                        on_event(event)
        except Exception as exc:  # noqa: BLE001
            events.append({"event": "client-error", "error": str(exc)})
        return events, (events[-1] if events else None)

    def log_lines(self, pattern):
        text = self.log_path.read_text(errors = "replace")
        return [line[:400] for line in text.splitlines() if re.search(pattern, line)]


def playwright(script, s, tokens, model):
    run = subprocess.run(
        [PW_PY, str(CI / script)],
        env = {**os.environ, "BASE": s.base, "TOKENS": str(tokens), "OUT": str(OUT / "shots"),
               "TAG": s.variant, "MODEL": model, "PYTHONPATH": str(CI)},
        capture_output = True, text = True, timeout = 3600,
    )
    print(run.stdout[-6000:], run.stderr[-4000:], flush = True)
    return run.returncode


def chat(s, model):
    status, body = s.call("/api/inference/load", {"model_path": f"lemonade:{model}"}, timeout = 1200)
    load = {"status": status, "body": json.dumps(body)[:400]}
    status, body = s.call(
        "/v1/chat/completions",
        {"model": f"lemonade:{model}", "max_tokens": 24, "stream": False,
         "messages": [{"role": "user", "content": "Reply with the single word: ready"}]},
        timeout = 600,
    )
    reply = body["choices"][0]["message"]["content"] if status == 200 and isinstance(body, dict) else body
    s.call("/api/inference/unload", {"model_path": f"lemonade:{model}"})
    return {"load": load, "status": status, "reply": reply if isinstance(reply, str) else json.dumps(reply)[:600]}


def files(s):
    root = s.home / "lemonade" / "flm" / "models"
    return {str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()}


def run_variant(variant):
    s = Studio(variant)
    res = summary.setdefault(variant, {})
    log(f"===== {variant}: starting Studio on {s.port}")
    s.start()
    tokens = s.mint()
    status, body = s.call("/api/npu/enable", {}, timeout = 1200)
    log("enable", status, json.dumps(body)[:300])
    res["enable"] = status

    # 1. Download progress in the picker across close, tab switch and reload.
    res["ui_exit"] = playwright("ab_ui.py", s, tokens, UI_MODEL)
    # 2. The Hub's NPU format through Run and a chat reply. The UI flow set the account
    # password, which retires the first token.
    res["hub_exit"] = playwright("ab_hub.py", s, s.mint(), HUB_MODEL)
    s.mint()
    s.call("/api/inference/unload", {"model_path": f"lemonade:{HUB_MODEL}"})

    # 3. The config route for an NPU id.
    start = time.monotonic()
    status, body = s.call(f"/api/models/config/lemonade:{UI_MODEL}")
    res["config"] = {
        "status": status,
        "seconds": round(time.monotonic() - start, 2),
        "hf_warnings": s.log_lines(r"lemonade:.*(Repo id must use|Could not read config|Vision check subprocess failed|Could not get model size)"),
    }
    log("config", json.dumps(res["config"])[:1500])

    # 4. Studio killed mid-download (the desktop window closed), started again, download again.
    killed = {}

    def on_event(event):
        pct = event.get("percent")
        if not killed and event.get("file") == "model.q4nx" and isinstance(pct, (int, float)) and pct >= 40:
            killed["at"] = event
            os.killpg(s.proc.pid, signal.SIGKILL)
            subprocess.run(["pkill", "-9", "-f", str(s.home)], check = False)

    s.pull(RESUME_MODEL, on_event)
    s.proc.wait(60)
    time.sleep(3)
    res["kill"] = {"killed_at": killed.get("at"), "files_after_kill": files(s)}
    log("killed", json.dumps(res["kill"])[:1500])
    s.start()
    tokens = s.mint()
    status, body = s.call("/api/npu/models")
    res["kill"]["catalog_after_restart"] = next(
        (m for m in body["models"] if m["id"] == RESUME_MODEL), None
    ) if status == 200 else body
    started = time.monotonic()
    res["kill"]["ui_exit"] = playwright("ab_resume.py", s, tokens, RESUME_MODEL)
    res["kill"]["seconds_to_finish"] = round(time.monotonic() - started, 1)
    res["kill"]["files_after_second"] = files(s)
    s.mint()
    res["kill"]["chat"] = chat(s, RESUME_MODEL)
    res["kill"]["backend_log"] = s.log_lines(r"NPU model download|Downloading NPU model|Downloaded NPU model|resuming")[-20:]
    res["kill"]["access_log"] = s.log_lines(r"/api/npu/models/.*/download")[-10:]
    log("kill", json.dumps(res["kill"])[:3000])
    s.stop()


def main():
    for variant in ("main", "fix"):
        try:
            run_variant(variant)
        except Exception as exc:  # noqa: BLE001
            log(variant, "FAILED", repr(exc))
            summary.setdefault(variant, {})["error"] = repr(exc)
            subprocess.run(["pkill", "-9", "-f", str(W / f"home-{variant}")], check = False)
        (OUT / "summary.json").write_text(json.dumps(summary, indent = 1))
    print(json.dumps(summary, indent = 1))


if __name__ == "__main__":
    main()
