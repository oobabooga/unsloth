#!/usr/bin/env python3
"""Real-NPU probe for PR 17530: launch this state's Studio on the shared install, then record
(1) Lemonade telemetry for an API chat with no tools, with tools, and with tools + tool_choice none,
(2) the Search / Code pills a fresh browser profile shows with an NPU model loaded, and the chat
request the composer sends. Observes only; npu_tools_criteria.py judges."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = os.environ.get("NPU_MODEL", "qwen3.8-dflash2:27b")
PROMPT = "Explain how a bicycle stays upright while moving, in about 150 words."
TOOLS = [
    {"type": "function", "function": {"name": "web_search", "description": "Search the web.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "python", "description": "Run Python code and return stdout.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]
HOME = Path(os.environ["UNSLOTH_STUDIO_HOME"])
PW_FILE = HOME.parent / "npu17530_password"
TELE_RE = re.compile(r"Inference completed: model=(\S+), tokens=(\d+) \(in=(\d+), out=(\d+)\), "
                     r"ttft=([\d.]+)s, tps=([\d.]+)")


def log(*a):
    print(*a, flush = True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Studio:
    def __init__(self, checkout: Path, python: str, out_dir: Path, state: str):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        first = not PW_FILE.exists()
        if first:
            PW_FILE.write_text(secrets.token_urlsafe(18), encoding = "utf-8")
        self.pw = PW_FILE.read_text(encoding = "utf-8").strip()
        dist = checkout / "studio" / "frontend" / "dist"
        # Studio refuses a supplied password once one is set, so only the first launch passes it.
        env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_STUDIO_PASSWORD"}
        if first:
            env["UNSLOTH_STUDIO_PASSWORD"] = self.pw
        self.log_path = out_dir / f"studio_{state}.log"
        self.proc = subprocess.Popen(
            [python, "run.py", "--port", str(self.port), "--frontend", str(dist)],
            cwd = checkout / "studio" / "backend", env = env,
            stdout = open(self.log_path, "wb"), stderr = subprocess.STDOUT, preexec_fn = os.setsid)
        for _ in range(600):
            if self.proc.poll() is not None:
                raise RuntimeError(f"Studio exited {self.proc.returncode}")
            if self.call("GET", "/api/health")[0] == 200:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Studio never became healthy")
        self.tok = None
        self.login()

    def call(self, method, path, body = None, timeout = 3000, auth = True):
        req = urllib.request.Request(self.base + path, method = method,
                                     data = json.dumps(body).encode() if body is not None else None)
        req.add_header("content-type", "application/json")
        if auth and getattr(self, "tok", None):
            req.add_header("Authorization", "Bearer " + self.tok)
        try:
            with urllib.request.urlopen(req, timeout = timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            return 0, repr(e)

    def login(self):
        c, b = self.call("POST", "/api/auth/login", {"username": "unsloth", "password": self.pw}, auth = False)
        data = json.loads(b)
        self.tok, self.refresh = data["access_token"], data.get("refresh_token", "")

    def stop(self):
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout = 30)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass


def telemetry() -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(HOME / "logs" / "lemonade" / "lemonade-*.log"))):
        for line in open(f, encoding = "utf-8", errors = "replace"):
            m = TELE_RE.search(line)
            if m:
                rows.append({"in": int(m.group(3)), "out": int(m.group(4)),
                             "ttft": float(m.group(5)), "tps": float(m.group(6))})
    return rows


def chat(st: Studio, mid: str, max_tokens: int, tools = None, tool_choice = None) -> dict:
    n0 = len(telemetry())
    body = {"model": "lemonade:" + mid, "max_tokens": max_tokens, "stream": False, "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT}]}
    if tools:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    t0 = time.time()
    c, b = st.call("POST", "/v1/chat/completions", body, timeout = 1800)
    wall = round(time.time() - t0, 2)
    time.sleep(1)
    new = telemetry()[n0:]
    return {"status": c, "wall": wall, "tele": new[-1] if new else None,
            "error": None if c == 200 else b[:300]}


def ensure_dist(checkout: Path) -> str:
    fe = checkout / "studio" / "frontend"
    if (fe / "dist" / "index.html").is_file():
        return "present"
    npm = shutil.which("npm")
    if npm is None:
        found = sorted(glob.glob(str(HOME / "**" / "bin" / "npm"), recursive = True))
        npm = found[0] if found else None
    if npm is None:
        raise RuntimeError("no npm to build this state's frontend")
    env = {**os.environ, "PATH": str(Path(npm).parent) + os.pathsep + os.environ.get("PATH", "")}
    for cmd in ([npm, "ci", "--no-audit", "--no-fund"], [npm, "run", "build"]):
        r = subprocess.run(cmd, cwd = fe, env = env, capture_output = True, text = True)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd[1:])} failed: {r.stdout[-800:]}{r.stderr[-800:]}")
    return "built"


def ui(st: Studio, out_dir: Path, state: str, pw_python: str) -> dict:
    """Fresh profile, NPU model already loaded: which pills light up, and what the composer sends."""
    script = out_dir / "ui_driver.py"
    script.write_text(UI_DRIVER, encoding = "utf-8")
    res = out_dir / f"ui_{state}.json"
    r = subprocess.run([pw_python, str(script), st.base, st.tok, st.refresh, str(out_dir), state, str(res)],
                       capture_output = True, text = True, timeout = 1500)
    (out_dir / f"ui_{state}.log").write_text(r.stdout + r.stderr, encoding = "utf-8")
    if res.is_file():
        return json.loads(res.read_text(encoding = "utf-8"))
    return {"error": f"ui driver rc={r.returncode}: {(r.stdout + r.stderr)[-600:]}"}


UI_DRIVER = r'''
import json, sys, time
from playwright.sync_api import sync_playwright
base, tok, refresh, out_dir, state, res = sys.argv[1:]
out = {"requests": []}
seed = json.dumps({"unsloth_auth_token": tok, "unsloth_auth_refresh_token": refresh})
with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport = {"width": 1440, "height": 900})
    ctx.add_init_script("const s=" + seed + "; for (const k in s) { try { localStorage.setItem(k, s[k]); } catch (e) {} }")
    page = ctx.new_page()
    def on_req(r):
        if r.method == "POST" and r.post_data and '"messages"' in r.post_data:
            try:
                d = json.loads(r.post_data)
            except Exception:
                d = {}
            out["requests"].append({"url": r.url, "enable_tools": d.get("enable_tools"),
                                    "enabled_tools": d.get("enabled_tools"), "has_tools": bool(d.get("tools")),
                                    "model": d.get("model")})
    page.on("request", on_req)
    page.goto(base + "/chat", wait_until = "networkidle")
    search = page.locator('[aria-label="Disable web search"], [aria-label="Enable web search"]').first
    search.wait_for(timeout = 120000)
    time.sleep(5)
    out["search_label"] = search.get_attribute("aria-label")
    out["search_active"] = search.get_attribute("data-active")
    out["store_hint"] = page.evaluate("Object.fromEntries(Object.entries(localStorage).filter(([k]) => /tools_enabled/.test(k)))")
    page.screenshot(path = f"{out_dir}/ui_{state}_composer.png")
    box = page.locator("textarea").first
    box.fill("What is 2 + 2? Reply with one word.")
    box.press("Enter")
    for _ in range(60):
        if out["requests"]:
            break
        time.sleep(1)
    time.sleep(20)
    page.screenshot(path = f"{out_dir}/ui_{state}_after_send.png")
    b.close()
open(res, "w").write(json.dumps(out))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--pw-python", default = os.environ.get("NPU_PW_PYTHON", ""))
    args = ap.parse_args()
    out_dir = args.out.parent
    obs: dict = {"model": MODEL, "checkout": str(args.checkout)}
    src = (args.checkout / "studio/backend/core/inference/external_provider.py").read_text(encoding = "utf-8")
    obs["tree_has_backend_fix"] = 'self.provider_type == "lemonade" and tool_choice_disabled' in src
    st = None
    try:
        obs["dist"] = ensure_dist(args.checkout)
        st = Studio(args.checkout, sys.executable, out_dir, args.state)
        log("enable", st.call("POST", "/api/npu/enable", {})[0])
        status = json.loads(st.call("GET", "/api/npu/status")[1])
        obs["npu_status"] = {k: status.get(k) for k in ("versions", "hardware", "ready", "enabled")}
        models = {m["checkpoint"]: m for m in json.loads(st.call("GET", "/api/npu/models")[1])["models"]}
        m = models.get(MODEL)
        if m is None:
            obs["error"] = f"{MODEL} not in catalog: {sorted(models)[:30]}"
            return 0
        mid = m["id"]
        obs["model_id"] = mid
        t0 = time.time()
        c, b = st.call("POST", f"/api/npu/models/{mid}/download", {})
        obs["download"] = {"status": c, "secs": round(time.time() - t0), "complete": '"complete"' in b[-400:]}
        for ctx in (16384, 8192, 4096):
            c, b = st.call("POST", "/api/inference/load",
                           {"model_path": "lemonade:" + mid, "max_seq_length": ctx, "force_reload": True})
            d = json.loads(b) if b.startswith("{") else {}
            obs["load"] = {"status": c, "is_npu": d.get("is_npu"), "supports_tools": d.get("supports_tools"),
                           "ctx": ctx, "error": d.get("_deferred_error") or (None if c == 200 else b[:300])}
            if c != 200 or not d.get("is_npu"):
                return 0
            warm = chat(st, mid, 16)
            obs.setdefault("warmups", []).append({"ctx": ctx, **warm})
            if warm["status"] == 200:
                break
        variants = {"no tools": {}, "tools": {"tools": TOOLS},
                    "tools + tool_choice none": {"tools": TOOLS, "tool_choice": "none"}}
        obs["api"] = {v: [] for v in variants}
        for rep in range(args.reps):
            for v, kw in variants.items():
                st.login()
                r = chat(st, mid, 128, **kw)
                log("ROW", args.state, v, rep, json.dumps(r))
                obs["api"][v].append(r)
                args.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
        if args.pw_python:
            st.login()
            try:
                obs["ui"] = ui(st, out_dir, args.state, args.pw_python)
            except Exception as e:  # noqa: BLE001
                obs["ui"] = {"error": repr(e)}
        st.call("POST", "/api/inference/unload", {"model_path": "lemonade:" + mid})
    except Exception as e:  # noqa: BLE001
        obs["error"] = repr(e)
    finally:
        if st is not None:
            st.stop()
        for f in glob.glob(str(HOME / "logs" / "lemonade" / "*.log")):
            shutil.copy(f, out_dir / f"{args.state}_{Path(f).name}")
        args.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
