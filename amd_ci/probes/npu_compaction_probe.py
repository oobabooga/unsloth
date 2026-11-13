#!/usr/bin/env python3
"""Probe: start this checkout's Studio backend, load an NPU (Lemonade / FastFlowLM) model with a small
context, and send chats larger than the window. Records each request's status, reply and the
context_truncated events; never judges. One Studio home is shared across states, so the NPU runtime
and the model download happen once."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WORDS = "sloth canopy river moss leaf branch night slow green rain".split()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _call(base, method, path, body = None, token = None, timeout = 1800):
    req = urllib.request.Request(base + path, method = method,
                                 data = json.dumps(body).encode() if body is not None else None)
    req.add_header("content-type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def _chat(turns: int) -> list[dict]:
    notes = " ".join(f"{WORDS[i % 10]}{i}" for i in range(250))
    msgs = [{"role": "system", "content": "Answer briefly."}]
    for t in range(1, turns + 1):
        msgs.append({"role": "user", "content": f"Turn {t}. Here are some notes: {notes}. Reply with only OK."})
        msgs.append({"role": "assistant", "content": "OK"})
    msgs.append({"role": "user", "content": "Reply with only the word DONE."})
    return msgs


def _summarise(status: int, body: str, stream: bool) -> dict:
    out = {"status": status, "body_head": body[:400]}
    if not stream:
        try:
            d = json.loads(body)
            msg = ((d.get("choices") or [{}])[0].get("message") or {})
            out["content"] = msg.get("content")
            out["error"] = d.get("error") or d.get("detail")
        except ValueError:
            out["error"] = body[:400]
        return out
    content, truncs, errors = "", [], []
    for line in body.splitlines():
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            continue
        try:
            ev = json.loads(line[5:])
        except ValueError:
            continue
        if "context_truncated" in ev:
            truncs.append(ev["context_truncated"])
        if ev.get("error"):
            errors.append(ev["error"])
        for ch in ev.get("choices") or []:
            content += (ch.get("delta") or {}).get("content") or ""
    out.update(content = content, truncations = truncs, error = errors[0] if errors else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--checkpoint", default = "qwen3.5:4b")
    ap.add_argument("--ctx", type = int, default = 2048)
    ap.add_argument("--turns", type = int, default = 6)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": args.checkout}
    home = os.environ.get("UNSLOTH_STUDIO_HOME", "")
    pw_file = Path(home) / ".amd_ci_probe_password"
    first = not pw_file.exists()
    pw = pw_file.read_text(encoding = "utf-8").strip() if not first else secrets.token_urlsafe(18)
    if first:
        pw_file.write_text(pw, encoding = "utf-8")
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ, PYTHONUTF8 = "1", PYTHONIOENCODING = "utf-8")
    if first:
        env["UNSLOTH_STUDIO_PASSWORD"] = pw
    log = open(Path(args.out).with_suffix(".studio.log"), "wb")
    proc = subprocess.Popen([sys.executable, "run.py", "--port", str(port), "--api-only"],
                            cwd = Path(args.checkout) / "studio" / "backend", env = env,
                            stdout = log, stderr = subprocess.STDOUT)
    try:
        for _ in range(600):
            if proc.poll() is not None:
                break
            if _call(base, "GET", "/api/health", timeout = 5)[0] == 200:
                break
            time.sleep(1)
        obs["studio_up"] = _call(base, "GET", "/api/health", timeout = 5)[0] == 200
        if not obs["studio_up"]:
            return 0
        code, body = _call(base, "POST", "/api/auth/login", {"username": "unsloth", "password": pw})
        token = json.loads(body).get("access_token") if code == 200 else None
        obs["login"] = code
        if not token:
            return 0
        if first:
            # A bootstrap login may require a password change before other routes answer.
            _call(base, "POST", "/api/auth/change-password",
                  {"current_password": pw, "new_password": pw}, token)
        code, body = _call(base, "POST", "/api/npu/enable", {}, token)
        obs["enable"] = {"status": code, "body": body[:400]}
        code, body = _call(base, "GET", "/api/npu/models", token = token)
        models = json.loads(body).get("models", []) if code == 200 else []
        obs["catalog"] = [m.get("checkpoint") for m in models][:40]
        model = next((m for m in models if m.get("checkpoint") == args.checkpoint), None)
        if model is None:
            obs["model_missing"] = args.checkpoint
            return 0
        code, body = _call(base, "POST", f"/api/npu/models/{model['id']}/download", {}, token)
        obs["download"] = {"status": code, "tail": body[-300:]}
        model_id = "lemonade:" + model["id"]
        code, body = _call(base, "POST", "/api/inference/load",
                           {"model_path": model_id, "max_seq_length": args.ctx}, token)
        loaded = json.loads(body) if code == 200 else {}
        obs["load"] = {"status": code, "context_length": loaded.get("context_length"),
                       "is_npu": loaded.get("is_npu"), "body": body[:300] if code != 200 else None}
        if code != 200:
            return 0

        def chat(name, messages, stream, overflow):
            payload = {"model": model_id, "messages": messages, "stream": stream, "max_tokens": 32,
                       "enable_thinking": False}
            if overflow is not None:
                payload["context_overflow"] = overflow
            t0 = time.time()
            code, body = _call(base, "POST", "/api/inference/chat/completions", payload, token)
            r = _summarise(code, body, stream)
            r["seconds"] = round(time.time() - t0, 1)
            r["messages_sent"] = len(messages)
            obs.setdefault("requests", {})[name] = r

        chat("control_short", _chat(1), False, "truncate_oldest")
        chat("over_window_auto_compact", _chat(args.turns), False, "truncate_oldest")
        chat("over_window_auto_compact_stream", _chat(args.turns), True, "truncate_oldest")
        chat("over_window_auto_compact_off", _chat(args.turns), True, "error")
        _call(base, "POST", "/api/inference/unload", {"model_path": model_id}, token)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout = 30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        Path(args.out).write_text(json.dumps(obs, indent = 1), encoding = "utf-8")


if __name__ == "__main__":
    sys.exit(main())
