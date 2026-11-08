#!/usr/bin/env python3
"""Probe: install Studio from this checkout on Windows, load MOSS-Image-SFT Q8_0 GGUF, render once.

Observes only: install / launch / load / generate outcomes, the engine Studio picked, the error text and the
sd-server lines from the Studio log. Writes JSON to --out.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _req(url, body = None, token = None, timeout = 60, method = None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data = data, method = method or ("POST" if body is not None else "GET"))
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _json(raw):
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return {}


def _tail(text, n = 4000):
    return text[-n:] if text else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--size", type = int, default = 1024)
    ap.add_argument("--steps", type = int, default = 8)
    args = ap.parse_args()

    work = Path(os.environ.get("AMD_CI_WORK") or os.environ.get("RUNNER_TEMP") or ".")
    home = work / f"studio_{args.state}"
    hf = work / "hf_home"
    hf.mkdir(parents = True, exist_ok = True)
    port = 18900 + (abs(hash(args.state)) % 50)
    base = f"http://127.0.0.1:{port}"
    obs: dict = {"state": args.state, "checkout": str(args.checkout), "home": str(home)}
    env = dict(os.environ, UNSLOTH_STUDIO_HOME = str(home), UNSLOTH_SKIP_AUTOSTART = "1", HF_HOME = str(hf),
               PYTHONIOENCODING = "utf-8")
    try:
        obs["rev"] = subprocess.run(["git", "-C", str(args.checkout), "log", "--oneline", "-1"], capture_output = True,
                                    text = True, encoding = "utf-8").stdout.strip()
    except Exception as e:  # noqa: BLE001
        obs["rev"] = f"unreadable: {e}"

    t0 = time.time()
    p = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(args.checkout / "install.ps1"), "--local", "--skip-autostart"],
                       cwd = str(args.checkout), env = env, stdin = subprocess.DEVNULL, capture_output = True,
                       text = True, encoding = "utf-8", errors = "replace", timeout = 3600)
    obs["install_rc"] = p.returncode
    obs["install_secs"] = round(time.time() - t0, 1)
    obs["install_tail"] = _tail(p.stdout + p.stderr, 3000)
    if p.returncode != 0:
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    exe = None
    for pat in ("bin/unsloth.exe", "bin/unsloth.cmd", "bin/unsloth.bat", "unsloth_studio/Scripts/unsloth.exe"):
        hits = glob.glob(str(home / pat))
        if hits:
            exe = hits[0]
            break
    obs["launcher"] = exe
    if not exe:
        obs["launch_error"] = "no unsloth launcher under the studio home"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    log_path = work / f"studio_{args.state}.log"
    log = open(log_path, "w", encoding = "utf-8", errors = "replace")
    srv = subprocess.Popen([exe, "studio", "-p", str(port), "-H", "127.0.0.1"], env = env, stdout = log,
                           stderr = subprocess.STDOUT, stdin = subprocess.DEVNULL,
                           creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    try:
        up = False
        for _ in range(300):
            try:
                if _req(f"{base}/api/health", timeout = 3)[0] == 200:
                    up = True
                    break
            except Exception:  # noqa: BLE001
                pass
            if srv.poll() is not None:
                break
            time.sleep(2)
        obs["launched"] = up
        if not up:
            return 0
        pw = (home / "auth" / ".bootstrap_password").read_text(encoding = "utf-8").strip()
        st, raw = _req(f"{base}/api/auth/login", {"username": "unsloth", "password": pw})
        tok = _json(raw).get("access_token")
        if _json(raw).get("must_change_password"):
            npw = pw + "X1!"
            _req(f"{base}/api/auth/change-password", {"current_password": pw, "new_password": npw}, tok)
            tok = _json(_req(f"{base}/api/auth/login", {"username": "unsloth", "password": npw})[1]).get("access_token")
        obs["monitor_before"] = _json(_req(f"{base}/api/inference/monitor", token = tok)[1])
        t1 = time.time()
        st, raw = _req(f"{base}/api/inference/images/load",
                       {"model_path": "unsloth/MOSS-Image-SFT-GGUF", "gguf_filename": "MOSS-Image-SFT-Q8_0.gguf"},
                       tok, timeout = 600)
        obs["load_http"] = st
        if st != 200:
            obs["load_error"] = raw[:2000]
            return 0
        status = {}
        for _ in range(1080):
            prog = _json(_req(f"{base}/api/inference/images/load-progress", token = tok)[1])
            status = _json(_req(f"{base}/api/inference/images/status", token = tok)[1])
            if prog.get("phase") == "error":
                obs["load_error"] = prog.get("error")
                return 0
            if status.get("loaded") and prog.get("phase") in (None, "ready"):
                break
            time.sleep(5)
        obs["load_secs"] = round(time.time() - t1, 1)
        obs["status"] = {k: status.get(k) for k in ("loaded", "engine", "repo_id", "device", "offload_policy",
                                                    "memory_mode", "gguf_variant")}
        obs["monitor_loaded"] = _json(_req(f"{base}/api/inference/monitor", token = tok)[1])
        t2 = time.time()
        st, raw = _req(f"{base}/api/inference/images/generate",
                       {"prompt": "a red fox sitting in the snow, photo", "width": args.size, "height": args.size,
                        "steps": args.steps, "guidance": 2.5, "seed": 42}, tok, timeout = 7200)
        obs["gen_http"] = st
        obs["gen_secs"] = round(time.time() - t2, 1)
        if st != 200:
            obs["gen_error"] = raw[:2000]
        else:
            j = _json(raw)
            imgs = j.get("images") or []
            obs["n_images"] = len(imgs) if isinstance(imgs, list) else None
    except Exception as e:  # noqa: BLE001
        obs["probe_error"] = f"{type(e).__name__}: {e}"
    finally:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output = True, timeout = 60)
        except Exception:  # noqa: BLE001
            pass
        log.close()
        text = log_path.read_text(encoding = "utf-8", errors = "replace") if log_path.exists() else ""
        keep = [l for l in text.splitlines() if any(k in l for k in (
            "sd_cpp", "sd-server", "Last output", "[E]", "[W]", "auto-fit", "generate_failed", "out of memory",
            "budget", "Vulkan", "HIP", "ROCm", "backend"))]
        obs["studio_log_tail"] = "\n".join(keep[-150:])[-12000:]
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
