#!/usr/bin/env python3
"""Probe (issue #9404): does a Z-Image-Turbo GGUF Q4_K_M render with default settings finish?

Observes only. Per state: `install.sh --local` into an isolated UNSLOTH_STUDIO_HOME (the
same Studio venv the Desktop app builds), launch `run.py --api-only`, log in, POST
/api/inference/images/load with the GGUF pick, POST /api/inference/images/generate with
the API defaults (1024x1024, 9 steps, guidance 0: the UI's defaults, speed_mode unset =
the UI's "auto"), and poll /images/generate-progress the way the UI does. While step 0
lasts it samples GPU busy % and the server tree's CPU %, and dumps every server thread
via faulthandler (SIGUSR1) so a stall can be located. A cap ends a render that never
finishes. A second (warm) render runs only after a finished first one.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "unsloth/Z-Image-Turbo-GGUF"
GGUF = "z-image-turbo-Q4_K_M.gguf"
PROMPT = "A cozy cabin in a snowy forest at dusk, warm light in the windows"

SITECUSTOMIZE = r'''
import faulthandler, os, signal
_d = os.environ.get("AMDCI_STACK_DIR")
if _d:
    try:
        os.makedirs(_d, exist_ok=True)
        _f = open(os.path.join(_d, "stacks_%d.txt" % os.getpid()), "a", buffering=1)
        faulthandler.register(signal.SIGUSR1, file=_f, all_threads=True, chain=False)
    except Exception:
        pass
'''


def sh(cmd, log, label, timeout=5400, **kw):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout, **kw)
        rc, out = r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc, out = -9, f"timeout after {timeout}s: {e}"
    log.append({"step": label, "rc": rc, "s": round(time.time() - t0), "tail": out[-6000:]})
    print(f"  [{label}] rc={rc} {round(time.time() - t0)}s", flush=True)
    return rc, out


def api(base, path, token=None, body=None, method=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                return r.status, json.loads(raw)
            except Exception:  # noqa: BLE001
                return r.status, raw
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:  # noqa: BLE001
            return e.code, None
    except Exception as e:  # noqa: BLE001
        return None, repr(e)


def gpu_busy():
    vals = []
    for p in glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"):
        try:
            vals.append(int(Path(p).read_text().strip()))
        except Exception:  # noqa: BLE001
            pass
    return max(vals) if vals else None


def tree(psutil, pid):
    try:
        root = psutil.Process(pid)
        return [root] + root.children(recursive=True)
    except psutil.Error:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=1800, help="seconds before a render counts as never finishing")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_dir = out_path.parent
    work = Path(os.environ.get("AMD_CI_WORK", "/tmp")) / f"imagegen_{args.state}"
    work.mkdir(parents=True, exist_ok=True)
    src = Path(args.checkout)
    rec: dict = {"state": args.state, "steps_log": [], "commit": None}
    flush = lambda: out_path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")  # noqa: E731

    rc, out = sh(["git", "-C", str(src), "rev-parse", "HEAD"], rec["steps_log"], "rev-parse")
    rec["commit"] = out.strip()
    home = work / "studio_home"
    env = dict(os.environ, UNSLOTH_STUDIO_HOME=str(home), UNSLOTH_SKIP_AUTOSTART="1",
               UV_EXTRA_INDEX_URL="https://pypi.org/simple")
    if os.environ.get("AMDCI_SKIP_INSTALL") == "1" and (home / "unsloth_studio" / "bin" / "python").exists():
        rc = 0
    else:
        rc, out = sh(["bash", "install.sh", "--local"], rec["steps_log"], "install.sh --local", cwd=str(src), env=env)
    rec["install_rc"] = rc
    py = None
    for c in [home / "unsloth_studio", *sorted(home.glob(".venv_t5_*"))]:
        if (c / "bin" / "python").exists():
            py = c / "bin" / "python"
            break
    if rc != 0 or py is None:
        rec["error"] = f"install failed rc={rc} venv={py}"
        flush()
        return 0
    rec["python"] = str(py)
    pins = os.environ.get("AMDCI_BASE_PINS", "").split()
    if args.state == "base" and pins:
        # The old tag's resolver now picks releases newer than its transformers accepts.
        rc, _ = sh([str(py), "-m", "pip", "install", "-q", *pins], rec["steps_log"], "base pins " + " ".join(pins))
        rec["base_pins"] = pins
    sh([str(py), "-m", "pip", "list", "--format=freeze"], rec["steps_log"], "pip freeze")
    sh([str(py), "-m", "pip", "install", "-q", "psutil"], rec["steps_log"], "psutil")
    rc, out = sh([str(py), "-c", "import torch,json;print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,"
                  "'dev':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
                  "'arch':getattr(torch.cuda.get_device_properties(0),'gcnArchName',None) if torch.cuda.is_available() else None}))"],
                 rec["steps_log"], "torch info")
    try:
        rec["torch"] = json.loads(out.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        rec["torch"] = out[-500:]
    flush()

    import psutil  # noqa: E402  (installed into the runner python by the workflow)

    shim = work / "shim"
    shim.mkdir(exist_ok=True)
    (shim / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    stack_dir = out_dir / f"stacks_{args.state}"
    port = args.port or (8930 if args.state == "base" else 8931)
    password = "Pw" + secrets.token_hex(10)
    senv = dict(env, PYTHONUNBUFFERED="1", UNSLOTH_DISABLE_AUTO_UPDATES="1",
                PYTHONPATH=str(shim) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
                AMDCI_STACK_DIR=str(stack_dir),
                HF_HOME=os.environ.get("AMDCI_HF_HOME", str(work / "hf")),
                TRITON_CACHE_DIR=str(work / "triton_cache"),
                MIOPEN_USER_DB_PATH=str(work / "miopen_db"),
                MIOPEN_CUSTOM_CACHE_DIR=str(work / "miopen_cache"))
    backend = src / "studio" / "backend"
    server_log = out_dir / f"server_{args.state}.log"
    logf = open(server_log, "w", encoding="utf-8")
    proc = subprocess.Popen([str(py), "-u", str(backend / "run.py"), "--api-only", "--host", "127.0.0.1",
                             "--port", str(port), "--password", password],
                            cwd=str(backend), env=senv, stdout=logf, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    base_url = f"http://127.0.0.1:{port}"
    try:
        t0 = time.time()
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                rec["error"] = f"server exited rc={proc.returncode}"
                flush()
                return 0
            st, _ = api(base_url, "/api/health", timeout=3)
            if st == 200:
                break
            time.sleep(2)
        else:
            rec["error"] = "server never healthy"
            flush()
            return 0
        rec["healthy_s"] = round(time.time() - t0, 1)
        st, tok = api(base_url, "/api/auth/login", body={"username": "unsloth", "password": password})
        if st != 200 or not isinstance(tok, dict):
            rec["error"] = f"login failed {st} {str(tok)[:300]}"
            flush()
            return 0
        token = tok.get("access_token")
        if tok.get("must_change_password"):
            new_pw = "Pw" + secrets.token_hex(12)
            st, tok2 = api(base_url, "/api/auth/change-password", token,
                           {"current_password": password, "new_password": new_pw})
            rec["changed_password"] = st
            if st == 200 and isinstance(tok2, dict):
                token = tok2.get("access_token") or token

        # Load.
        t_load = time.time()
        load_body = {"model_path": REPO, "gguf_filename": GGUF}
        result: dict = {}

        def call(path, body, key, timeout):
            result[key] = api(base_url, path, token, body, timeout=timeout)

        th = threading.Thread(target=call, args=("/api/inference/images/load", load_body, "load", 3600), daemon=True)
        th.start()
        while th.is_alive() and time.time() - t_load < 3600:
            th.join(10)
        st, body = result.get("load", (None, "load timed out"))
        rec["load_post_status"] = st
        rec["load_body"] = body if isinstance(body, dict) else str(body)[:2000]
        flush()
        if st != 200:
            rec["error"] = f"load failed {st}"
            flush()
            return 0
        # The route can return while the load continues in the background (the UI polls).
        prog = status = None
        while time.time() - t_load < 3600:
            _, status = api(base_url, "/api/inference/images/status", token, timeout=30)
            if isinstance(status, dict) and status.get("loaded"):
                break
            _, prog = api(base_url, "/api/inference/images/load-progress", token, timeout=30)
            if isinstance(prog, dict) and prog.get("phase") == "error":
                break
            if proc.poll() is not None:
                break
            time.sleep(5)
        rec["load_s"] = round(time.time() - t_load, 1)
        rec["load_status_body"] = status if isinstance(status, dict) else str(status)[:2000]
        rec["load_progress"] = prog
        if not (isinstance(status, dict) and status.get("loaded")):
            rec["error"] = f"model never loaded: {str(prog)[:1000]}"
            flush()
            return 0
        rec["load_status"] = 200

        renders = []
        for idx in range(2):
            r = render(base_url, token, proc, psutil, stack_dir, args.cap, idx)
            renders.append(r)
            rec["renders"] = renders
            flush()
            if not r.get("finished"):
                break
        return 0
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        logf.close()
        try:
            text = server_log.read_text(encoding="utf-8", errors="replace")
            keep = [ln for ln in text.splitlines() if "diffusion" in ln or "Traceback" in ln or "Error" in ln
                    or "it/s" in ln or "s/it" in ln]
            rec["server_log_diffusion"] = keep[-200:]
            rec["server_log_tail"] = text[-6000:]
        except Exception:  # noqa: BLE001
            pass
        flush()


def render(base_url, token, proc, psutil, stack_dir, cap, idx) -> dict:
    body = {"prompt": PROMPT, "seed": 42}
    out: dict = {"index": idx, "timeline": [], "samples": [], "dumps": []}
    result: dict = {}

    def call():
        result["gen"] = api(base_url, "/api/inference/images/generate", token, body, timeout=cap + 120)

    t0 = time.time()
    th = threading.Thread(target=call, daemon=True)
    th.start()
    last_step = -1
    dump_at = [60, 180, 420, 900, 1500]
    for p in tree(psutil, proc.pid):
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    while th.is_alive() and time.time() - t0 < cap:
        th.join(5)
        el = round(time.time() - t0, 1)
        st, prog = api(base_url, "/api/inference/images/generate-progress", token, timeout=10)
        step = prog.get("step") if isinstance(prog, dict) else None
        if step is not None and step != last_step:
            out["timeline"].append({"t": el, "step": step, "total": prog.get("total_steps")})
            last_step = step
            if step >= 1 and "first_step_s" not in out:
                out["first_step_s"] = el
        cpu = 0.0
        for p in tree(psutil, proc.pid):
            try:
                cpu += p.cpu_percent(None)
            except psutil.Error:
                pass
        out["samples"].append({"t": el, "step": step, "gpu_busy": gpu_busy(), "cpu_pct": round(cpu)})
        if dump_at and el >= dump_at[0]:
            dump_at.pop(0)
            if (last_step or 0) < 1:
                for p in tree(psutil, proc.pid):
                    try:
                        os.kill(p.pid, signal.SIGUSR1)
                    except Exception:  # noqa: BLE001
                        pass
                out["dumps"].append(el)
    out["elapsed_s"] = round(time.time() - t0, 1)
    if th.is_alive():
        out["finished"] = False
        out["error"] = f"no result after {cap}s (last step {last_step})"
        return out
    st, res = result.get("gen", (None, None))
    out["status"] = st
    imgs = res.get("images") if isinstance(res, dict) else None
    out["finished"] = st == 200 and bool(imgs)
    if not out["finished"]:
        out["error"] = f"generate {st}: {str(res)[:1500]}"
    else:
        first = imgs[0]
        url = first.get("url") if isinstance(first, dict) else None
        if url:
            req = urllib.request.Request(base_url + url, headers={"Authorization": f"Bearer {token}"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                out["png_bytes"] = len(data)
                out["png_ok"] = data[:8] == b"\x89PNG\r\n\x1a\n"
            except Exception as e:  # noqa: BLE001
                out["png_error"] = repr(e)
        out["response_keys"] = sorted(res.keys())
    return out


if __name__ == "__main__":
    sys.exit(main())
