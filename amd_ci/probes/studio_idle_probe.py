#!/usr/bin/env python3
"""Probe: how much CPU does an idle Studio backend (--api-only) burn on Windows ROCm?

Observes only (issue #12942). The state picks a Studio release via --tag-map (base=902,
head=903), not the checkout, because the question is release-to-release. Per state:
shallow-clone that tag, build its venv the way setup.ps1 does (ROCm multi-arch torch the
tag's install.ps1 pins, unsloth_zoo + the checkout's unsloth, then install_python_stack.py),
launch run.py --api-only with an isolated UNSLOTH_STUDIO_HOME, poll the four routes the
Desktop UI polls, and sample the INTERPRETER's CPU from outside (the venv python.exe is a
launcher). Ends with per-thread CPU and a py-spy dump. Judging is the criteria module's job.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MULTIARCH_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"
UPSTREAM = "https://github.com/unslothai/unsloth"
POLL_PATHS = ("/api/inference/status", "/api/inference/monitor", "/api/engines", "/api/liveness")


def run(cmd, log: list, label: str, timeout: int = 3600, **kw) -> int:
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8",
                           errors = "replace", timeout = timeout, **kw)
        rc, out = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = -9, f"timeout after {timeout}s: {e}"
    log.append({"step": label, "rc": rc, "s": round(time.time() - t0), "tail": out[-4000:]})
    print(f"  [{label}] rc={rc} {round(time.time() - t0)}s", flush = True)
    return rc


def build(work: Path, tag: str, gfx: str, log: list) -> tuple[Path, Path]:
    src = work / f"src_{tag}"
    if not (src / "install.ps1").is_file():
        if run(["git", "clone", "-q", "--depth", "1", "-b", tag, UPSTREAM, str(src)], log, "clone") != 0:
            raise SystemExit("clone failed")
    text = (src / "install.ps1").read_text(encoding = "utf-8", errors = "replace")
    wtag = re.search(r'^\s*\$MultiArchTag\s*=\s*"([^"]+)"', text, re.M).group(1)
    wver = re.search(r'^\s*\$MultiArchTorchVersion\s*=\s*"([^"]+)"', text, re.M).group(1)
    venv = work / f"studio_venv_{tag}"
    py = venv / "Scripts" / "python.exe"
    if not py.is_file():
        base = getattr(sys, "_base_executable", None) or sys.executable
        if run([base, "-m", "venv", str(venv)], log, "venv") != 0:
            raise SystemExit("venv failed")
    pip = [str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    if run(pip + ["--index-url", MULTIARCH_INDEX, "--extra-index-url", "https://pypi.org/simple",
                  f"torch[device-{gfx}]=={wver}+{wtag}", "psutil", "py-spy"], log, f"torch {wver}+{wtag}") != 0:
        raise SystemExit("torch install failed")
    if run(pip + ["unsloth_zoo"], log, "unsloth_zoo") != 0:
        raise SystemExit("unsloth_zoo install failed")
    if run(pip + ["--no-deps", str(src)], log, "unsloth (checkout, no deps)") != 0:
        raise SystemExit("unsloth install failed")
    env = dict(os.environ, UNSLOTH_EXPECTED_TORCH_TAG = "rocm",
               UNSLOTH_TORCH_INSTALL_INDEX_URL = MULTIARCH_INDEX,
               UNSLOTH_STUDIO_HOME = str(work / f"home_{tag}"), PYTHONUTF8 = "1")
    rc = run([str(py), str(src / "studio" / "install_python_stack.py")], log, "install_python_stack",
             timeout = 3000, env = env, cwd = str(src / "studio"))
    if rc != 0:
        raise SystemExit("install_python_stack failed")
    return py, src


def poller(port: int, password: str, every: float, stop: threading.Event, status: dict) -> None:
    base = f"http://127.0.0.1:{port}"
    token = None
    try:
        req = urllib.request.Request(base + "/api/auth/login", method = "POST",
                                     data = json.dumps({"username": "unsloth", "password": password}).encode(),
                                     headers = {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout = 30) as r:
            token = json.loads(r.read()).get("access_token")
    except Exception as e:  # noqa: BLE001
        status["login_error"] = repr(e)
    while not stop.is_set():
        for path in POLL_PATHS:
            try:
                req = urllib.request.Request(base + path, headers = {"Authorization": f"Bearer {token}"} if token else {})
                with urllib.request.urlopen(req, timeout = 30) as r:
                    code = r.status
            except urllib.error.HTTPError as e:
                code = e.code
            except Exception as e:  # noqa: BLE001
                code = type(e).__name__
            status.setdefault(path, {}).setdefault(str(code), 0)
            status[path][str(code)] += 1
        stop.wait(every)


def interpreter(psutil, launcher) -> "psutil.Process":
    """The process actually running run.py: the launcher's busiest-threaded descendant, else itself."""
    best = launcher
    for c in launcher.children(recursive = True):
        try:
            if "python" in c.name().lower() and c.num_threads() > best.num_threads():
                best = c
        except psutil.Error:
            pass
    return best


def idle_arm(py: Path, src: Path, work: Path, tag: str, arm: str, env_extra: dict, port: int,
             duration: float, interval: float, poll: float) -> dict:
    import psutil
    home = work / f"home_{tag}_{arm}"
    password = secrets.token_hex(12)
    env = dict(os.environ, UNSLOTH_STUDIO_HOME = str(home), UNSLOTH_DISABLE_AUTO_UPDATES = "1",
               PYTHONUNBUFFERED = "1", PYTHONUTF8 = "1")
    env.update(env_extra)
    backend = src / "studio" / "backend"
    logf = open(work / f"backend_{tag}_{arm}.log", "w", encoding = "utf-8")
    proc = subprocess.Popen([str(py), "-u", str(backend / "run.py"), "--api-only", "--host", "127.0.0.1",
                             "--port", str(port), "--password", password],
                            cwd = str(backend), env = env, stdout = logf, stderr = subprocess.STDOUT)
    rec: dict = {"arm": arm, "env": env_extra, "timeline": [], "log": logf.name}
    stop = threading.Event()
    try:
        t0, healthy = time.time(), None
        while time.time() - t0 < 600 and healthy is None:
            if proc.poll() is not None:
                rec["error"] = f"backend exited rc={proc.returncode}"
                return rec
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout = 3) as r:
                    if r.status == 200:
                        healthy = time.time() - t0
            except Exception:  # noqa: BLE001
                time.sleep(2)
        rec["healthy_s"] = healthy
        if healthy is None:
            rec["error"] = "never healthy"
            return rec
        rec["poll_status"] = {}
        threading.Thread(target = poller, args = (port, password, poll, stop, rec["poll_status"]),
                         daemon = True).start()
        launcher = psutil.Process(proc.pid)
        p = interpreter(psutil, launcher)
        rec["launcher_pid"], rec["measured_pid"] = launcher.pid, p.pid
        prev_cpu = sum(p.cpu_times()[:2]); prev_t = time.time()
        th_mark = None
        while time.time() - t0 - healthy < duration:
            time.sleep(interval)
            if proc.poll() is not None:
                rec["error"] = f"backend exited rc={proc.returncode} while idle"
                break
            cpu, now = sum(p.cpu_times()[:2]), time.time()
            rec["timeline"].append({"t": round(now - t0, 1), "busy": round((cpu - prev_cpu) / (now - prev_t), 3),
                                    "threads": p.num_threads()})
            prev_cpu, prev_t = cpu, now
            if th_mark is None and duration - (now - t0 - healthy) <= 60:
                th_mark = ({t.id: t.user_time + t.system_time for t in p.threads()}, now)
        if th_mark is not None and proc.poll() is None:
            dt = time.time() - th_mark[1]
            per = sorted((((t.user_time + t.system_time) - th_mark[0].get(t.id, 0.0)) / dt, t.id) for t in p.threads())
            rec["top_threads"] = [[round(b, 3), tid] for b, tid in per[-12:][::-1]]
            rec["threads_over_half_core"] = sum(1 for b, _ in per if b > 0.5)
            spy = py.parent / "py-spy.exe"
            if spy.is_file():
                d = subprocess.run([str(spy), "dump", "--pid", str(p.pid)], capture_output = True, text = True,
                                   encoding = "utf-8", errors = "replace", timeout = 120)
                rec["py_spy"] = (d.stdout or d.stderr)[-30000:]
        last = [r["busy"] for r in rec["timeline"][-int(60 / interval):]]
        rec["busy_last60"] = round(sum(last) / len(last), 3) if last else None
        rec["busy_max"] = max((r["busy"] for r in rec["timeline"]), default = None)
        return rec
    finally:
        stop.set()
        try:
            for c in psutil.Process(proc.pid).children(recursive = True):
                c.kill()
        except Exception:  # noqa: BLE001
            pass
        proc.kill()
        proc.wait(timeout = 60)
        logf.close()
        try:
            rec["log_tail"] = Path(logf.name).read_text(encoding = "utf-8", errors = "replace")[-6000:]
        except OSError:
            pass


ARMS = {
    "default": {},
    "no_torch_warm": {"UNSLOTH_STUDIO_DISABLE_TORCH_WARM": "1"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--tag-map", default = "base=v0.1.902-beta,head=v0.1.903-beta")
    ap.add_argument("--gfx", default = "gfx1151")
    ap.add_argument("--duration", type = float, default = 300)
    ap.add_argument("--interval", type = float, default = 5)
    ap.add_argument("--poll", type = float, default = 10)
    args = ap.parse_args()

    tags = dict(kv.split("=", 1) for kv in args.tag_map.split(","))
    work = Path(os.environ.get("AMD_CI_WORK") or args.out.parent)
    obs: dict = {"state": args.state, "cpu_count": os.cpu_count(), "build": [], "arms": {}}
    try:
        tag = tags[args.state]
        obs["tag"] = tag
        py, src = build(work, tag, args.gfx, obs["build"])
        for i, (arm, env_extra) in enumerate(ARMS.items()):
            print(f"[{args.state}] {tag} arm {arm}", flush = True)
            r = idle_arm(py, src, work, tag, arm, env_extra, 8890 + i + (10 if args.state == "head" else 0),
                         args.duration, args.interval, args.poll)
            obs["arms"][arm] = r
            print(f"    busy_last60={r.get('busy_last60')} max={r.get('busy_max')} "
                  f"hot={r.get('threads_over_half_core')} err={r.get('error')}", flush = True)
    except BaseException as e:  # noqa: BLE001 -- recorded; the criteria gates turn it into INCONCLUSIVE
        obs["probe_error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
