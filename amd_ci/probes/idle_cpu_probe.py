#!/usr/bin/env python3
"""Probe: does this checkout's Windows ROCm torch wheel leave CPU threads spinning at idle?

Observes only (issue #12942). Reads the multi-arch ROCm tag + torch version the checkout's
install.ps1 pins, installs exactly that wheel into a per-tag venv, then runs a set of arms,
each in a FRESH python.exe: the child performs one action (import, GPU init, one CPU matmul,
...), prints a READY line, then sleeps. The parent samples the idle child from outside with
psutil, so the sampling adds no BLAS work to the measured process.

Per arm and repeat it records busy cores (process CPU seconds / wall seconds), thread count,
the hottest threads, and what the loaded OpenBLAS DLL reports about itself
(openblas_get_config / get_parallel / get_num_threads). Judging is the criteria module's job.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MULTIARCH_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"

CHILD = r'''
import ctypes, json, os, sys, threading, time

action = sys.argv[1]
# Symbol renames seen in shipped OpenBLAS builds (numpy's is scipy_openblas_*64_).
NAMINGS = [(p, s) for p in ("", "rocm_", "scipy_") for s in ("", "64_", "_64")]
info = {"action": action, "env": {k: os.environ.get(k) for k in
        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS")}}

def openblas_report(tag):
    import psutil
    found = []
    try:
        for m in psutil.Process().memory_maps():
            p = m.path
            if "openblas" in os.path.basename(p).lower() and p not in found:
                found.append(p)
    except Exception as e:
        info[tag + "_maps_error"] = repr(e)
    out = []
    for p in found:
        rec = {"path": p}
        try:
            lib = ctypes.CDLL(p)
            for prefix, suffix in NAMINGS:
                fn = getattr(lib, prefix + "openblas_get_config" + suffix, None)
                if fn is None:
                    continue
                fn.restype = ctypes.c_char_p
                rec["prefix"], rec["suffix"] = prefix, suffix
                rec["config"] = (fn() or b"").decode("utf-8", "replace")
                for name in ("openblas_get_parallel", "openblas_get_num_threads"):
                    g = getattr(lib, prefix + name + suffix, None)
                    if g is not None:
                        g.restype = ctypes.c_int
                        rec[name] = g()
                break
            else:
                rec["error"] = "no openblas_get_config export under any known naming"
        except Exception as e:
            rec["error"] = repr(e)
        out.append(rec)
    info[tag] = out
    return out

def matmul():
    import torch
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1024)
    c = a @ b
    info["matmul_ok"] = bool(torch.isfinite(c).all().item())

def gpu_init():
    import torch
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        x = torch.ones(256, 256, device="cuda")
        info["gpu_sum"] = float((x @ x).sum().item())
        info["gpu_name"] = torch.cuda.get_device_name(0)

def run():
    if action == "noop":
        return
    import torch
    info["torch"] = torch.__version__
    info["hip"] = getattr(torch.version, "hip", None)
    info["torch_threads"] = torch.get_num_threads()
    if action in ("gpu_init", "gpu_init_env1"):
        gpu_init()
    elif action in ("matmul", "matmul_env1", "matmul_omp1"):
        matmul()
    elif action == "matmul_bg_env1":
        t = threading.Thread(target=matmul, name="warm")
        t.start(); t.join()
    elif action == "matmul_setter":
        matmul()
        openblas_report("openblas_before_setter")
        for rec in info["openblas_before_setter"]:
            lib = ctypes.CDLL(rec["path"])
            fn = getattr(lib, rec.get("prefix", "") + "openblas_set_num_threads" + rec.get("suffix", ""), None)
            if fn is not None:
                fn(1)
                rec["set_called"] = True
    else:
        raise SystemExit("unknown action " + action)

try:
    run()
except Exception as e:
    info["action_error"] = repr(e)
openblas_report("openblas")
print("READY " + json.dumps(info), flush=True)
time.sleep(100000)
'''

ARMS = {
    # name: env overrides on top of a scrubbed base env
    "noop": {},
    "import": {},
    "gpu_init": {},
    "gpu_init_env1": {"OPENBLAS_NUM_THREADS": "1"},
    "matmul": {},
    "matmul_env1": {"OPENBLAS_NUM_THREADS": "1"},
    "matmul_omp1": {"OMP_NUM_THREADS": "1"},
    "matmul_bg_env1": {"OPENBLAS_NUM_THREADS": "1"},
    "matmul_setter": {},
}
SCRUB = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS",
         "GOTO_NUM_THREADS", "OPENBLAS_DEFAULT_NUM_THREADS")


def pinned_wheel(checkout: Path) -> tuple[str, str]:
    text = (checkout / "install.ps1").read_text(encoding = "utf-8", errors = "replace")
    tag = re.search(r'^\s*\$MultiArchTag\s*=\s*"([^"]+)"', text, re.M)
    ver = re.search(r'^\s*\$MultiArchTorchVersion\s*=\s*"([^"]+)"', text, re.M)
    if not tag or not ver:
        raise SystemExit("install.ps1 pins no multi-arch ROCm torch")
    return tag.group(1), ver.group(1)


def ensure_venv(work: Path, tag: str, ver: str, gfx: str, log: list) -> Path:
    venv = work / f"venv_{tag}"
    py = venv / "Scripts" / "python.exe"
    if not py.is_file():
        base = getattr(sys, "_base_executable", None) or sys.executable
        subprocess.run([base, "-m", "venv", str(venv)], check = True)
    spec = f"torch[device-{gfx}]=={ver}+{tag}"
    cmd = [str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check",
           "--index-url", MULTIARCH_INDEX, "--extra-index-url", "https://pypi.org/simple",
           spec, "psutil"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output = True, text = True)
    log.append({"cmd": " ".join(cmd), "rc": r.returncode, "s": round(time.time() - t0),
                "tail": (r.stdout + r.stderr)[-3000:]})
    if r.returncode != 0:
        raise SystemExit(f"pip install {spec} failed")
    return py


def measure(py: Path, child: Path, arm: str, env_extra: dict, settle: float, window: float) -> dict:
    import psutil
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env.update(env_extra)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen([str(py), str(child), arm], env = env, stdout = subprocess.PIPE,
                            stderr = subprocess.STDOUT, text = True, encoding = "utf-8",
                            errors = "replace")
    rec: dict = {"arm": arm, "env": env_extra}
    try:
        t0 = time.time()
        info = None
        noise = []
        while time.time() - t0 < 300:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith("READY "):
                info = json.loads(line[6:])
                break
            noise.append(line.rstrip())
        rec["child_output"] = noise[-20:]
        if info is None:
            rec["error"] = "child never became READY"
            return rec
        rec["child"] = info
        rec["ready_s"] = round(time.time() - t0, 1)
        p = psutil.Process(proc.pid)
        time.sleep(settle)
        c0 = p.cpu_times()
        th0 = {t.id: t.user_time + t.system_time for t in p.threads()}
        w0 = time.perf_counter()
        time.sleep(window)
        c1 = p.cpu_times()
        threads1 = p.threads()
        wall = time.perf_counter() - w0
        busy = (c1.user + c1.system - c0.user - c0.system) / wall
        per = sorted(((t.user_time + t.system_time - th0.get(t.id, 0.0)) / wall, t.id)
                     for t in threads1)
        rec.update({
            "busy_cores": round(busy, 3),
            "threads": len(threads1),
            "threads_over_half_core": sum(1 for b, _ in per if b > 0.5),
            "top_threads": [[round(b, 3), tid] for b, tid in per[-8:][::-1]],
            "window_s": round(wall, 1),
        })
        return rec
    finally:
        proc.kill()
        proc.wait(timeout = 30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--gfx", default = "gfx1151")
    ap.add_argument("--repeats", type = int, default = 3)
    ap.add_argument("--settle", type = float, default = 5.0)
    ap.add_argument("--window", type = float, default = 40.0)
    args = ap.parse_args()

    work = Path(os.environ.get("AMD_CI_WORK") or args.out.parent)
    obs: dict = {"state": args.state, "cpu_count": os.cpu_count(), "install": [], "arms": {}}
    try:
        tag, ver = pinned_wheel(args.checkout)
        obs["tag"], obs["torch_pin"] = tag, ver
        py = ensure_venv(work, tag, ver, args.gfx, obs["install"])
        child = work / "idle_cpu_child.py"
        child.write_text(CHILD, encoding = "utf-8")
        for rep in range(args.repeats):
            for arm, env_extra in ARMS.items():
                print(f"[{args.state}] {tag} rep {rep} arm {arm}", flush = True)
                r = measure(py, child, arm, env_extra, args.settle, args.window)
                obs["arms"].setdefault(arm, []).append(r)
                print(f"    busy_cores={r.get('busy_cores')} threads={r.get('threads')} "
                      f"hot={r.get('threads_over_half_core')} err={r.get('error')}", flush = True)
    except BaseException as e:  # noqa: BLE001 -- record it; the criteria gates turn it into INCONCLUSIVE
        obs["probe_error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
