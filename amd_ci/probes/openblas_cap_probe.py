#!/usr/bin/env python3
"""Probe: does this checkout's thread cap reach rocm-openblas.dll on Windows ROCm? (#12942, PR #13048)

Observes only. Per state: a venv on the ROCm multi-arch torch the checkout's install.ps1 pins. A fresh child
does what run.py does first (configure_cpu_threads() from THIS checkout's studio/backend), imports torch, reports
the loaded OpenBLAS thread count, then keeps CPU BLAS busy; the parent samples the child's busy cores from outside
(the venv python.exe is a launcher, so the interpreter PID is the one the child reports). Then the real Studio
stack is built from the checkout and run.py --api-only is booted and polled like the Desktop UI, to show the
backend still starts and idles. Judging is the criteria module's job.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("studio_idle_probe", HERE / "studio_idle_probe.py")
sip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sip)

CHILD = r'''
import ctypes, json, os, sys, threading, time
backend = sys.argv[1]
sys.path.insert(0, backend)
info = {"pid": os.getpid(), "mode": sys.argv[3]}
if sys.argv[3] == "user1":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
from utils.cpu_threads import configure_cpu_threads
configure_cpu_threads()
info["env_openblas"] = os.environ.get("OPENBLAS_NUM_THREADS")
import torch
info["torch"] = torch.__version__
info["hip"] = getattr(torch.version, "hip", None)
info["torch_threads"] = torch.get_num_threads()
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
k32.GetModuleHandleW.restype = ctypes.c_void_p
h = k32.GetModuleHandleW("rocm-openblas.dll")
info["dll_loaded"] = bool(h)
if h:
    lib = ctypes.CDLL("rocm-openblas.dll", handle=h)
    lib.openblas_get_num_threads.restype = ctypes.c_int
    info["openblas_threads"] = lib.openblas_get_num_threads()
a = torch.randn(1024, 1024); b = torch.randn(1024, 1024)
info["matmul_ok"] = bool(torch.isfinite(a @ b).all().item())
stop = time.time() + float(sys.argv[2])
print("READY " + json.dumps(info), flush=True)
while time.time() < stop:
    a @ b
print("DONE", flush=True)
time.sleep(100000)
'''


# A Desktop-spawned worker: no run.py, OPENBLAS_NUM_THREADS inherited from the backend, the worker module imported
# first (as unpickling the spawn target does), then torch.
WORKER = r'''
import ctypes, json, os, sys
sys.path.insert(0, os.getcwd())
info = {"pid": os.getpid(), "env_openblas": os.environ.get("OPENBLAS_NUM_THREADS")}
try:
    import core.training.worker  # noqa: F401
    info["worker_import"] = "ok"
except Exception as e:
    info["worker_import"] = repr(e)[:300]
import torch
info["torch_threads"] = torch.get_num_threads()
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
k32.GetModuleHandleW.restype = ctypes.c_void_p
h = k32.GetModuleHandleW("rocm-openblas.dll")
info["dll_loaded"] = bool(h)
if h:
    lib = ctypes.CDLL("rocm-openblas.dll", handle=h)
    lib.openblas_get_num_threads.restype = ctypes.c_int
    info["openblas_threads"] = lib.openblas_get_num_threads()
print("WORKER " + json.dumps(info), flush=True)
'''


def worker_arm(py: Path, checkout: Path, work: Path) -> dict:
    clean = {k: v for k, v in os.environ.items()
             if k not in ("UNSLOTH_CPU_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    backend = checkout / "studio" / "backend"
    # The env the backend hands its workers: whatever THIS checkout's configure_cpu_threads() leaves behind.
    cfg = subprocess.run([str(py), "-c", "import json, os, sys; sys.path.insert(0, os.getcwd()); "
                          "from utils.cpu_threads import configure_cpu_threads; configure_cpu_threads(); "
                          "print('ENV ' + json.dumps(dict(os.environ)))"], cwd = str(backend), env = clean,
                         capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = 300)
    env = next((json.loads(x[4:]) for x in cfg.stdout.splitlines() if x.startswith("ENV ")), None)
    if env is None:
        return {"error": f"configure rc={cfg.returncode}", "tail": (cfg.stdout + cfg.stderr)[-3000:]}
    env.update(UNSLOTH_STUDIO_HOME = str(work / "home_worker"), PYTHONUTF8 = "1")
    r = subprocess.run([str(py), "-c", WORKER], cwd = str(checkout / "studio" / "backend"), env = env,
                       capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = 900)
    for line in r.stdout.splitlines():
        if line.startswith("WORKER "):
            out = json.loads(line[7:])
            out["env_marker"] = env.get("UNSLOTH_OPENBLAS_DEFAULTED")
            return out
    return {"error": f"rc={r.returncode}", "tail": (r.stdout + r.stderr)[-3000:]}


# Plain `import numpy`, then torch's CPU matmul at several rocm-openblas thread counts; one fresh process per cell.
PERF = r"""
import ctypes, json, os, sys, time
import psutil
me = psutil.Process()
def snap():
    m = me.memory_info()
    return {"threads": me.num_threads(), "private_mb": round(getattr(m, "private", 0) / 2**20, 1),
            "vms_mb": round(m.vms / 2**20, 1), "rss_mb": round(m.rss / 2**20, 1)}
def med(f, n=7):
    f(); f()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return round(sorted(ts)[n // 2] * 1e3, 2)
kind = sys.argv[1]
out = {"kind": kind, "env": os.environ.get("OPENBLAS_NUM_THREADS"), "logical": psutil.cpu_count(),
       "physical": psutil.cpu_count(logical=False), "before": snap()}
if kind == "numpy":
    import numpy as np
    out["numpy"] = np.__version__
    try:
        out["blas"] = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    except Exception as e:
        out["blas"] = repr(e)
    out["after_import"] = snap()
    a = np.random.default_rng(0).standard_normal((2048, 2048))
    out["matmul_2048_ms"] = med(lambda: a @ a)
    d = np.random.default_rng(1).standard_normal((1000, 1024)); q = d[0]
    out["cosine_1000x1024_ms"] = med(lambda: d @ q / (np.linalg.norm(d, axis=1) * np.linalg.norm(q)), 31)
    out["after_blas"] = snap()
else:
    import torch
    out["after_import"] = snap()
    out["torch_threads"] = torch.get_num_threads()
    n = {"one": 1, "torch": torch.get_num_threads(), "logical": psutil.cpu_count()}[kind]
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]; k32.GetModuleHandleW.restype = ctypes.c_void_p
    lib = ctypes.CDLL("rocm-openblas.dll", handle=k32.GetModuleHandleW("rocm-openblas.dll"))
    lib.openblas_set_num_threads.argtypes = [ctypes.c_int]
    lib.openblas_get_num_threads.restype = ctypes.c_int
    lib.openblas_set_num_threads(n)
    out["dll_threads"] = lib.openblas_get_num_threads()
    a = torch.randn(2048, 2048)
    out["matmul_2048_ms"] = med(lambda: a @ a)
    x = torch.randn(64, 1500, 384); w = torch.nn.Linear(384, 1536)
    with torch.no_grad():
        out["linear_64x1500x384_ms"] = med(lambda: w(x))
    out["after_blas"] = snap()
print("PERF " + json.dumps(out), flush=True)
"""


def perf_arm(py: Path, work: Path) -> list:
    script = work / "perf_child.py"
    script.write_text(PERF, encoding = "utf-8")
    base = {k: v for k, v in os.environ.items()
            if k not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "UNSLOTH_CPU_THREADS", "MKL_NUM_THREADS")}
    cells = [("numpy", None), ("numpy", "1"), ("one", None), ("torch", None), ("logical", None)]
    rows = []
    for rep in range(3):
        for kind, val in cells:
            env = dict(base, **({"OPENBLAS_NUM_THREADS": val} if val else {}))
            r = subprocess.run([str(py), str(script), kind], env = env, capture_output = True, text = True,
                               encoding = "utf-8", errors = "replace", timeout = 900, cwd = str(work))
            row = next((json.loads(x[5:]) for x in r.stdout.splitlines() if x.startswith("PERF ")), None)
            rows.append(dict(row or {"kind": kind, "env": val, "error": (r.stdout + r.stderr)[-1500:]}, rep = rep))
            print(f"[perf] {rows[-1]}"[:400], flush = True)
    return rows


def pinned(src: Path) -> tuple[str, str]:
    import re
    text = (src / "install.ps1").read_text(encoding = "utf-8", errors = "replace")
    tag = re.search(r'^\s*\$MultiArchTag\s*=\s*"([^"]+)"', text, re.M).group(1)
    ver = re.search(r'^\s*\$MultiArchTorchVersion\s*=\s*"([^"]+)"', text, re.M).group(1)
    return tag, ver


def cap_arm(py: Path, checkout: Path, work: Path, busy_s: float, mode: str = "default") -> dict:
    import psutil
    child = work / "openblas_cap_child.py"
    child.write_text(CHILD, encoding = "utf-8")
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "UNSLOTH_CPU_THREADS", "MKL_NUM_THREADS")}
    proc = subprocess.Popen([str(py), str(child), str(checkout / "studio" / "backend"), str(busy_s), mode],
                            env = env, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True,
                            encoding = "utf-8", errors = "replace")
    rec: dict = {}
    noise = []
    try:
        t0 = time.time()
        while time.time() - t0 < 600:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith("READY "):
                rec["child"] = json.loads(line[6:])
                break
            noise.append(line.rstrip())
        rec["child_output"] = noise[-20:]
        if "child" not in rec:
            rec["error"] = "child never became READY"
            return rec
        p = psutil.Process(rec["child"]["pid"])
        time.sleep(2)
        c0, w0 = sum(p.cpu_times()[:2]), time.perf_counter()
        time.sleep(max(1.0, busy_s - 6))
        c1, w1 = sum(p.cpu_times()[:2]), time.perf_counter()
        rec["busy_cores_under_blas"] = round((c1 - c0) / (w1 - w0), 3)
        rec["threads_under_blas"] = p.num_threads()
        return rec
    finally:
        try:
            for c in psutil.Process(proc.pid).children(recursive = True):
                c.kill()
        except Exception:  # noqa: BLE001
            pass
        proc.kill()
        proc.wait(timeout = 60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--gfx", default = "gfx1151")
    ap.add_argument("--busy", type = float, default = 20.0)
    ap.add_argument("--repeats", type = int, default = 3)
    ap.add_argument("--idle", type = float, default = 120.0)
    args = ap.parse_args()

    work = Path(os.environ.get("AMD_CI_WORK") or args.out.parent) / f"cap_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "cpu_count": os.cpu_count(), "build": [], "cap": [], "studio": None, "perf": None}
    try:
        checkout = args.checkout.resolve()
        tag, ver = pinned(checkout)
        obs["wheel"] = f"{ver}+{tag}"
        venv = work / "venv"
        py = venv / "Scripts" / "python.exe"
        base = getattr(sys, "_base_executable", None) or sys.executable
        if sip.run([base, "-m", "venv", str(venv)], obs["build"], "venv") != 0:
            raise SystemExit("venv failed")
        pip = [str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check"]
        if sip.run(pip + ["--index-url", sip.MULTIARCH_INDEX, "--extra-index-url", "https://pypi.org/simple",
                          f"torch[device-{args.gfx}]=={ver}+{tag}", "psutil", "py-spy", "numpy"], obs["build"], "torch") != 0:
            raise SystemExit("torch install failed")
        obs["cap_user1"] = []
        for i in range(args.repeats):
            for mode, key in (("default", "cap"), ("user1", "cap_user1")):
                r = cap_arm(py, checkout, work, args.busy, mode)
                obs[key].append(r)
                print(f"[{args.state}] cap {mode} rep {i}: threads={r.get('child', {}).get('openblas_threads')} "
                      f"torch={r.get('child', {}).get('torch_threads')} busy={r.get('busy_cores_under_blas')} "
                      f"err={r.get('error')}", flush = True)
        if args.state == "base":
            obs["perf"] = perf_arm(py, work)
        # The real backend from this checkout on the same venv, booted and polled like Desktop.
        for step, cmd in (("unsloth_zoo", pip + ["unsloth_zoo"]), ("unsloth", pip + ["--no-deps", str(checkout)])):
            if sip.run(cmd, obs["build"], step) != 0:
                raise SystemExit(f"{step} install failed")
        env = dict(os.environ, UNSLOTH_EXPECTED_TORCH_TAG = "rocm", UNSLOTH_TORCH_INSTALL_INDEX_URL = sip.MULTIARCH_INDEX,
                   UNSLOTH_STUDIO_HOME = str(work / "home_build"), PYTHONUTF8 = "1")
        if sip.run([str(py), str(checkout / "studio" / "install_python_stack.py")], obs["build"],
                   "install_python_stack", timeout = 3000, env = env, cwd = str(checkout / "studio")) != 0:
            raise SystemExit("install_python_stack failed")
        obs["worker"] = worker_arm(py, checkout, work)
        print(f"[{args.state}] worker: {obs['worker']}", flush = True)
        obs["studio"] = sip.idle_arm(py, checkout, work, args.state, "default", {},
                                     8870 + (10 if args.state == "head" else 0), args.idle, 5, 10)
    except BaseException as e:  # noqa: BLE001 -- recorded; the criteria gates turn it into INCONCLUSIVE
        obs["probe_error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
