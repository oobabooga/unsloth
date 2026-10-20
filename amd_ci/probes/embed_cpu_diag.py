#!/usr/bin/env python3
"""Diagnostic (#14313): why is bge-small CPU embedding ~7 chunks/s on 32 busy CPUs with Windows ROCm torch?

Two venvs on the same box: AMD's ROCm multi-arch torch (as install.ps1 pins it) and the pytorch.org CPU wheel of
the same version. Fresh process per cell: sgemm GFLOPS, numpy reference, bge-small encode chunks/s + busy cores at
several (torch threads, rocm-openblas threads) pairs, and a profiler op table for the ROCm default.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MULTIARCH = "https://repo.amd.com/rocm/whl-multi-arch/"

CELL = r'''
import ctypes, json, os, sys, time
torch_threads, blas_threads, profile = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
import psutil
me = psutil.Process()
import torch
out = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None)}
blas = None
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]; k32.GetModuleHandleW.restype = ctypes.c_void_p
h = k32.GetModuleHandleW("rocm-openblas.dll")
if h:
    blas = ctypes.CDLL("rocm-openblas.dll", handle=h)
    blas.openblas_get_num_threads.restype = ctypes.c_int
    blas.openblas_set_num_threads.argtypes = [ctypes.c_int]
    if blas_threads > 0:
        blas.openblas_set_num_threads(blas_threads)
if torch_threads > 0:
    torch.set_num_threads(torch_threads)
out["torch_threads"] = torch.get_num_threads()
out["openblas"] = blas.openblas_get_num_threads() if blas else None
out["config"] = torch.__config__.show()[:3000]
out["parallel"] = torch.__config__.parallel_info()[:1500]
def gflops(m, k, n, reps=10):
    a, b = torch.randn(m, k), torch.randn(k, n)
    for _ in range(3): a @ b
    t = time.perf_counter()
    for _ in range(reps): a @ b
    return round(2 * m * k * n * reps / (time.perf_counter() - t) / 1e9, 1)
with torch.inference_mode():
    out["gflops_4096"] = gflops(4096, 4096, 4096, 3)
    out["gflops_linear_1024x384x1536"] = gflops(1024, 384, 1536, 50)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("unsloth/bge-small-en-v1.5", device="cpu")
words = ("retrieval augmented generation splits documents into chunks and embeds each chunk with a small "
         "sentence transformer so later questions find the closest passages by cosine similarity ").split()
texts = [" ".join(words[(i + j) % len(words)] for j in range(110)) + f" #{i}" for i in range(512)]
model.encode(texts[:32], batch_size=32)
c0, w0, done = sum(me.cpu_times()[:2]), time.perf_counter(), 0
while time.perf_counter() - w0 < 25:
    model.encode(texts[done % 480: done % 480 + 32], batch_size=32); done += 32
wall = time.perf_counter() - w0
out["chunks_per_s"] = round(done / wall, 2)
out["busy_cores"] = round((sum(me.cpu_times()[:2]) - c0) / wall, 2)
if profile:
    from torch.profiler import profile as prof, ProfilerActivity
    with prof(activities=[ProfilerActivity.CPU]) as p:
        model.encode(texts[:32], batch_size=32)
    out["profile"] = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
print("CELL " + json.dumps(out), flush=True)
'''

NUMPY = r'''
import json, time, numpy as np
a = np.random.rand(4096, 4096).astype(np.float32); a @ a
t = time.perf_counter(); [a @ a for _ in range(3)]
print("CELL " + json.dumps({"numpy": np.__version__, "gflops_4096": round(2 * 4096**3 * 3 / (time.perf_counter() - t) / 1e9, 1)}))
'''


def sh(cmd, log, label, timeout=3600, **kw):
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, **kw)
    log.append({"step": label, "rc": r.returncode, "s": round(time.time() - t), "tail": (r.stdout + r.stderr)[-3000:]})
    print(f"[{label}] rc={r.returncode} {round(time.time() - t)}s", flush=True)
    return r


def venv(work, name, pkgs, log):
    py = work / name / "Scripts" / "python.exe"
    base = getattr(sys, "_base_executable", None) or sys.executable
    sh([base, "-m", "venv", str(work / name)], log, f"{name} venv")
    r = sh([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", *pkgs], log, f"{name} install")
    if r.returncode:
        raise SystemExit(f"{name} install failed")
    return py


def cell(py, work, script, args, log):
    path = work / "cell.py"
    path.write_text(script, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    env.update(HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", PYTHONUTF8="1")
    r = subprocess.run([str(py), str(path), *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=1200, env=env, cwd=str(work))
    for line in r.stdout.splitlines():
        if line.startswith("CELL "):
            return json.loads(line[5:])
    return {"error": f"rc={r.returncode}", "tail": (r.stdout + r.stderr)[-3000:]}


def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("AMD_CI_WORK", out_dir)) / "diag"; work.mkdir(parents=True, exist_ok=True)
    obs = {"cpu_count": os.cpu_count(), "build": [], "rocm": [], "cpu": [], "numpy": None}
    common = ["psutil", "numpy", "sentence-transformers"]
    try:
        rocm = venv(work, "v_rocm", ["--index-url", MULTIARCH, "--extra-index-url", "https://pypi.org/simple",
                                     "torch[device-gfx1151]==2.11.0+rocm7.14.0", *common], obs["build"])
        cpu = venv(work, "v_cpu", ["--index-url", "https://download.pytorch.org/whl/cpu", "--extra-index-url",
                                   "https://pypi.org/simple", "torch==2.11.0", *common], obs["build"])
        obs["numpy"] = cell(cpu, work, NUMPY, [], obs["build"])
        print("numpy", obs["numpy"], flush=True)
        for rep in range(2):
            for tt, bt in ((0, 0), (16, 1), (1, 16), (8, 8), (16, 16), (1, 1)):
                r = dict(cell(rocm, work, CELL, [str(tt), str(bt), "1" if (rep == 0 and (tt, bt) in ((0, 0), (16, 1))) else "0"], obs["build"]), req=[tt, bt], rep=rep)
                obs["rocm"].append(r)
                print("rocm", tt, bt, {k: r.get(k) for k in ("torch_threads", "openblas", "gflops_4096", "gflops_linear_1024x384x1536", "chunks_per_s", "busy_cores", "error")}, flush=True)
            for tt in (0, 1):
                r = dict(cell(cpu, work, CELL, [str(tt), "0", "1" if rep == 0 and tt == 0 else "0"], obs["build"]), req=[tt, 0], rep=rep)
                obs["cpu"].append(r)
                print("cpu", tt, {k: r.get(k) for k in ("torch_threads", "gflops_4096", "gflops_linear_1024x384x1536", "chunks_per_s", "busy_cores", "error")}, flush=True)
    except BaseException as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
    (out_dir / "diag.json").write_text(json.dumps(obs, indent=2), encoding="utf-8")
    for r in obs["rocm"][:2] + obs["cpu"][:1]:
        if r.get("profile"):
            print(r["req"], "\n", r["profile"], flush=True)
        if r.get("parallel"):
            print(r["parallel"], flush=True)


if __name__ == "__main__":
    main()
