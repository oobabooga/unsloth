#!/usr/bin/env python3
"""Probe: Studio's OpenBLAS default under a Windows job-object memory cap (#12374, PR #13048).

Observes only. Per state a plain venv (numpy + psutil + the pinned ROCm torch). Each cell is a fresh child that puts
itself in a new job with JOB_OBJECT_LIMIT_PROCESS_MEMORY (the limit #12374 reproduced against), then:
`studio` runs this checkout's configure_cpu_threads(); `unset` / `fixed8` set nothing / OPENBLAS_NUM_THREADS=8.
Then `import numpy`, a CPU matmul, `import torch`, a CPU matmul. Records the thread count chosen, the reported
headroom (head only), committed memory, and whether the process survived (OpenBLAS exits 1 when a buffer fails).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MULTIARCH_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"

CHILD = r'''
import ctypes, json, os, sys, time
from ctypes import wintypes
cap_mb, arm, backend = int(sys.argv[1]), sys.argv[2], sys.argv[3]
out = {"cap_mb": cap_mb, "arm": arm, "stage": "start"}
def emit():
    print("CELL " + json.dumps(out), flush=True)
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
if cap_mb:
    class Basic(ctypes.Structure):
        _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                    ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", wintypes.DWORD), ("f", ctypes.c_size_t),
                    ("g", wintypes.DWORD), ("h", wintypes.DWORD)]
    class Ext(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("Io", ctypes.c_ulonglong * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("p", ctypes.c_size_t), ("q", ctypes.c_size_t)]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    info = Ext(); info.Basic.LimitFlags = 0x100; info.ProcessMemoryLimit = cap_mb << 20
    out["job_set"] = bool(k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)))
    out["job_assigned"] = bool(k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()))
for k in ("OPENBLAS_NUM_THREADS", "UNSLOTH_CPU_THREADS", "OMP_NUM_THREADS"):
    os.environ.pop(k, None)
if arm == "fixed8":
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
if arm == "studio":
    sys.path.insert(0, backend)
    import utils.cpu_threads as ct
    if hasattr(ct, "_openblas_memory_headroom"):
        h = ct._openblas_memory_headroom()
        out["headroom_mb"] = None if h is None else round(h / 2**20)
    ct.configure_cpu_threads()
out["openblas_env"] = os.environ.get("OPENBLAS_NUM_THREADS")
import psutil
me = psutil.Process()
def priv():
    return round(me.memory_info().private / 2**20, 1)
out["stage"] = "import numpy"; emit()
import numpy as np
out["numpy_private_mb"] = priv(); out["numpy_threads"] = me.num_threads()
a = np.ones((1024, 1024)); a @ a
out["stage"] = "import torch"; emit()
import torch
b = torch.ones(512, 512); b @ b
out["torch_private_mb"] = priv()
k = ctypes.WinDLL("kernel32"); k.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]; k.GetModuleHandleW.restype = ctypes.c_void_p
h = k.GetModuleHandleW("rocm-openblas.dll")
if h:
    lib = ctypes.CDLL("rocm-openblas.dll", handle=h); lib.openblas_get_num_threads.restype = ctypes.c_int
    out["dll_threads"] = lib.openblas_get_num_threads()
out["stage"] = "done"; out["ok"] = True; emit()
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--gfx", default = "gfx1151")
    args, _ = ap.parse_known_args()
    work = Path(os.environ.get("AMD_CI_WORK") or args.out.parent) / f"memcap_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "cpu_count": os.cpu_count(), "cells": []}
    try:
        checkout = args.checkout.resolve()
        text = (checkout / "install.ps1").read_text(encoding = "utf-8", errors = "replace")
        tag = re.search(r'^\s*\$MultiArchTag\s*=\s*"([^"]+)"', text, re.M).group(1)
        ver = re.search(r'^\s*\$MultiArchTorchVersion\s*=\s*"([^"]+)"', text, re.M).group(1)
        venv = work / "venv"
        base = getattr(sys, "_base_executable", None) or sys.executable
        subprocess.run([base, "-m", "venv", str(venv)], check = True, timeout = 600)
        py = venv / "Scripts" / "python.exe"
        subprocess.run([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", "--index-url",
                        MULTIARCH_INDEX, "--extra-index-url", "https://pypi.org/simple",
                        f"torch[device-{args.gfx}]=={ver}+{tag}", "numpy", "psutil"], check = True, timeout = 3600)
        child = work / "memcap_child.py"
        child.write_text(CHILD, encoding = "utf-8")
        backend = str(checkout / "studio" / "backend")
        for rep in range(2):
            for cap in (0, 5000, 3500, 2500):
                for arm in ("studio", "unset", "fixed8"):
                    try:
                        r = subprocess.run([str(py), str(child), str(cap), arm, backend], capture_output = True,
                                           text = True, encoding = "utf-8", errors = "replace", timeout = 600,
                                           cwd = str(work))
                        stdout, stderr, rc = r.stdout, r.stderr, r.returncode
                    except subprocess.TimeoutExpired as e:
                        # A child starved under the cap can hang instead of exiting: a failed cell, not a dead probe.
                        stdout = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
                        stderr, rc = "timed out after 600 s", "hung"
                    cells = [json.loads(x[5:]) for x in stdout.splitlines() if x.startswith("CELL ")]
                    cell = cells[-1] if cells else {"cap_mb": cap, "arm": arm}
                    cell.update(rc = rc, rep = rep, ok = bool(cell.get("ok")) and rc == 0)
                    if not cell["ok"]:
                        cell["tail"] = (stdout + stderr)[-600:]
                    obs["cells"].append(cell)
                    print(f"[{args.state}] cap={cap} {arm}: ok={cell['ok']} env={cell.get('openblas_env')} "
                          f"headroom={cell.get('headroom_mb')} stage={cell.get('stage')} rc={rc}", flush = True)
    except BaseException as e:  # noqa: BLE001
        obs["probe_error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
