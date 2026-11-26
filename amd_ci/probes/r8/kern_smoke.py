#!/usr/bin/env python3
"""Seconds-long gate for the kernel items: Studio's RDNA W8A8 kernels on this card (prepare + one small GEMM vs
reference_mm), the INT8 attention table, and what ROCm tooling this host has (rocprofv3, HIP CMake config, clang).
Writes one JSON via --out.

  kern_smoke.py --backend B --out F"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time


def rocm_roots():
    """Candidate ROCm roots: the venv's rocm-sdk wheels first (same runtime as torch), then the host install."""
    out = []
    try:
        r = subprocess.run([sys.executable, "-m", "rocm_sdk", "path", "--root"], capture_output = True, text = True,
                           timeout = 60)
        if r.returncode == 0 and r.stdout.strip():
            out.append(("rocm_sdk", r.stdout.strip()))
    except Exception:  # noqa: BLE001
        pass
    sp = [p for p in sys.path if p.endswith("site-packages")]
    for s in sp:
        for name in ("_rocm_sdk_devel", "_rocm_sdk_core"):
            d = os.path.join(s, name)
            if os.path.isdir(d):
                out.append((f"venv:{name}", d))
    for d in sorted(glob.glob("/opt/rocm*")):
        out.append((f"host:{d}", d))
    seen, rows = set(), []
    for label, root in out:
        real = os.path.realpath(root)
        if real in seen:
            continue
        seen.add(real)
        rows.append({"label": label, "root": root,
                     "hip_cmake": any(os.path.isfile(os.path.join(root, p)) for p in
                                      ("lib/cmake/hip/hip-config.cmake", "lib/cmake/hip/hipConfig.cmake")),
                     "clang": next((p for p in ("lib/llvm/bin/clang++", "bin/amdclang++", "llvm/bin/clang++",
                                                "bin/hipcc") if os.path.isfile(os.path.join(root, p))), None),
                     "rocprofv3": os.path.isfile(os.path.join(root, "bin", "rocprofv3")),
                     "hip_runtime_h": os.path.isfile(os.path.join(root, "include", "hip", "hip_runtime.h"))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    obs: dict = {"t0": time.time()}
    try:
        obs["rocm_roots"] = rocm_roots()
        obs["which"] = {n: shutil.which(n) for n in ("rocprofv3", "hipcc", "amdclang++", "cmake", "ninja", "git")}
        obs["nproc"] = os.cpu_count()
        obs["disk_free_gib"] = round(shutil.disk_usage(os.path.dirname(os.path.abspath(a.out))).free / 2 ** 30, 1)
        sys.path.insert(0, a.backend)
        import torch
        import triton

        from core.inference import diffusion_int8_attention as M
        from core.inference import diffusion_native_w8a8_rdna as R
        dev = torch.device("cuda", 0)
        obs.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None), triton = triton.__version__,
                   arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None))
        obs["prepare"] = R.prepare(dev)
        obs["rdna_state"] = R.state()
        g = torch.Generator(device = "cuda").manual_seed(0)
        m, k, n = 1040, 3072, 256
        xq = torch.randint(-127, 128, (m, k), generator = g, device = "cuda", dtype = torch.int8)
        wq = torch.randint(-127, 128, (n, k), generator = g, device = "cuda", dtype = torch.int8)
        xs = torch.rand(m, 1, generator = g, device = "cuda") * 1e-3 + 1e-4
        ws = torch.rand(n, generator = g, device = "cuda") * 1e-3 + 1e-4
        bias = torch.randn(n, generator = g, device = "cuda").to(torch.bfloat16)
        if obs["prepare"]:
            out = R.w8a8_mm(xq, xs, wq, ws, bias, torch.bfloat16)
            ref = R.reference_mm(xq, xs, wq, ws, bias, torch.bfloat16)
            obs["mm_eq_reference_mm"] = bool(torch.equal(out, ref))
            obs["mm_maxabs"] = float((out.float() - ref.float()).abs().max())
        obs["attn_table"] = M.rocm_kernel_config(M._gfx_arch(0), 128)
        obs["ok"] = bool(obs["prepare"]) and str(obs.get("arch") or "").startswith("gfx1151")
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
        obs["ok"] = False
    obs["seconds"] = round(time.time() - obs["t0"], 1)
    open(a.out, "w", encoding = "utf-8").write(json.dumps(obs, default = str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
