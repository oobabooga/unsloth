#!/usr/bin/env python3
"""carbon6 #1 E2E arm: e2e_arm.py (same argv) with Studio's RDNA INT8 GEMM writing into a row-pitch N+128 buffer and
returning the [M, N] view, no copy. Patches ``_mm_launch`` (allocation) and ``_launch_mm`` (the strided copy of
Studio's kernel, kern_gemm.strided_kernel); e2e_arm's own launch counter wraps the patched ``_launch_mm``, so
rdna_mm counts prove engagement. Adds ``pad_launches`` / ``pad_arm`` to the arm JSON at the end."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAD = 128


def main():
    root = sys.argv[sys.argv.index("--root") + 1]
    sys.path.insert(0, str(Path(root).resolve()))
    sys.path.insert(0, str(HERE))
    import torch

    from core.inference import diffusion_native_w8a8_rdna as R
    from kern_gemm import strided_kernel
    skern = strided_kernel()
    import triton
    count = {"n": 0}

    def _mm_launch(xq, xs, wq, ws, bias, out_dtype):
        xq, xs, wq, ws = xq.contiguous(), xs.contiguous(), wq.contiguous(), ws.contiguous()
        bias = bias.contiguous() if bias is not None else None
        m, k = xq.shape
        n = wq.shape[0]
        out = torch.empty((m, n + PAD), device = xq.device, dtype = out_dtype)[:, :n]
        if m == 0:
            return out
        if R._STATE.get("broken"):
            return R.reference_mm_stock(xq, xs, wq, ws, bias, out_dtype)
        R._launch_mm(xq, xs, wq, ws, bias, out, m, n, k)
        return out

    def _launch_mm(xq, xs, wq, ws, bias, out, m, n, k):
        bm, bn, bk, gm, warps, stages = R._mm_cfg(m, k)
        count["n"] += 1
        with torch.cuda.device(xq.device):
            skern[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
                xq, wq, xs, ws, bias if bias is not None else ws, out, m, n, k, out.stride(0),
                HAS_BIAS = bias is not None, EVEN_K = k % bk == 0, BLOCK_M = bm, BLOCK_N = bn, BLOCK_K = bk,
                GROUP_M = gm, num_warps = warps, num_stages = stages, enable_fp_fusion = False)
    R._mm_launch, R._launch_mm = _mm_launch, _launch_mm
    out_dir = Path(sys.argv[sys.argv.index("--out") + 1])
    tag = sys.argv[sys.argv.index("--tag") + 1]
    import os
    real_exit = os._exit

    def _exit(code):
        p = out_dir / f"{tag}.json"
        try:
            rec = json.loads(p.read_text(encoding = "utf-8"))
            rec["pad_launches"] = count["n"]
            rec["pad_arm"] = True
            p.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")
        except Exception:  # noqa: BLE001
            pass
        real_exit(code)
    os._exit = _exit
    sys.argv = [str(HERE / "e2e_arm.py"), *sys.argv[1:]]
    runpy.run_path(str(HERE / "e2e_arm.py"), run_name = "__main__")


if __name__ == "__main__":
    main()
