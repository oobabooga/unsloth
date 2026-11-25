#!/usr/bin/env python3
"""Item 5.8: Studio's RDNA W8A8 Triton GEMM, its own tile (``_mm_cfg``) vs the same tile with num_stages=1
(and the 64-wide tile with stages 1), interleaved in one process; the harness runs 3+ processes (A/A spread).
Every arm's output is compared to the Studio-tile output with torch.equal (int32 accumulate)."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a12common import env_info, interleave, write  # noqa: E402

# (name, K, N): Qwen-Image-2.1 (4096, mlp 12288), FLUX.3 klein (3072; fused single block 3072 -> 18432; down 9216),
# MiniMax-H3 (5376, ffn 14336), Wan 2.2 5B (3072, ffn 14336)
SHAPES = [("q21_qkv", 4096, 12288), ("q21_down", 12288, 4096), ("kl_fused", 3072, 18432), ("kl_down", 9216, 3072),
          ("h3_up", 5376, 14336), ("h3_down", 14336, 5376), ("wan_up", 3072, 14336), ("wan_down", 14336, 3072)]
MS = [4096, 8192, 16384]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--rounds", type = int, default = 7)
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    obs = {"rows": []}
    t0 = time.time()
    try:
        import torch

        from core.inference import diffusion_native_w8a8_rdna as R
        obs["env"] = env_info()
        obs["prepare"] = R.prepare(torch.device("cuda", 0))
        obs["studio_cfg"] = {k: list(v) for k, v in R._CFG.items()}
        kern = R._kernels()

        def launch(cfg, xq, wq, xs, ws, bias, out):
            bm, bn, bk, gm, warps, stages = cfg
            m, k = xq.shape
            n = wq.shape[0]
            kern.w8a8_mm[(kern.triton.cdiv(m, bm) * kern.triton.cdiv(n, bn),)](
                xq, wq, xs, ws, bias, out, m, n, k, HAS_BIAS = True, EVEN_K = k % bk == 0, BLOCK_M = bm,
                BLOCK_N = bn, BLOCK_K = bk, GROUP_M = gm, num_warps = warps, num_stages = stages,
                enable_fp_fusion = False)
        for name, k, n in SHAPES:
            for m in MS:
                g = torch.Generator(device = "cuda").manual_seed(m + k + n)
                xq = torch.randint(-127, 128, (m, k), generator = g, device = "cuda", dtype = torch.int8)
                wq = torch.randint(-127, 128, (n, k), generator = g, device = "cuda", dtype = torch.int8)
                xs = torch.rand(m, generator = g, device = "cuda") * 1e-3 + 1e-4
                ws = torch.rand(n, generator = g, device = "cuda") * 1e-3 + 1e-4
                bias = torch.randn(n, generator = g, device = "cuda").to(torch.bfloat16)
                st_cfg = tuple(R._mm_cfg(m, k))
                cfgs = {"studio": st_cfg, "s1": st_cfg[:5] + (1,)}
                if st_cfg[0] != 64:
                    cfgs["t64_s1"] = (64, 128, 128, 8, 4, 1)
                outs, fns, row = {}, {}, {"shape": name, "m": m, "k": k, "n": n, "cfgs": {x: list(y) for x, y in cfgs.items()}}
                for arm, cfg in cfgs.items():
                    o = torch.empty((m, n), device = "cuda", dtype = torch.bfloat16)
                    try:
                        launch(cfg, xq, wq, xs, ws, bias, o)
                        torch.cuda.synchronize()
                        outs[arm] = o
                        fns[arm] = (lambda cfg = cfg, o = o: launch(cfg, xq, wq, xs, ws, bias, o))
                    except Exception as exc:  # noqa: BLE001
                        row.setdefault("errors", {})[arm] = repr(exc)[:300]
                row["exact"] = {arm: bool(torch.equal(o, outs["studio"])) for arm, o in outs.items()}
                res = interleave(fns, a.rounds)
                row["med_ms"] = {arm: res[arm]["med"] for arm in fns}
                row["ms"] = {arm: res[arm]["ms"] for arm in fns}
                obs["rows"].append(row)
                del xq, wq, outs, fns
                torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    obs["seconds"] = round(time.time() - t0, 1)
    write(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
