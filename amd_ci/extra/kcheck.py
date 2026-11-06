#!/usr/bin/env python3
"""On-device check of Studio's eager Qwen-Image-2.1 q/k RMSNorm + RoPE kernel against the stock diffusers pair.

  PYTHONPATH=<studio>/studio/backend python q21_eager_kernel_check.py --out check.json

Real decode shape (1, 4096, 32, 128) bf16, real RoPE table from the diffusers module. Reports the probed fma pairing,
whether the kernel engaged, bitwise mismatches and the largest ulp distance, and median ms of each path (20 calls).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    args = ap.parse_args()
    res: dict = {}
    try:
        import torch
        from diffusers.models.transformers import transformer_qwenimage21 as qmod

        from core.inference import diffusion_qwenimage21_eager as e

        res["torch"] = torch.__version__
        res["hip"] = torch.version.hip
        dev = torch.device("cuda", 0)
        res["install"] = bool(e.install())
        res["fusion"] = e._fusion(dev)
        g = torch.Generator(device = "cpu").manual_seed(0)
        tokens, heads, hd = 4096, 32, 128
        res["shape"] = [1, tokens, heads, hd]
        x = (torch.randn(1, tokens, heads * hd, generator = g) * 3).to(dev, torch.bfloat16).unflatten(-1, (heads, hd))
        norm = qmod.RMSNorm(hd, eps = 1e-6).to(dev, torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_((torch.rand(hd, generator = g) + 0.5).to(torch.bfloat16))
        ang = torch.rand(tokens, hd // 2, generator = g) * 6.28
        freqs = torch.polar(torch.ones(tokens, hd // 2), ang).to(dev)

        def stock():
            return qmod.apply_rotary_emb_qwen(norm(x).to(x.dtype), freqs, use_real = False)

        with torch.no_grad():
            ref = stock()
            e._FAILED[0] = False
            got = e._norm_rope(x, norm, freqs)
            res["engaged"] = got is not None
            res["failed_flag"] = bool(e._FAILED[0])
            if got is not None:
                diff = got != ref
                res["mismatch"] = int(diff.sum())
                res["elements"] = int(ref.numel())
                ri = ref.view(torch.int16).to(torch.int32)
                gi = got.view(torch.int16).to(torch.int32)
                res["max_ulp"] = int((ri - gi).abs().max())
                res["max_abs"] = float((got.float() - ref.float()).abs().max())
                import hashlib

                res["fused_sha"] = hashlib.sha256(got.cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:16]
                res["stock_sha"] = hashlib.sha256(ref.cpu().view(torch.int16).numpy().tobytes()).hexdigest()[:16]
                res["device"] = torch.cuda.get_device_name(0)

                def bench(fn):
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    ts = []
                    for _ in range(20):
                        t0 = time.perf_counter()
                        fn()
                        torch.cuda.synchronize()
                        ts.append((time.perf_counter() - t0) * 1000)
                    return round(sorted(ts)[len(ts) // 2], 3)

                res["stock_ms"] = bench(stock)
                res["fused_ms"] = bench(lambda: e._norm_rope(x, norm, freqs))
        # gate + value projections: one N=24576 GEMM vs two N=12288 ones (the decode-step SwiGLU shape)
        import torch.nn.functional as F

        xs = (torch.randn(4096, 4096, generator = g) * 0.5).to(dev, torch.bfloat16)
        wg = (torch.randn(12288, 4096, generator = g) * 0.02).to(dev, torch.bfloat16)
        wp = (torch.randn(12288, 4096, generator = g) * 0.02).to(dev, torch.bfloat16)
        wf = torch.cat([wg, wp])
        with torch.no_grad():
            sep = torch.cat([F.linear(xs, wg), F.linear(xs, wp)], dim = -1)
            fus = F.linear(xs, wf)
            res["gate_up_mismatch"] = int((sep != fus).sum())
            res["gate_up_max_abs"] = float((sep.float() - fus.float()).abs().max())

            def bench2(fn):
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                ts = []
                for _ in range(15):
                    t0 = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
                return round(sorted(ts)[len(ts) // 2], 3)

            res["gate_up_two_ms"] = bench2(lambda: (F.linear(xs, wg), F.linear(xs, wp)))
            res["gate_up_fused_ms"] = bench2(lambda: F.linear(xs, wf))
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"[:1500]
        res["traceback"] = traceback.format_exc()[-3000:]
    with open(args.out, "w", encoding = "utf-8") as f:
        json.dump(res, f, indent = 1, default = str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
