#!/usr/bin/env python3
"""Layer micro-benchmark: is W8A8 through torch._int_mm faster than the weight-only int8 path on this GPU?

Times one Linear on the DiT's real shapes (M = one 1024px image's tokens plus text) four ways, eager and
under torch.compile: bf16 F.linear, weight-only int8 (dequantise + bf16 GEMM, what PR 11631 ships), W8A8
(per-row int8 activations x int8 weight via torch._int_mm), and the two bare GEMMs as the ceiling. Error is
relative Frobenius error against an fp32 reference on activations with a few outlier channels, as DiT
activations have. Observes only; prints one JSON object and writes it to --out.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--rows", type = int, default = 4160)
    ap.add_argument("--iters", type = int, default = 30)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.checkout) / "studio" / "backend"))
    import torch
    import torch.nn.functional as F

    import core.inference.diffusion_native_quant as nq

    dev = "cuda"
    obs: dict = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(0),
        "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
        "rows": args.rows,
        "shapes": {},
    }

    def bench(fn, x):
        for _ in range(5):
            fn(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(args.iters):
            a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            a.record()
            fn(x)
            b.record()
            torch.cuda.synchronize()
            times.append(a.elapsed_time(b))
        return round(statistics.median(times), 4)

    cls = nq.native_linear_class()
    for k, n in ((3072, 3072), (3072, 12288), (12288, 3072), (4096, 4096), (4096, 12288), (12288, 4096)):
        key = f"{args.rows}x{k}x{n}"
        rec: dict = {}
        obs["shapes"][key] = rec
        try:
            torch.manual_seed(0)
            lin = torch.nn.Linear(k, n).to(dev, torch.bfloat16)
            x = torch.randn(args.rows, k, device = dev, dtype = torch.bfloat16)
            x[:, torch.randperm(k, device = dev)[: max(1, k // 100)]] *= 20.0
            ref = x.float() @ lin.weight.float().t() + lin.bias.float()
            wo = cls(lin, "int8")
            w8 = cls(lin, "int8", act_int8 = True)
            w8r = cls(lin, "int8", act_int8 = True, rot_group = 256)
            arms = {"bf16": lin, "weight_only": wo, "w8a8": w8, "w8a8_rot": w8r}
            for name, mod in arms.items():
                with torch.no_grad():
                    y = mod(x).float()
                    rec[f"{name}_rel_err"] = round(((y - ref).norm() / ref.norm()).item(), 5)
                    rec[f"{name}_eager_ms"] = bench(mod, x)
                    try:
                        torch._dynamo.reset()
                        c = torch.compile(mod)
                        c(x)
                        rec[f"{name}_compiled_ms"] = bench(c, x)
                    except Exception as exc:  # noqa: BLE001
                        rec[f"{name}_compiled_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            xq = torch.randint(-127, 127, (args.rows, k), device = dev, dtype = torch.int8)
            wqt = w8.weight_q.t()
            with torch.no_grad():
                rec["gemm_int8_ms"] = bench(lambda a: torch._int_mm(a, wqt), xq)
                rec["gemm_bf16_ms"] = bench(lambda a: F.linear(a, lin.weight), x)
            flops = 2 * args.rows * k * n
            rec["gemm_int8_tops"] = round(flops / rec["gemm_int8_ms"] / 1e9, 2)
            rec["gemm_bf16_tflops"] = round(flops / rec["gemm_bf16_ms"] / 1e9, 2)
            del lin, wo, w8, w8r, arms, x, ref
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            rec["tail"] = traceback.format_exc().splitlines()[-6:]
        print(key, json.dumps(rec), flush = True)
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
