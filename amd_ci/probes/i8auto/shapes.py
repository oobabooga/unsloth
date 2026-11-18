#!/usr/bin/env python3
"""Per call, synthetic inputs at real shapes: Studio's INT8 attention (forced, the shipped gfx1151 config) vs torch
SDPA and flash_attn (where it imports), plus the module's own accuracy check verdict per head dim.

The q / k / v are the module's probe inputs (``probe_inputs``: RMS-normalised q / k with a learned gain, v with a
per-channel spread and offset; no attention sinks). Accuracy: rel L2 and cosine vs fp32 attention on 256 query rows.
Timing: interleaved rotating order, 2 warm calls each, CUDA events around the call, median of N rounds.

  python shapes.py --root STUDIO_BACKEND --out OUT.json [--rounds 15]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# (B, Sq, Sk, H, D): the dominant per-step self-attention call of each family at the size the e2e arms render, and
# MiniMax-H3-Image (opt-in everywhere, no e2e here: 135 GB of weights) at 1024 and 2048.
SHAPES = {
    "klein_1024": (1, 4608, 4608, 24, 128),
    "zimage_1024": (1, 4128, 4128, 30, 128),
    "qwen21_1024": (1, 4096, 4127, 32, 128),
    "wan22_5b_480p_33f": (1, 14040, 14040, 24, 128),
    "h3img_1024": (1, 1324, 1324, 56, 128),
    "h3img_2048": (1, 4396, 4396, 56, 128),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--rounds", type = int, default = 15)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.root).resolve()))
    os.environ["UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL"] = "1"
    rec: dict = {"rows": [], "checks": {}}
    try:
        import torch
        import torch.nn.functional as F

        from core.inference import diffusion_int8_attention as I8

        try:
            from flash_attn import flash_attn_func
        except Exception as exc:  # noqa: BLE001
            flash_attn_func = None
            rec["flash_import_error"] = repr(exc)[:200]
        dev = torch.device("cuda", 0)
        dtype = torch.bfloat16
        rec["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        rec["torch"] = torch.__version__
        rec["impl"] = {d: I8.implementation(0, d) for d in (64, 128)}
        for d in (64, 128):
            t0 = time.perf_counter()
            found = I8.verdict(dev, dtype, 24, d, None, "native", timed = True)
            torch.cuda.synchronize()
            rec["checks"][str(d)] = {**{k: v for k, v in found.items() if k != "long"},
                                     "long": found.get("long"), "s": round(time.perf_counter() - t0, 3)}

        def sdpa(q, k, v):
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

        def race(fns, rounds):
            for fn in fns.values():
                fn()
                fn()
            torch.cuda.synchronize()
            times = {n: [] for n in fns}
            names = list(fns)
            for r in range(rounds):
                for n in names[r % len(names):] + names[: r % len(names)]:
                    x, y = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                    x.record()
                    fns[n]()
                    y.record()
                    y.synchronize()
                    times[n].append(x.elapsed_time(y))
            return {n: {"median": round(sorted(t)[len(t) // 2], 4), "min": round(min(t), 4), "max": round(max(t), 4)}
                    for n, t in times.items()}

        for name, (B, Sq, Sk, H, D) in SHAPES.items():
            row: dict = {"shape": name, "B": B, "Sq": Sq, "Sk": Sk, "H": H, "D": D}
            try:
                with torch.no_grad():
                    q, k, v = I8.probe_inputs(dev, dtype, max(Sq, Sk), H, D, seed = Sk)
                    q = q[:, :Sq].contiguous()
                    k, v = k[:, :Sk].contiguous(), v[:, :Sk].contiguous()
                    t0 = time.perf_counter()
                    out = I8.int8_attention(q, k, v)
                    torch.cuda.synchronize()
                    row["first_call_s"] = round(time.perf_counter() - t0, 3)
                    rows = torch.linspace(0, Sq - 1, min(256, Sq), device = dev).round().long()
                    ref = I8.reference_attention(q[:, rows], k, v)
                    row["int8"] = I8.accuracy(out[:, rows], ref)
                    row["int8"]["rel_l2"] = float((out[:, rows].float() - ref).norm() / ref.norm())
                    nat = sdpa(q[:, rows], k, v)
                    row["sdpa"] = {"rel_l2": float((nat.float() - ref).norm() / ref.norm())}
                    fns = {"int8": lambda: I8.int8_attention(q, k, v), "sdpa": lambda: sdpa(q, k, v)}
                    if flash_attn_func is not None:
                        try:
                            fl = flash_attn_func(q[:, rows], k, v)
                            row["flash"] = {"rel_l2": float((fl.float() - ref).norm() / ref.norm())}
                            fns["flash"] = lambda: flash_attn_func(q, k, v)
                        except Exception as exc:  # noqa: BLE001
                            row["flash_error"] = repr(exc)[:200]
                    row["ms"] = race(fns, a.rounds)
                    row["triton_calls"] = I8.STATS.get("triton", 0)
            except Exception as exc:  # noqa: BLE001
                row["error"] = repr(exc)[:400]
            rec["rows"].append(row)
            print("SHAPE", json.dumps(row, default = str), flush = True)
            q = k = v = out = None
            torch.cuda.empty_cache()
        rec["verdict"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["verdict"] = "error"
        rec["error"] = repr(exc)[:1000]
        rec["traceback"] = traceback.format_exc()[-4000:]
    Path(a.out).write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
