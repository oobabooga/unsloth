#!/usr/bin/env python3
"""Probe: the RDNA int8 W8A8 pieces at FLUX.3 Klein's 1024^2 shapes on this AMD GPU (observes only; judged by
criteria/rotq_bench.py).

Head state only (the base is the same commit plus nothing): Studio's ``diffusion_native_w8a8_rdna`` from the checkout.
  rotq   the rotate (ConvRot group 256) + row-quantise kernel at its shipped tile, a tile sweep, and torch.compile of
         the eager reference (``reference_rotq``); codes / scales vs the eager reference.
  gemm   the Triton W8A8 GEMM vs bf16 F.linear (torch's hipBLASLt, with Studio's shipped GEMM table loaded through
         TunableOp when the checkout has one) at the same shapes.
Timing: 3 warm calls, then N rounds with CUDA events, arms interleaved, median.

  python rotq_bench_probe.py --state head --checkout <tree> --out obs.json [--rounds 20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# (rows, K): the activations quantised per FLUX.3 Klein step (1024^2: 4096 image rows in the mode blocks, 4176 joint
# rows in the single blocks; K 3072 for Q / K / V / MLP-in / out, 9216 for the MLP down projection)
ROTQ_SHAPES = [(4176, 3072), (4176, 9216), (4096, 3072), (4096, 9216)]
# (rows, K, N): the GEMMs per block
GEMM_SHAPES = [(4176, 3072, 3072), (4176, 3072, 18432), (4176, 9216, 3072), (4096, 3072, 18432)]
SWEEP = [(bm, bk, w, st) for bm in (16, 32, 64, 128) for bk in (16, 32, 64) for w in (2, 4, 8) for st in (1, 2)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--rounds", type = int, default = 20)
    a = ap.parse_args()
    out = Path(a.out)
    obs: dict = {"state": a.state, "rows": []}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")

    if a.state != "head":
        obs["skipped"] = "measured on the head tree only"
        save()
        return 0
    backend = Path(a.checkout).resolve() / "studio" / "backend"
    table = None
    tables = sorted((backend / "core" / "inference" / "rocm_gemm_tables").glob("*.csv"))
    if tables:
        table = str(tables[0])
        os.environ.update({"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "0",
                           "PYTORCH_TUNABLEOP_FILENAME": table})
    obs["gemm_table"] = table
    sys.path.insert(0, str(backend))
    try:
        import torch
        import torch.nn.functional as F

        from core.inference import diffusion_native_w8a8_rdna as R

        obs["torch"] = torch.__version__
        obs["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        import triton
        obs["triton"] = triton.__version__
        dev = torch.device("cuda", 0)
        obs["prepare"] = R.prepare(dev)
        kern = R._kernels()
        from core.inference.diffusion_convrot import build_convrot_hadamard

        def timeit(fns: dict) -> dict:
            for fn in fns.values():
                for _ in range(3):
                    fn()
            torch.cuda.synchronize()
            ts = {k: [] for k in fns}
            names = list(fns)
            for r in range(a.rounds):
                for n in names[r % len(names):] + names[: r % len(names)]:
                    e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                    e0.record()
                    fns[n]()
                    e1.record()
                    e1.synchronize()
                    ts[n].append(e0.elapsed_time(e1))
            return {k: round(sorted(v)[len(v) // 2], 4) for k, v in ts.items()}

        def rotq_cfg(x, cfg):
            m, k = x.shape
            g = 256
            ng = k // g
            bm, bk, warps, stages = cfg
            if bm < ng:
                return None
            rows = bm // ng
            q = torch.empty((m, k), device = dev, dtype = torch.int8)
            s = torch.empty((m, 1), device = dev, dtype = torch.float32)
            h = build_convrot_hadamard(g, device = dev, dtype = x.dtype)
            R._launch_rotq(kern, x, h, q, s, m, ng, g, bm, bk, rows, g, warps, stages)
            return q, s

        ref_c = torch.compile(R.reference_rotq, dynamic = False)

        def plain_quant(x):
            s = x.float().abs().amax(1, keepdim = True).clamp(min = 1e-12) * (1.0 / 127.0)
            return (x.float() * (1.0 / s)).round().clamp(-127, 127).to(torch.int8), s

        plain_c = torch.compile(plain_quant, dynamic = False)
        gen = torch.Generator(device = "cpu").manual_seed(0)
        for m, k in ROTQ_SHAPES:
            x = (torch.randn(m, k, generator = gen) * 2).to(torch.bfloat16).to(dev)
            rq, rs = R.reference_rotq(x, 256)
            row: dict = {"kind": "rotq", "m": m, "k": k, "shipped": list(R._CFG["rotq"]), "variants": {}}

            def acc(q, s):
                d = (q.int() - rq.int()).abs()
                return {"max_code_diff": int(d.max()), "frac_diff": float((d > 0).float().mean()),
                        "scale_rel": float(((s - rs).abs() / rs).max())}

            try:
                row["shipped_acc"] = acc(*R._rotq_launch(x, 256))
            except Exception as exc:  # noqa: BLE001
                row["shipped_error"] = repr(exc)[:300]
            try:
                row["inductor_acc"] = acc(*ref_c(x, 256))
            except Exception as exc:  # noqa: BLE001
                row["inductor_error"] = repr(exc)[:300]
            fns = {"shipped": lambda: R._rotq_launch(x, 256), "inductor_ref": lambda: ref_c(x, 256),
                   "plain_quant_no_rotation": lambda: plain_c(x)}
            row["ms"] = timeit(fns)
            best = None
            for cfg in SWEEP:
                try:
                    got = rotq_cfg(x, cfg)
                    if got is None:
                        continue
                    ac = acc(*got)
                    ms = timeit({"c": lambda c = cfg: rotq_cfg(x, c)})["c"]
                    row["variants"][str(cfg)] = {"ms": ms, **ac}
                    if ac["max_code_diff"] <= 1 and (best is None or ms < best[1]):
                        best = (cfg, ms)
                except Exception as exc:  # noqa: BLE001
                    row["variants"][str(cfg)] = {"error": repr(exc)[:200]}
            row["best"] = best
            row["bytes_gb"] = round((m * k * 2 + m * k) / 1e9, 4)
            obs["rows"].append(row)
            print("ROTQ", m, k, row["ms"], best, flush = True)
            save()
        for m, k, n in GEMM_SHAPES:
            x = (torch.randn(m, k, generator = gen) * 2).to(torch.bfloat16).to(dev)
            w = (torch.randn(n, k, generator = gen) * 0.02).to(torch.bfloat16).to(dev)
            wq = torch.randint(-127, 128, (n, k), generator = gen, dtype = torch.int8).to(dev)
            ws = (torch.rand(n, generator = gen) * 1e-3 + 1e-4).to(dev)
            xq, xs = R._rotq_launch(x, 256)
            fns = {"w8a8_mm": lambda: R._mm_launch(xq, xs, wq, ws, None, torch.bfloat16),
                   "bf16_linear": lambda: F.linear(x, w)}
            row = {"kind": "gemm", "m": m, "k": k, "n": n, "ms": timeit(fns)}
            row["tflops"] = {kk: round(2 * m * n * k / (v * 1e-3) / 1e12, 2) for kk, v in row["ms"].items()}
            obs["rows"].append(row)
            print("GEMM", m, k, n, row["ms"], flush = True)
            save()
        obs["verdict"] = "ok"
    except Exception as exc:  # noqa: BLE001
        obs["verdict"] = "error"
        obs["error"] = repr(exc)[:1000]
        obs["traceback"] = traceback.format_exc()[-4000:]
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
