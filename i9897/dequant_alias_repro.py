"""Compile diffusers' GGUF dequant on real FLUX.2-klein Q2_K / Q4 weights under a given inductor config; report
aliasing errors, generated-code markers, graph counts and steady-state time (eager vs compiled).

  python dequant_alias_repro.py --gguf PATH --out OUT.json [--memory-planning 0|1] [--n 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--memory-planning", type = int, default = 1)
    ap.add_argument("--n", type = int, default = 8, help = "weights (distinct quant types first)")
    ap.add_argument("--iters", type = int, default = 20)
    args = ap.parse_args()

    import torch
    import torch._inductor.config as ic
    from torch._dynamo.utils import counters

    res: dict = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(0),
        "memory_planning": bool(args.memory_planning),
        "error_on_alias_env": os.environ.get("TORCHINDUCTOR_ERROR_ON_CUSTOM_OP_ALIASING"),
    }
    ic.memory_planning = bool(args.memory_planning)
    ic.memory_pool = "none"

    import gguf
    from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor

    reader = gguf.GGUFReader(args.gguf)
    picked, seen = [], set()
    for t in reader.tensors:
        qt = t.tensor_type
        if qt.name in ("F32", "F16", "BF16") or len(t.shape) < 2:
            continue
        if qt.name not in seen or len(picked) < args.n:
            if qt.name in seen and len([p for p in picked if p[0] == qt.name]) >= 2:
                continue
            seen.add(qt.name)
            raw = torch.from_numpy(t.data.copy()).cuda()
            picked.append((qt.name, t.name, GGUFParameter(raw, quant_type = qt)))
        if len(picked) >= args.n:
            break
    res["weights"] = [(q, n, list(p.shape)) for q, n, p in picked]

    eager = dequantize_gguf_tensor
    compiled = torch.compile(eager, dynamic = True)
    rows = []
    for q, name, p in picked:
        row = {"quant": q, "name": name}
        try:
            ref = eager(p)
            out = compiled(p)
            torch.cuda.synchronize()
            row["max_abs_diff"] = float((out.float() - ref.float()).abs().max())
            for label, fn in (("eager", eager), ("compiled", compiled)):
                for _ in range(3):
                    fn(p)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    fn(p)
                torch.cuda.synchronize()
                row[f"{label}_us"] = round((time.perf_counter() - t0) / args.iters * 1e6, 1)
            row["ok"] = True
        except Exception as exc:  # noqa: BLE001
            row["ok"] = False
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)[:400]
            row["tb"] = traceback.format_exc()[-1500:]
        rows.append(row)
    res["rows"] = rows
    res["counters"] = {k: dict(v) for k, v in counters.items() if k in ("stats", "graph_break", "inductor", "frames")}
    with open(args.out, "w", encoding = "utf-8") as fh:
        json.dump(res, fh, indent = 1, default = str)
    print(json.dumps({k: res[k] for k in ("torch", "memory_planning")}),
          [(r["quant"], r["ok"], r.get("error_type"), r.get("eager_us"), r.get("compiled_us")) for r in rows])
    return 0


if __name__ == "__main__":
    sys.exit(main())
