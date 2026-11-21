#!/usr/bin/env python3
"""Observe hipBLASLt GEMM timings on this host for the DiT shapes in Studio's shipped TunableOp table(s), default pick vs
TunableOp's tuned pick, and write the tuned table with THIS host's validators (torch / HIP / hipBLASLt / rocBLAS / arch).

  python gemm_table_probe.py --tables DIR_OR_CSV [...] --out obs.json --table-out tuned.csv [--iters 30]

Observes only (no verdict). Per shape: default ms (TunableOp off), tuned ms (the solution TunableOp picked here),
max |tuned - default| / max |default| on the same inputs, tuning seconds. Also whether every shipped table's
validators equal this host's (shipped tables apply only on an exact match). Writes JSON via --out, never stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path


def parse_tables(paths):
    rows, vals = [], {}
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
        for f in files:
            v = {}
            for line in f.read_text(encoding = "utf-8").splitlines():
                parts = line.strip().split(",")
                if len(parts) >= 3 and parts[0] == "Validator":
                    v[parts[1]] = ",".join(parts[2:])
                elif len(parts) >= 2 and parts[0].startswith("GemmTunableOp_"):
                    key = (parts[0], parts[1])
                    if key not in [(r[0], r[1]) for r in rows]:
                        rows.append((parts[0], parts[1], f.name))
            vals[f.name] = v
    return rows, vals


def shape_args(op: str, sig: str):
    """torch.cuda.tunable._process_single_offline_gemm's parse: (m, n, k, lda, ldb, ldc, transA, transB, dtype)."""
    import torch

    _, data_type, layout = op.split("_")
    dtype = {"BFloat16": torch.bfloat16, "Half": torch.half, "Float": torch.float32}[data_type]
    transB = layout[0] == "T"
    transA = layout[1] == "T"
    t = sig.split("_")
    n, m, k = (int(g) for g in t[1:4])
    ldb, lda, ldc = (int(g) for g in t[5:8])
    return m, n, k, lda, ldb, ldc, transA, transB, dtype


def make(m, n, k, lda, ldb, ldc, transA, transB, dtype):
    import torch
    from torch.cuda import tunable as T

    sub = not all(x in (n, m, k) for x in (lda, ldb, ldc))  # torch 2.11 offline-tuning rule
    return T._create_matrices(m, n, k, lda, ldb, ldc, transA, transB, dtype, "cuda:0", subMatrix = sub)


def timed(fn, iters):
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts), (max(ts) - min(ts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs = "+", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--table-out", required = True)
    ap.add_argument("--iters", type = int, default = 30)
    ap.add_argument("--max-shapes", type = int, default = 0)
    a = ap.parse_args()
    obs = {"host": {"os": platform.system(), "python": platform.python_version()}, "shapes": [], "errors": []}
    out = Path(a.out)

    def flush():
        out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")

    try:
        import torch
        from torch.cuda import tunable as T

        props = torch.cuda.get_device_properties(0)
        obs["host"].update(torch = torch.__version__, hip = getattr(torch.version, "hip", None),
                           arch = str(getattr(props, "gcnArchName", "")), device = props.name)
        rows, shipped = parse_tables(a.tables)
        if a.max_shapes:
            rows = rows[: a.max_shapes]
        T.enable(True)
        T.tuning_enable(False)
        T.set_filename(str(Path(a.table_out).with_suffix(".work.csv")), insert_device_ordinal = False)
        T.set_max_tuning_duration(30)
        T.set_max_tuning_iterations(20)
        torch.mm(torch.randn(64, 64, device = "cuda", dtype = torch.bfloat16),
                 torch.randn(64, 64, device = "cuda", dtype = torch.bfloat16))  # primes the validators
        host_val = {str(k): str(v) for k, v in (T.get_validators() or ())}
        obs["validators_here"] = host_val
        obs["shipped"] = {n: {"validators": v, "matches_here": bool(v) and v == host_val} for n, v in shipped.items()}
        flush()
        for op, sig, src in rows:
            rec = {"op": op, "shape": sig, "table": src}
            try:
                args = shape_args(op, sig)
                A, B = make(*args)
                T.enable(False)
                ref = torch.mm(A, B).float()
                rec["default_ms"], rec["default_spread_ms"] = timed(lambda: torch.mm(A, B), a.iters)
                T.enable(True)
                T.tuning_enable(True)
                t0 = time.time()
                torch.mm(A, B)
                torch.cuda.synchronize()
                rec["tune_s"] = round(time.time() - t0, 2)
                T.tuning_enable(False)
                got = torch.mm(A, B).float()
                rec["rel_err"] = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
                rec["tuned_ms"], rec["tuned_spread_ms"] = timed(lambda: torch.mm(A, B), a.iters)
                rec["speedup"] = rec["default_ms"] / rec["tuned_ms"] if rec["tuned_ms"] else None
                del A, B, ref, got
            except Exception as exc:  # noqa: BLE001 - recorded, the next shape still runs
                rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
            obs["shapes"].append(rec)
            flush()
        res = list(T.get_results() or ())
        lines = [f"Validator,{k},{v}" for k, v in (T.get_validators() or ())]
        lines += [",".join(str(x) for x in r) for r in res]
        Path(a.table_out).write_text("\n".join(lines) + "\n", encoding = "utf-8")
        obs["table_rows"] = len(res)
        obs["table_text"] = "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        obs["errors"].append(f"{type(exc).__name__}: {exc}"[:800])
    flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
