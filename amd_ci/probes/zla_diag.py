#!/usr/bin/env python3
"""Driver: localise the gfx1151 illegal access in the zoo-vendored gated-delta (FLA) kernels.

Runs zla_cell.py cells one process each (a HIP illegal access kills the context), records rc,
the cell JSON and the last launches from each cell's launch log. Adaptive: the faulting kernel
found in the first crashing op cell gets one cell per autotune config (pinned, serialized).
Observes only; summary JSON via --out.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELL = HERE / "zla_cell.py"


def run_cell(py, name, cell, out_dir, zoo, extra_env = None, serialize = True, timeout = 1500):
    d = out_dir / name
    d.mkdir(parents = True, exist_ok = True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(zoo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["UNSLOTH_IS_PRESENT"] = "1"
    env["FLA_CACHE_RESULTS"] = "0"
    env["UNSLOTH_COMPILE_LOCATION"] = str(d / "compiled")
    env["PYTHONUNBUFFERED"] = "1"
    for k in ("ZLA_PIN", "ZLA_UPSTREAM_FLA", "AMD_SERIALIZE_KERNEL", "UNSLOTH_DISABLE_VENDORED_FLA", "AMDGCN_USE_BUFFER_OPS"):
        env.pop(k, None)
    if serialize:
        env["AMD_SERIALIZE_KERNEL"] = "3"
    env.update(extra_env or {})
    cmd = [py, str(CELL), "--cell", cell, "--out", str(d / "cell.json"), "--launch-log", str(d / "launches.jsonl"),
           "--asm-dir", str(out_dir / "asm")]
    t0 = time.time()
    with open(d / "log.txt", "wb") as fh:
        try:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env, timeout = timeout, cwd = str(d)).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rec = {"name": name, "cell": cell, "rc": rc, "seconds": round(time.time() - t0, 1), "env": extra_env or {}, "serialize": serialize}
    try:
        rec["json"] = json.loads((d / "cell.json").read_text(encoding = "utf-8"))
    except Exception as e:  # noqa: BLE001
        rec["json_err"] = repr(e)
    try:
        lines = (d / "launches.jsonl").read_text(encoding = "utf-8").splitlines()
        rec["n_launch_lines"] = len(lines)
        rec["last_launches"] = [json.loads(x) for x in lines[-8:]]
    except Exception as e:  # noqa: BLE001
        rec["launch_err"] = repr(e)
    try:
        log = (d / "log.txt").read_text(encoding = "utf-8", errors = "replace")
        rec["illegal_access"] = ("illegal memory access" in log) or ("hipErrorIllegalAddress" in log) or ("Code: 700" in log)
        rec["log_tail"] = log[-2500:]
    except Exception:  # noqa: BLE001
        pass
    rec["crashed"] = bool(rec.get("illegal_access")) or (rc not in (0,))
    return rec


def faulting_kernel(rec):
    """Last kernel launched (or benchmarked) before the fault, from the launch log."""
    for item in reversed(rec.get("last_launches") or []):
        if "k" in item:
            return item["k"], item.get("cfg")
        if "bench" in item:
            return item["bench"], item.get("config")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--zoo", required = True, type = Path)
    ap.add_argument("--upstream-fla", type = Path)
    ap.add_argument("--out-dir", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    a = ap.parse_args()
    a.out_dir.mkdir(parents = True, exist_ok = True)
    res = {"zoo": str(a.zoo), "cells": []}

    def save():
        a.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    def add(rec):
        res["cells"].append(rec)
        save()
        print(f"[zla] {rec['name']}: rc={rec['rc']} crashed={rec['crashed']} {rec['seconds']}s "
              f"last={faulting_kernel(rec)} steps={(rec.get('json') or {}).get('steps')}", flush = True)
        return rec

    env_rec = add(run_cell(a.python, "env", "env", a.out_dir, a.zoo, serialize = False))
    tuners = (env_rec.get("json") or {}).get("autotuners") or {}

    ops = [
        "op:float32:3:6:32:128:128",
        "op:bfloat16:3:6:32:128:128",
        "op:float32:1:6:32:128:128",
        "op:float32:3:64:32:128:128",
        "op:float32:2:200:16:128:128",
        "op:bfloat16:2:200:16:128:128",
        "op:float32:3:6:4:64:64",
        "op:bfloat16:3:6:4:64:64",
    ]
    add(run_cell(a.python, "model_fp32", "model:float32", a.out_dir, a.zoo))
    add(run_cell(a.python, "model_bf16", "model:bfloat16", a.out_dir, a.zoo))
    add(run_cell(a.python, "model_fp32_torchpath", "model:float32", a.out_dir, a.zoo,
                 extra_env = {"UNSLOTH_DISABLE_VENDORED_FLA": "1"}))
    op_recs = []
    for i, spec in enumerate(ops):
        op_recs.append(add(run_cell(a.python, f"op{i}", spec, a.out_dir, a.zoo)))

    crashing = [r for r in op_recs if r["crashed"]]
    res["crashing_ops"] = [r["cell"] for r in crashing]
    if crashing:
        first = crashing[0]
        kname, kcfg = faulting_kernel(first)
        res["fault_kernel"] = {"name": kname, "cfg": kcfg, "cell": first["cell"]}
        save()
        configs = tuners.get(kname) or []
        res["fault_kernel"]["configs"] = configs
        for idx in range(len(configs)):
            add(run_cell(a.python, f"pin_{kname}_{idx}", first["cell"], a.out_dir, a.zoo, extra_env = {"ZLA_PIN": f"{kname}={idx}"}))
        # The same pins in bf16 at a training shape too, when that shape crashed.
        for r in crashing[1:3]:
            k2, _ = faulting_kernel(r)
            res.setdefault("other_fault_kernels", []).append({"cell": r["cell"], "kernel": k2})
        for knob in ("0", "1"):
            add(run_cell(a.python, f"bufops{knob}", first["cell"], a.out_dir, a.zoo, extra_env = {"AMDGCN_USE_BUFFER_OPS": knob}))
        add(run_cell(a.python, "noserialize", first["cell"], a.out_dir, a.zoo, serialize = False))
        if a.upstream_fla and a.upstream_fla.is_dir():
            add(run_cell(a.python, "upstream_fla", first["cell"], a.out_dir, a.zoo,
                         extra_env = {"ZLA_UPSTREAM_FLA": str(a.upstream_fla)}))
    elif a.upstream_fla and a.upstream_fla.is_dir():
        add(run_cell(a.python, "upstream_fla", ops[0], a.out_dir, a.zoo, extra_env = {"ZLA_UPSTREAM_FLA": str(a.upstream_fla)}))
    res["done"] = True
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
