"""Item 2: the 9 Triton kernels of PR 11801 (video_minimax_h3_vae.py) and the NVFP4 bias kernel of PR 10731.

Delegates the numerics to ``scripts/triton_vs_inductor/verify.py`` (snapshotted into the bundle): each kernel is
compared with a float64 run of the stock Diffusers math, Inductor's compile of the same math, and stock eager at the
production dtype. One verify process per kernel, so a device fault in one kernel cannot hide the others.
Also records the production gate on this device (``cuda_fast_path_available``, the levers ``plan_h3_vae_levers``
would engage), since a kernel that is correct here but gated off is a fallback, not a failure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from portlib import FAILS, FALLBACK, RUNS, SKIPPED, WRONG  # noqa: E402

KERNELS = ["gn_stats", "gn_silu_pad", "add_residual", "add_rmsnorm", "qk_norm_rope", "swiglu", "quant_rows",
           "dequant_epilogue", "nvfp4_bias_add"]
INF = os.path.join("studio", "backend", "core", "inference")


def _status(verdict: str) -> str:
    v = (verdict or "").upper()
    if v.startswith("PASS") or v.startswith("EXPECTED"):
        return RUNS
    if v.startswith("FALLBACK"):
        return FALLBACK
    if v.startswith("FLAG"):
        return WRONG
    if v.startswith("ERROR"):
        return FAILS
    return RUNS if v == "N/A" else FAILS


def run(ctx):
    ws = os.environ.get("WORKSPACE") or os.path.abspath(os.path.join(HERE, "..", ".."))
    verify = ctx.bundle_path("verify", "verify.py") or os.path.join(ws, "scripts", "triton_vs_inductor", "verify.py")
    tree = ctx.bundle_path("trees", "h3") or os.path.join(ws, "wt_port_h3")
    btree = ctx.bundle_path("trees", "bias") or os.path.join(ws, "wt_verify_bias")
    if not os.path.exists(verify):
        ctx.row("verify", status = SKIPPED, note = f"verifier not found at {verify}")
        return ctx.rows

    # production gate, read from the module under test
    gate = {}
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("h3vae_gate", os.path.join(tree, INF, "video_minimax_h3_vae.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        gate["cuda_fast_path_available"] = bool(m.cuda_fast_path_available())
        gate["triton_kernels_build"] = m._kernels() is not None
        for tier in ("eager", "default", "max"):
            gate[f"levers_{tier}"] = list(m.plan_h3_vae_levers(tier, consumer_gpu = ctx.tag not in ("b200", "h100",
                                                                                                    "a100")))
    except Exception as exc:  # noqa: BLE001
        gate["error"] = f"{type(exc).__name__}: {exc}"[:300]
    gate_ok = gate.get("cuda_fast_path_available")
    ctx.row("production_gate", status = RUNS if gate_ok else FALLBACK, engaged = gate_ok,
            note = json.dumps(gate)[:600], gate = gate)

    mask = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    gpu = mask[0] if mask else "0"
    vout = os.path.join(ctx.out_dir, "h3vae_verify")
    vlog = os.path.join(ctx.log_dir, "inductor")
    kernels = os.environ.get("PORT_H3_KERNELS", ",".join(KERNELS)).split(",")
    for k in kernels:
        cmd = [sys.executable, "-u", verify, "--tree", tree, "--bias-tree", btree, "--kernels", k, "--gpu", gpu,
               "--iters", str(max(5, ctx.iters)), "--out-dir", vout, "--log-dir", vlog] + (
            ["--quick"] if ctx.quick else [])
        part = os.path.join(vout, "parts", f"{k}.json")
        if os.path.exists(part):
            os.remove(part)
        ctx.log("verify", k)
        try:
            p = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                               timeout = 1800, cwd = ctx.out_dir)
            with open(os.path.join(ctx.log_dir, f"verify_{k}.log"), "w", encoding = "utf-8") as f:
                f.write(p.stdout + "\n--- stderr ---\n" + p.stderr)
            rc = p.returncode
            tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
        except subprocess.TimeoutExpired:
            rc, tail = "timeout", ["timeout 1800s"]
        if not os.path.exists(part):
            ctx.row(k, status = FAILS, note = f"verify exit {rc}: {' / '.join(tail)}"[:400])
            continue
        with open(part, encoding = "utf-8") as f:
            data = json.load(f)
        for r in data.get("rows", []):
            o = r.get("ours") or {}
            err = {"max_abs": o.get("max_abs"), "mean_abs": o.get("mean_abs"), "max_rel": o.get("max_rel")} if o else None
            st = _status(r.get("verdict", ""))
            if r.get("error"):
                st = FAILS
            note = r.get("verdict", "")
            if r.get("error"):
                note += " " + str(r["error"])[:200]
            ind = r.get("inductor") or {}
            eag = r.get("eager") or {}
            ctx.row(k, shape = r.get("case", ""), dtype = r.get("dtype") or "-", status = st,
                    engaged = (st in (RUNS, WRONG)), err = err, ms = r.get("ours_ms"), eager_ms = r.get("eager_ms"),
                    compiled_ms = r.get("inductor_ms"), note = note,
                    inductor_max_abs = ind.get("max_abs"), eager_max_abs = eag.get("max_abs"),
                    ratio_max = r.get("ratio_max"), bits_vs_eager = r.get("bits_vs_eager"))
    return ctx.rows
