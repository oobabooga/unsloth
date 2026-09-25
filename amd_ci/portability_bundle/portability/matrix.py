#!/usr/bin/env python3
"""Device x item matrix from every outputs/portability/<device>/results.json -> outputs/portability/MATRIX.md (+ .json).

Cell: status counts over the shapes/cases of that item (runs / fallback / refuses / fails / wrong), the geometric
mean speedup vs base eager and vs base compiled, and the worst error (rel RMS for GEMM/conv/compile, max abs for the
PR kernels, whose own verdict already compares against Inductor and eager).
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.environ.get("WORKSPACE") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(WS, "outputs", "portability")
ORDER = ["t4", "l4", "a100", "g4", "b200", "gfx1151-linux", "gfx1151-windows"]

# (item label, case, variant predicate, split by dtype)
ITEMS = [
    ("1 int8 torch._int_mm eager", "int8_gemm", lambda v: v == "intmm_eager", False),
    ("1 int8 torch._int_mm compiled", "int8_gemm", lambda v: v == "intmm_compiled", False),
    ("1 torchao W8A8 eager", "int8_gemm", lambda v: v == "torchao_eager", False),
    ("1 torchao W8A8 compiled", "int8_gemm", lambda v: v == "torchao_compiled", False),
    ("1 Triton quant + int8 mm (2 kernels)", "int8_gemm", lambda v: v == "triton_2k", False),
    ("1 Triton fused act-quant int8 mm", "int8_gemm", lambda v: v == "triton_fused", False),
    ("1 PR11801 _int8_linear", "int8_gemm", lambda v: v == "pr11801_int8_linear", False),
    ("2 PR11801 production gate", "h3vae_triton", lambda v: v == "production_gate", False),
] + [(f"2 PR11801 {k}", "h3vae_triton", (lambda kk: lambda v: v == kk)(k), False) for k in
     ("gn_stats", "gn_silu_pad", "add_residual", "add_rmsnorm", "qk_norm_rope", "swiglu", "quant_rows",
      "dequant_epilogue")] + [
    ("2 PR10731 nvfp4_bias_add kernel", "h3vae_triton", lambda v: v == "nvfp4_bias_add", False),
    ("2 PR10731 NVFP4 resolver", "nvfp4_gate", lambda v: v == "resolver", False),
    ("2 PR10731 fused bias engage + bitexact", "nvfp4_gate", lambda v: v.startswith("bias_add"), False),
    ("2 torchao NVFP4 linear", "nvfp4_gate", lambda v: v == "torchao_nvfp4", False),
    ("3 conv eager (cuDNN/MIOpen)", "conv_lowering", lambda v: v == "eager", True),
    ("3 conv max-autotune ATEN,TRITON", "conv_lowering", lambda v: v == "ma_aten_triton", True),
    ("3 conv Triton template only", "conv_lowering", lambda v: v == "ma_triton", True),
    ("4 Studio default tier", "studio_compile", lambda v: v == "default", True),
    ("4 Studio default + CUDA graph", "studio_compile", lambda v: v == "default+graph", True),
    ("4 Studio max tier", "studio_compile", lambda v: v == "max", True),
    ("4 Studio max + CUDA graph", "studio_compile", lambda v: v == "max+graph", True),
]


def _gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def cell(rows):
    if not rows:
        return None
    st = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    rel = [(r.get("err") or {}).get("rel_rms") for r in rows]
    rel = [x for x in rel if x is not None]
    mx = [(r.get("err") or {}).get("max_abs") for r in rows]
    mx = [x for x in mx if x is not None]
    return {"status": st, "n": len(rows), "x_eager": _gm([r.get("speedup_vs_eager") for r in rows]),
            "x_compiled": _gm([r.get("speedup_vs_compiled") for r in rows]),
            "worst_rel_rms": max(rel) if rel else None, "worst_max_abs": max(mx) if mx else None,
            "engaged": sum(1 for r in rows if r.get("engaged")),
            "notes": sorted({str(r.get("note", ""))[:160] for r in rows if r["status"] in ("fails", "wrong",
                                                                                          "refuses", "fallback")})[:3]}


def fmt(c):
    if c is None:
        return "-"
    s = c["status"]
    main = max(s, key = s.get)
    tag = main if len(s) == 1 else " ".join(f"{k} {v}" for k, v in sorted(s.items()))
    if len(s) == 1 and c["n"] > 1:
        tag += f" {c['n']}"
    parts = [tag]
    if c["worst_rel_rms"] is not None:
        parts.append(f"rel {c['worst_rel_rms']:.3g}")
    elif c["worst_max_abs"] is not None:
        parts.append(f"err {c['worst_max_abs']:.3g}")
    if c["x_eager"]:
        parts.append(f"{c['x_eager']:.2f}x eager")
    if c["x_compiled"]:
        parts.append(f"{c['x_compiled']:.2f}x comp")
    return ", ".join(parts)


def main(argv = None):
    devs = {}
    for d in sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []:
        p = os.path.join(OUT, d, "results.json")
        if os.path.exists(p) and not d.startswith("_"):
            with open(p, encoding = "utf-8") as f:
                devs[d] = json.load(f)
    order = [d for d in ORDER if d in devs] + [d for d in devs if d not in ORDER]
    table = {}
    labels = []
    for label, case, pred, by_dt in ITEMS:
        dts = sorted({r.get("dtype") for d in devs.values() for r in d["rows"] if r.get("case") == case
                      and pred(r.get("variant", ""))}) if by_dt else [None]
        for dt in dts:
            lab = f"{label} [{dt}]" if dt else label
            labels.append(lab)
            for d in order:
                rows = [r for r in devs[d]["rows"] if r.get("case") == case and pred(r.get("variant", "")) and
                        (dt is None or r.get("dtype") == dt)]
                table.setdefault(lab, {})[d] = cell(rows)
    L = ["# Portability matrix", "",
         "Cell: status (count), worst error (rel RMS vs fp32/fp64 reference, or max abs for the PR kernels), geomean "
         "speedup vs base eager and vs base compiled (base = bf16, fp16 on T4). Per-shape rows: "
         "`outputs/portability/<device>/report.md`.", ""]
    L.append("| item | " + " | ".join(order) + " |")
    L.append("|---|" + "---|" * len(order))
    for lab in labels:
        L.append(f"| {lab} | " + " | ".join(fmt(table[lab].get(d)) for d in order) + " |")
    L += ["", "## Devices", ""]
    for d in order:
        i = devs[d]["device"]
        L.append(f"- {d}: {i.get('name')} ({i.get('arch')}), torch {i.get('torch')}, triton {i.get('triton')}, "
                 f"torchao {i.get('torchao')}, cuda {i.get('cuda_runtime')}, hip {i.get('hip')}, SMs {i.get('sm_count')}, "
                 f"inductor big_gpu {i.get('inductor_big_gpu')}, {i.get('platform')}, "
                 f"run {devs[d].get('time')}")
    L += ["", "## Non-runs (first notes)", ""]
    for lab in labels:
        for d in order:
            c = table[lab].get(d)
            if c and c["notes"]:
                L.append(f"- {d} / {lab}: " + " || ".join(c["notes"]))
    with open(os.path.join(OUT, "MATRIX.md"), "w", encoding = "utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "MATRIX.json"), "w", encoding = "utf-8") as f:
        json.dump({"devices": order, "items": labels, "cells": table}, f, indent = 1, default = str)
    print("wrote", os.path.join(OUT, "MATRIX.md"))


if __name__ == "__main__":
    main(sys.argv[1:])
