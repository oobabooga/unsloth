#!/usr/bin/env python3
"""Criteria (gfx1151, regression): the head's ROCm GELU(tanh) Triton helper builds and matches ATen bit for bit on
every finite bf16 input, and FLUX.1 on ROCm behaves as at the base (forward unchanged by the install, the CUDA-only
GELU producer never armed). Judges probes/flux1_gelu_probe.py."""

from __future__ import annotations

TITLE = "gfx1151: ConvRot GELU(tanh) helper on ROCm + FLUX.1 install, base versus head"
MODE = "regression"
NEEDS = ["gpu", "rocm", "linux", "windows", "windows_rocm_wddm", "nvidia", "discrete_gpu"]


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name}: on the ROCm GPU", bool(o.get("cuda")) and bool(o.get("hip")), f"{o.get('arch')} hip {o.get('hip')}"))
        out.append((f"{name}: kernels module built", bool(o.get("kernels_built")), str(o.get("kernel_error") or "")[:300]))
        out.append((f"{name}: FLUX.1 tiny forward ran", "flux1_forward_unchanged" in o, str(o.get("flux1_error") or "")[:300]))
    h = obs.get("head") or {}
    out.append(("head: GELU helper present and measured", h.get("gelu_bit_diffs") is not None, str(h.get("kernel_error") or "")[:300]))
    return out


def head_is_worse(base, head):
    why = []
    if head.get("gelu_bit_diffs") not in (0,):
        why.append(f"GELU helper differs from ATen on {head.get('gelu_bit_diffs')} of {head.get('gelu_n')} bf16 inputs "
                   f"(max abs {head.get('gelu_max_abs_diff')}, e.g. {head.get('gelu_diff_examples')})")
    if not head.get("flux1_forward_unchanged"):
        why.append("FLUX.1 forward changed by the install")
    if base.get("flux1_out_sum") is not None and head.get("flux1_out_sum") != base.get("flux1_out_sum"):
        why.append(f"FLUX.1 output differs from the base ({base.get('flux1_out_sum')} vs {head.get('flux1_out_sum')})")
    if head.get("gelu_armed"):
        why.append("GELU producer armed on ROCm")
    if base.get("kernels_built") and not head.get("kernels_built"):
        why.append("kernels module no longer builds")
    return (bool(why), "; ".join(why) or f"GELU helper bit-exact on {head.get('gelu_n')} inputs; FLUX.1 unchanged; "
            f"rotq_h16x compiles on HIP: base {base.get('h16x_compiles')} head {head.get('h16x_compiles')}")


def table(obs):
    rows = ["| state | arch | triton | GELU diffs | h16x compiles | FLUX.1 unchanged | GELU armed |", "|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        rows.append(f"| {name} | {o.get('arch')} | {o.get('triton')} | {o.get('gelu_bit_diffs', '-')} | {o.get('h16x_compiles')} | "
                    f"{o.get('flux1_forward_unchanged')} | {o.get('gelu_armed')} |")
    return "\n".join(rows)
