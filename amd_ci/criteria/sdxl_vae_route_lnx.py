#!/usr/bin/env python3
"""Criteria (Linux ROCm, regression leg): the SDXL LoRA trainer's training latents stay within 1% of a CPU fp32 encode
when the head routes the VAE's attention (judges probes/sdxl_vae_route_probe.py). Linux ROCm's fused kernels are
correct at head dim 512, so the base is accurate too; the head must not be worse."""

from __future__ import annotations

TITLE = "gfx1151 Linux: SDXL LoRA training latents vs a CPU fp32 encode, base versus head (routed VAE attention)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm", "linux", "nvidia", "discrete_gpu"]


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name}: training latents captured", "finite" in o,
                    f"{o.get('error') or ''} {o.get('torch')} {o.get('arch')} {o.get('platform')}"[:300]))
        out.append((f"{name}: on the GPU", bool(o.get("cuda")), str(o.get("arch"))))
    h = obs.get("head") or {}
    out.append(("head: VAE attention routed", (h.get("patched") or 0) > 0, str(h.get("patched"))))
    return out


def head_is_worse(base, head):
    bad = (not head.get("finite")) or head.get("rel_l2") is None or head["rel_l2"] > max(0.01, 2 * (base.get("rel_l2") or 0))
    return (bool(bad), f"rel L2 base {base.get('rel_l2')} head {head.get('rel_l2')}; patched head {head.get('patched')}")


def table(obs):
    rows = ["| state | platform | arch | finite | rel L2 | max abs | patched |", "|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        rows.append(f"| {name} | {o.get('platform')} | {o.get('arch')} | {o.get('finite')} | {o.get('rel_l2')} | "
                    f"{o.get('max_abs')} | {o.get('patched')} |")
    return "\n".join(rows)
