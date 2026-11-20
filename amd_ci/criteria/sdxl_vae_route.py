#!/usr/bin/env python3
"""Criteria: the SDXL LoRA trainer's training latents on gfx1151 (judges probes/sdxl_vae_route_probe.py).

base_shows_defect: the captured training latents are non-finite or more than 5% relative L2 off a CPU fp32 encode
of the same pixels (Windows ROCm's fused SDPA at the VAE's head dim 512).
head_is_fixed: the VAE's attention is routed (patched modules > 0) and the latents are within 1% of the reference.
"""

from __future__ import annotations

TITLE = "gfx1151: SDXL LoRA training latents vs a CPU fp32 encode (VAE attention at head dim 512)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm", "linux", "nvidia", "discrete_gpu"]


def _bad(o: dict) -> bool:
    return (not o.get("finite", False)) or (o.get("rel_l2") is None) or o["rel_l2"] > 0.05


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name}: training latents captured", "finite" in o,
                    f"{o.get('error') or ''} {o.get('torch')} {o.get('arch')} {o.get('platform')}"[:300]))
        out.append((f"{name}: on the GPU", bool(o.get("cuda")), str(o.get("arch"))))
    return out


def base_shows_defect(base):
    return (_bad(base), f"finite {base.get('finite')}, rel L2 {base.get('rel_l2')}, patched {base.get('patched')}")


def head_is_fixed(head):
    ok = head.get("finite") and head.get("rel_l2") is not None and head["rel_l2"] <= 0.01 and (head.get("patched") or 0) > 0
    return (bool(ok), f"finite {head.get('finite')}, rel L2 {head.get('rel_l2')}, patched {head.get('patched')}")


def table(obs):
    rows = ["| state | platform | arch | finite | rel L2 | max abs | patched |", "|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        rows.append(f"| {name} | {o.get('platform')} | {o.get('arch')} | {o.get('finite')} | {o.get('rel_l2')} | "
                    f"{o.get('max_abs')} | {o.get('patched')} |")
    return "\n".join(rows)
