#!/usr/bin/env python3
"""Criteria (gfx1151, regression + measurement): the head's VAE upsample fold stays off by default on ROCm (MIOpen
unmeasured) and, forced, matches the 2x path at rounding level; the table reports MIOpen's fold vs 2x chain timings so
main can decide whether to add gfx1151 to the fold's cards. Judges probes/vae_fold_probe.py."""

from __future__ import annotations

TITLE = "gfx1151: VAE upsample fold (ConvTranspose2d k4 s2) vs nearest 2x + conv3x3 on MIOpen"
MODE = "regression"
NEEDS = ["gpu", "rocm", "linux", "windows", "windows_rocm_wddm", "nvidia", "discrete_gpu", "mps"]


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name}: on the ROCm GPU", bool(o.get("cuda")) and bool(o.get("hip")), f"{o.get('arch')} hip {o.get('hip')}"))
        out.append((f"{name}: shapes measured", len(o.get("shapes") or {}) == 6, str(o.get("error") or "")[:300]))
    h = obs.get("head") or {}
    out.append(("head: fold present", bool(h.get("has_fold")), str(h.get("tree_error") or "")[:300]))
    return out


def head_is_worse(base, head):
    why = []
    if head.get("fold_default_on"):
        why.append("the fold arms by default on ROCm (unmeasured there)")
    m = head.get("forced_max_abs_vs_2x")
    if m is None or m > 0.25:
        why.append(f"forced fold vs 2x path max abs {m}")
    for k, v in (head.get("shapes") or {}).items():
        if not v.get("finite"):
            why.append(f"{k}: non-finite fold output")
    return (bool(why), "; ".join(why) or "fold off by default on ROCm; forced fold within rounding of the 2x path")


def table(obs):
    rows = ["| state | shape | 2x chain ms | fold ms | fold speedup | max err chain / fold vs fp64 |", "|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for k, v in sorted((o.get("shapes") or {}).items()):
            rows.append(f"| {name} | {k} | {v['chain_ms']} | {v['fold_ms']} | {v['x']} | {v['chain_err']:.3g} / {v['fold_err']:.3g} |")
    return "\n".join(rows)
