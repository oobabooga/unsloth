#!/usr/bin/env python3
"""Diagnostic (regression mode, never judges the PR): records which dW variants lose the existing
gradient on gfx1151. Gates: ran on ROCm and produced every DIAG line at both states."""
from __future__ import annotations
TITLE = "Diagnostic: MoE dW accumulate cancellation on gfx1151 (per variant)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "windows"]
N_VARIANTS = 28


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        h = o.get("header") or {}
        out.append((f"{name} ran on ROCm", bool(h.get("hip")), str(h)[:300] or str(o.get("stderr_tail"))[-300:]))
        out.append((f"{name} produced every variant", len(o.get("diag") or []) == N_VARIANTS,
                    f"{len(o.get('diag') or [])} lines"))
    return out


def table(obs):
    rows = ["| state | kernel | acc | BLOCK_M | big | rows | ok | grad values | ref | error |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for d in (obs.get(name) or {}).get("diag") or []:
            rows.append(f"| {name} | {d['kernel']} | {d['acc']} | {d['bm']} | {d['big']} | {d['rows']} | {d.get('ok')} "
                        f"| {d.get('grad_unique')} | {d.get('ref_unique')} | {d.get('error', '')} |")
    return "\n".join(rows)


def head_is_worse(base, head):
    bad = [d for d in head.get("diag") or [] if not d.get("ok")]
    return False, f"{len(bad)} of {len(head.get('diag') or [])} head variants lose the existing gradient (diagnostic only)"
