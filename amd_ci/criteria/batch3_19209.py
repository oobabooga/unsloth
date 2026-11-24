#!/usr/bin/env python3
"""PR 19209: the refactored ROCm flash probe byte estimate must equal the base's, key for key, on the real card."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import base_gates, states  # noqa: E402

TITLE = "PR 19209: _rocm_flash_probe_bytes base versus head (gfx1151)"
MODE = "regression"
NEEDS = ["rocm", "gpu"]


def gates(obs):
    out = base_gates(obs)
    for name, o in states(obs).items():
        out.append((f"{name}: every grid point computed", bool(o.get("bytes")) and not o.get("errors"),
                    f"{len(o.get('bytes') or {})} points, errors {list((o.get('errors') or {}))[:3]}"))
        if name != "base":
            out.append((f"{name}: carries block_activation_bytes", bool(o.get("has_block_activation_bytes")), ""))
    return out


def table(obs):
    rows = ["| state | points | fused-SDPA points | non-fused points | replica_activation_mib (3072/24x128, 4096 tok) |",
            "|---|---|---|---|---|"]
    for name, o in states(obs).items():
        rows.append(f"| {name} | {len(o.get('bytes') or {})} | {o.get('fused_true')} | {o.get('fused_false')} "
                    f"| {o.get('replica_activation_mib_flux_like_4096', '-')} |")
    return "\n".join(rows)


def head_is_worse(base, head):
    b, h = base.get("bytes") or {}, head.get("bytes") or {}
    diff = [k for k in sorted(set(b) | set(h)) if b.get(k) != h.get(k)]
    if diff:
        return True, f"{len(diff)} grid points differ, e.g. " + "; ".join(f"{k}: {b.get(k)} -> {h.get(k)}" for k in diff[:4])
    return False, f"all {len(b)} grid points identical ({base.get('fused_true')} fused-SDPA, {base.get('fused_false')} math)"
