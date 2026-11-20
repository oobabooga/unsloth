#!/usr/bin/env python3
"""Criteria: native W8A8 RDNA kernel selection by card on gfx1151 (judges probes/rdna_card_probe.py).

base_shows_defect: a layer quantised on cuda:1 while cuda:0 (marked unsupported) is current never takes the RDNA
kernels for cuda:1 operands.
head_is_fixed: that layer takes them (rdna.rotq called) with a finite output within 2% relative L2 of the fp32
reference, and the ordinary single-card case engages and stays accurate on both arms.
"""

from __future__ import annotations

TITLE = "gfx1151: native W8A8 layer quantised on card 1 takes the RDNA int8 kernels"
MODE = "differential"
NEEDS = ["gpu", "rocm", "linux"]


def _ok(c: dict) -> bool:
    return bool(c) and c.get("rdna_calls", 0) > 0 and c.get("finite") and (c.get("rel_l2") or 1) <= 0.02


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name}: probe ran", "same" in o, f"{o.get('error') or ''} {o.get('arch')} {o.get('torch')}"[:300]))
        out.append((f"{name}: two devices visible (spoofed)", (o.get("devices") or 0) >= 2, str(o.get("devices"))))
        out.append((f"{name}: single-card load engages RDNA, accurate", _ok(o.get("same") or {}),
                    str(o.get("same"))[:300]))
    return out


def base_shows_defect(base):
    c = base.get("cross") or {}
    return (bool(c) and c.get("rdna_calls", 1) == 0, f"cross: {c}"[:300])


def head_is_fixed(head):
    c = head.get("cross") or {}
    return (_ok(c), f"cross: {c}"[:300])


def table(obs):
    rows = ["| state | case | card | current | RDNA flag | rotq calls | finite | rel L2 |", "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for case in ("same", "cross"):
            c = o.get(case) or {}
            rows.append(f"| {name} | {case} | {c.get('card')} | {c.get('current')} | {c.get('rdna_flag')} | "
                        f"{c.get('rdna_calls')} | {c.get('finite')} | {c.get('rel_l2')} |")
    return "\n".join(rows)
