#!/usr/bin/env python3
"""Criteria: Studio's multi-GPU paths on tiny pipelines under the device multiplier (wiring only).

Pairs with probes/mgpu_e2e_probe.py. A cell is DEFECTIVE when it crashed, when a CFG cell did
not arm or never ran a replica call, when a split cell planned a split but did not place it on
both cards, or when a multi-card render is not bit-identical to the one-card reference.
Differential: the base must show a defective cell (else VOID) and the head must show none.

The phantom card is the real gfx1151 renumbered: bit-identity and engagement are wiring facts;
nothing here says anything about two-card speed, memory or mixed architectures.
"""

from __future__ import annotations

import json

TITLE = "Multi-GPU wiring on tiny pipelines (spoofed second device)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "multi_gpu", "discrete_gpu", "linux"]

_CFG = ("cfg_flux", "cfg_qwen", "cfg_flux_compiled", "zimage_pair", "video_wan_cfg")
_SPLIT = ("split_flux", "split_wan")


def _defects(o: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, c in (o.get("cells") or {}).items():
        if not c.get("ok"):
            out[name] = c.get("error") or "crashed"
            continue
        if name in _CFG:
            if not c.get("armed"):
                out[name] = f"not armed: {c.get('decline')}"
            elif not (c.get("router_stats") or {}).get("replica_calls"):
                out[name] = f"armed but 0 replica calls (disabled: {c.get('router_disabled')})"
            elif c.get("bit_identical") is not True:
                out[name] = "two-card render differs from one card"
        elif name in _SPLIT:
            if not c.get("engaged_plan"):
                out[name] = f"no split planned (declined: {c.get('declined')})"
            elif c.get("bit_identical") is not True:
                out[name] = "split render differs from one card"
            elif len(c.get("devices_after_place") or []) < 2:
                out[name] = f"split placed on {c.get('devices_after_place')}"
        elif name == "component_flux":
            if c.get("secondary_ordinal") is None:
                out[name] = "no companion split"
            elif c.get("bit_identical") is not True:
                out[name] = "companion split render differs from one card"
    return out


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        host = o.get("host") or {}
        out.append((f"{name}: torch sees two devices", host.get("device_count") == 2, json.dumps(host)[:300]))
        out.append((f"{name}: ROCm torch", bool(host.get("hip")), str(host.get("hip"))))
        out.append((f"{name}: cells ran", bool(o.get("cells")), f"{len(o.get('cells') or {})} cells"))
    return out


def table(obs: dict) -> str:
    rows = ["| cell | state | ok | armed / planned | replica calls | pairs | bit-identical | note |",
            "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        bad = _defects(o)
        for cell, c in (o.get("cells") or {}).items():
            rs = c.get("router_stats") or {}
            armed = c.get("armed", c.get("engaged_plan", c.get("secondary_ordinal")))
            rows.append(f"| {cell} | {name} | {c.get('ok')} | {armed} | {rs.get('replica_calls', '')} "
                        f"| {rs.get('pairs_parallel', '')} | {c.get('bit_identical')} "
                        f"| {str(bad.get(cell, ''))[:160]} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return bool(_defects(base))


def head_is_fixed(head: dict) -> bool:
    return bool(head.get("cells")) and not _defects(head)
