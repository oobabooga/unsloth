#!/usr/bin/env python3
"""Criteria: on a real ROCm host whose card the wheel covers, head must still accept gpu_ids=[0]
and the gate must report the same uncovered set as base. Pairs with probes/gpu_pick_gate_probe.py."""

from __future__ import annotations

TITLE = "Explicit GPU pick through the torch-kernel gate, base versus head"
MODE = "regression"
NEEDS: list[str] = ["linux", "rocm", "gpu"]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name} imported the backend", "import_error" not in o and "error" not in o,
                    o.get("import_error") or o.get("error") or "ok"))
        out.append((f"{name} saw a ROCm torch with a GPU",
                    bool(o.get("hip")) and (o.get("device_count") or 0) > 0,
                    f"torch={o.get('torch')} hip={o.get('hip')} devices={o.get('device_count')} "
                    f"archs={o.get('device_archs')} wheel={o.get('arch_list')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | uncovered | gpu_ids=[0] |", "|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if o:
            rows.append(f"| {name} | {o.get('uncovered')} | {o.get('pick0')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = base.get("pick0") or {}, head.get("pick0") or {}
    if b.get("ok") and not h.get("ok"):
        return True, f"head refuses a pick base accepts: {h.get('error')}"
    if base.get("uncovered") != head.get("uncovered"):
        return True, f"gate result moved: {base.get('uncovered')} -> {head.get('uncovered')}"
    return False, f"same: uncovered={head.get('uncovered')}, pick0={h}"
