#!/usr/bin/env python3
"""Criteria: a frontend-only PR leaves the real NPU status (the footer's precondition) unchanged."""

from __future__ import annotations

TITLE = "Real NPU status, base versus head"
MODE = "regression"
NEEDS: list[str] = []


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ok = isinstance(o.get("status"), dict) and "supported" in o["status"]
        out.append((f"{name} read the NPU status", ok, o.get("error") or o.get("module_file", "")))
    return out


def table(obs: dict) -> str:
    rows = ["| state | supported | hardware | state | footer shows |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        s = (obs.get(name) or {}).get("status") or {}
        hw = s.get("hardware") or {}
        rows.append(f"| {name} | {s.get('supported')} | {hw.get('name') or hw.get('family') or '-'} "
                    f"| {s.get('state')} | {(obs.get(name) or {}).get('footer_would_show')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    keys = ("supported", "hardware", "state", "ready")
    b = {k: (base.get("status") or {}).get(k) for k in keys}
    h = {k: (head.get("status") or {}).get(k) for k in keys}
    if b != h:
        return True, f"NPU status moved: base {b} head {h}"
    return False, f"identical NPU status on both states: {b}"
