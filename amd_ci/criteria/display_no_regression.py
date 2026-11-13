#!/usr/bin/env python3
"""Criteria (#17523): on this host the VRAM tile and floating monitor read the same on head as on base
(the PR only re-files an unreadable iGPU that sits beside discrete cards)."""

from __future__ import annotations

TITLE = "VRAM display unchanged on a lone APU host"
MODE = "regression"
NEEDS = ["gpu", "multi_gpu", "nvidia"]
KEYS = ("shared_devices", "usage_devices", "totals", "used_gb", "floating_total_gb", "floating_used_gb", "tile_capacity")


def _states(obs):
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def gates(obs):
    st = _states(obs)
    return [("every state rendered a display", all(v.get("display") for v in st.values()),
             "; ".join(f"{k}: {v.get('payload_error') or v.get('display_error') or v.get('node_stderr', '')[-300:] or 'ok'}" for k, v in st.items())),
            ("backend reported a GPU", all((v.get("gpu") or {}).get("devices") for v in st.values()), "")]


def table(obs):
    rows = ["| state | platform | " + " | ".join(KEYS) + " |", "|---|---|" + "---|" * len(KEYS)]
    for k, v in _states(obs).items():
        d = v.get("display") or {}
        rows.append(f"| {k} | {v.get('platform')} | " + " | ".join(str(d.get(x)) for x in KEYS) + " |")
    return "\n".join(rows)


def head_is_worse(base, head):
    b, h = base.get("display") or {}, head.get("display") or {}
    moved = [k for k in KEYS if k not in ("used_gb", "floating_used_gb") and b.get(k) != h.get(k)]
    return bool(moved), ("moved: " + ", ".join(moved)) if moved else "display identical"
