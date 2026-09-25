#!/usr/bin/env python3
"""Criteria: a card the change does not target keeps the exact route and detection it had."""

from __future__ import annotations

TITLE = "Torch route and GPU detection on an untargeted AMD card, base versus head"
MODE = "regression"
NEEDS = ["gpu", "rocm", "windows"]
_KEYS = ("gfx", "index_url", "specs", "has_rocm_gpu", "device", "gpu_name", "mismatch_eligible")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        err = o.get("route_error") or o.get("hw_error")
        out.append((f"{name} probe ran", not err and bool(o.get("device")), err or o.get("device", "no device")))
        if o.get("os") == "Windows":
            out.append((f"{name} found a gfx arch", bool(o.get("gfx")), str(o.get("gfx"))))
        out.append((f"{name} detected the GPU, not CPU", "cpu" not in str(o.get("device", "")).lower(), str(o.get("device"))))
    return out


def table(obs: dict) -> str:
    rows = ["| state | " + " | ".join(_KEYS) + " | detect s |", "|---" * (len(_KEYS) + 2) + "|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        rows.append(f"| {name} | " + " | ".join(str(o.get(k)) for k in _KEYS) + f" | {o.get('detect_s', 0):.2f} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    moved = [k for k in _KEYS if base.get(k) != head.get(k)]
    if moved:
        return True, "moved: " + ", ".join(f"{k} {base.get(k)!r} -> {head.get(k)!r}" for k in moved)
    return False, "route and detection identical"
