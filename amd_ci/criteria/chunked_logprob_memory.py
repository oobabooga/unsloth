#!/usr/bin/env python3
"""Criteria: base's log-prob head memory grows with sequence length; head's stays flat and is at least as accurate."""

from __future__ import annotations

TITLE = "Chunked log-prob head: memory flat in sequence length, accuracy vs fp64"
MODE = "differential"
NEEDS = ["gpu", "multi_gpu", "nvidia", "mig", "xpu", "mlx"]


def _r(state, s):
    return ((state.get("results") or {}).get(str(s)) or {})


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = {n: v for n, v in obs.items() if not n.startswith("_")}
    ran = all("results" in v and "error" not in v for v in states.values())
    finite = all(_r(v, s).get("finite") for v in states.values() for s in (1024, 4096))
    return [
        ("every state ran on the GPU", ran, "; ".join(f"{n}: {v.get('error', 'ok')}" for n, v in states.items())),
        ("every gradient finite", finite, ""),
    ]


def base_shows_defect(state: dict) -> tuple[bool, str]:
    a, b = _r(state, 1024).get("peak_extra_gib", 0), _r(state, 4096).get("peak_extra_gib", 0)
    return b > 2.5 * a and b > 1.0, f"extra peak {a} GiB at 1k -> {b} GiB at 4k"


def head_is_fixed(state: dict) -> tuple[bool, str]:
    a, b = _r(state, 1024).get("peak_extra_gib", 0), _r(state, 4096).get("peak_extra_gib", 0)
    err = _r(state, 1024).get("grad_rel_err", 1.0)
    return b < 2.5 and err < 1e-2, f"extra peak {a} GiB at 1k -> {b} GiB at 4k, grad err {err:.2e}"
