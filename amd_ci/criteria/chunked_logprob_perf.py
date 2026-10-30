#!/usr/bin/env python3
"""Criteria: the chunked log-prob head (unsloth-zoo#1803) is not slower than zoo main's path on this GPU."""

from __future__ import annotations

import statistics

TITLE = "Chunked log-prob head: step time vs zoo main, per variant"
MODE = "regression"
NEEDS = ["gpu", "multi_gpu", "nvidia", "mig", "xpu", "mlx"]


def _med(state: dict, variant: str, s: int, key: str = "total_ms"):
    xs = (((state.get("variants") or {}).get(variant) or {}).get(str(s)) or {}).get(key) or []
    return statistics.median(xs[1:]) if len(xs) > 1 else None


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = {n: v for n, v in obs.items() if not n.startswith("_")}
    errs = {f"{n}/{k}": (v.get("error") or w.get("error")) for n, v in states.items()
            for k, w in (v.get("variants") or {"-": {}}).items() if v.get("error") or w.get("error")}
    return [("every state and variant ran", not errs, "; ".join(f"{k}: {e[-300:]}" for k, e in errs.items()))]


def table(obs: dict) -> str:
    rows = ["| state | variant | 1k total ms | 1k no-grad ms | 4k total ms | 4k no-grad ms |", "|---|---|---|---|---|---|"]
    for n, v in obs.items():
        if n.startswith("_"):
            continue
        for k in v.get("variants") or {}:
            rows.append(f"| {n} | {k} | {_med(v, k, 1024)} | {_med(v, k, 1024, 'nograd_ms')} | "
                        f"{_med(v, k, 4096)} | {_med(v, k, 4096, 'nograd_ms')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = _med(base, "default", 4096), _med(head, "default", 4096)
    if b is None or h is None:
        return True, "missing 4k timing"
    return h > 1.25 * b, f"4k total: base {b} ms, head {h} ms"
