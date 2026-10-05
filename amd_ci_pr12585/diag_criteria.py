#!/usr/bin/env python3
"""Criteria for the PR 12585 diagnostics. The GA variants are reported, not judged; the
proposed kernel-binding test must pass at the head (it is the fix being proposed)."""

from __future__ import annotations

TITLE = "PR 12585 diagnostics on gfx1151: GA equivalence variants, kernel-binding test"
MODE = "regression"
NEEDS = ["gpu", "nvidia"]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    head = obs.get("head") or {}
    runs = head.get("runs") or {}
    want = ("ga_default", "ga_highest_precision", "ga_no_gpu", "ga_one_thread", "kernels_original", "kernels_proposed")
    missing = [w for w in want if w not in runs or runs[w].get("rc") in (-1, 2, 3, 4, 5)]
    return [("every head run executed pytest", not missing, ", ".join(missing) or "all"),
            ("base has no decision tests", bool((obs.get("base") or {}).get("absent")), "")]


def table(obs: dict) -> str:
    head = obs.get("head") or {}
    rows = [f"transformers {head.get('transformers')}", "",
            "| run | rc | result | max abs / rel | mismatched | kernels | trainer forward |", "|---|---|---|---|---|---|---|"]
    for name, r in (head.get("runs") or {}).items():
        seen = "; ".join(r.get("seen") or [])[:600].replace("|", "/")
        ma = f"{r['max_abs']:.2e} / {r['max_rel']:.2e}" if "max_abs" in r else "-"
        rows.append(f"| {name} | {r.get('rc')} | {r.get('summary', '')} | {ma} | {r.get('mismatched', '-')} | "
                    f"{', '.join(r.get('kernels') or []) or '-'} | {seen or '-'} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    r = (head.get("runs") or {}).get("kernels_proposed") or {}
    if r.get("rc") != 0:
        return True, f"proposed kernel-binding test failed: {r.get('summary')}"
    return False, "proposed kernel-binding test passes on this host; GA variants reported above"
