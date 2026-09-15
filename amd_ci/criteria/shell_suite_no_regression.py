#!/usr/bin/env python3
"""Criteria: does a shell installer suite that passed at the base fail at the head?

Set difference of failing suite names (suite@shell), never counts: a PR that adds
suites raises the run count without regressing anything. Pairs with
probes/shell_suite_probe.py.
"""

from __future__ import annotations

TITLE = "Shell installer suites (tests/sh), base versus head"
MODE = "regression"
NEEDS: list[str] = []


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name} ran at least one suite", (o.get("n_run") or 0) > 0,
                    f"{o.get('n_run', 0)} run, {o.get('n_failed', 0)} failed"))
        missing = [k for k, v in (o.get("results") or {}).items() if v.get("rc") == -2]
        out.append((f"{name} had every requested interpreter", not missing,
                    ", ".join(missing) if missing else "ok"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | run | failed | failing suites |", "|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        failed = ", ".join(f"`{f}`" for f in o.get("failed", [])) or "none"
        rows.append(f"| {name} | {o.get('n_run', 0)} | {o.get('n_failed', 0)} | {failed} |")
    absent = (obs.get("base") or {}).get("absent") or []
    note = ("\n\nSuites absent at the base (added by this change): "
            + ", ".join(f"`{a}`" for a in absent)) if absent else ""
    return "\n".join(rows) + note


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b, h = set(base.get("failed", [])), set(head.get("failed", []))
    new = sorted(h - b)
    if new:
        return True, "newly failing at the head: " + ", ".join(f"`{n}`" for n in new)
    detail = "no suite that passed at the base fails at the head"
    if h:
        detail += "; " + ", ".join(f"`{n}`" for n in sorted(h)) + " fails at both states"
    return False, detail
