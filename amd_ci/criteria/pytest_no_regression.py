#!/usr/bin/env python3
"""Criteria: does this change break a test that used to pass?

Compares by test ID, never by count. A PR that adds tests raises the pass count
and can raise the fail count without regressing anything, and a PR that deletes
a failing test lowers both while fixing nothing. Only the set difference of
FAILED ids answers the question asked.

Pairs with probes/pytest_probe.py.
"""

from __future__ import annotations

TITLE = "Backend suites, base versus head"
MODE = "regression"
NEEDS: list[str] = []


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ran = o.get("rc") is not None
        out.append((f"{name} suite actually ran", ran,
                    o.get("error") or o.get("note") or f"rc={o.get('rc')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        failed = ", ".join(f"`{f.split('::')[-1]}`" for f in o.get("failed", [])) or "none"
        rows.append(f"| {name} | {o.get('n_passed', 0)} | {o.get('n_failed', 0)} "
                    f"| {o.get('n_skipped', 0)} | {failed} |")
    note = ""
    absent = (obs.get("base") or {}).get("absent_at_this_state") or []
    if absent:
        note = ("\n\nTests absent at the base (added by this change), excluded from the "
                "comparison: " + ", ".join(f"`{a}`" for a in absent))
    return "\n".join(rows) + note


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b = set(base.get("failed", [])) | set(base.get("errors", []))
    h = set(head.get("failed", [])) | set(head.get("errors", []))
    new = sorted(h - b)
    fixed = sorted(b - h)
    if new:
        return True, ("newly failing at the head: "
                      + ", ".join(f"`{n.split('::')[-1]}`" for n in new))
    detail = "no test that passed at the base fails at the head"
    if fixed:
        detail += "; also newly passing: " + ", ".join(f"`{n.split('::')[-1]}`" for n in fixed)
    if h:
        detail += (". Note "
                   + ", ".join(f"`{n.split('::')[-1]}`" for n in sorted(h))
                   + " fails at BOTH states, so it pre-dates this change")
    return False, detail
