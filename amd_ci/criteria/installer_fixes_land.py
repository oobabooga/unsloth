#!/usr/bin/env python3
"""Criteria: do the installer fixes actually fix something that was broken?

`tests/sh` is where install.sh's ROCm arch gate, uv cache colocation and shell-rc
fallback are asserted, and all three test scripts are ADDED by the change. Absent
at the base, they say nothing: a test that does not exist cannot fail, and a run
that only observes them passing at the head shows the harness worked.

So the probe ports the head's scripts into the base checkout and runs them there.
Same assertions, older install.sh. The base must fail them, or the change is
fixing nothing and the verdict is VOID.

Pairs with probes/shell_suite_probe.py --scripts-from.
"""

from __future__ import annotations

TITLE = "install.sh fixes, asserted against the old implementation"
MODE = "differential"
NEEDS = [
    "linux", "gpu", "rocm", "gfx1033_vangogh", "glibc_pre_228",
    "windows", "windows_docker", "mlx", "nvidia", "xpu", "multi_gpu",
    "discrete_gpu",
]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ran = (o.get("n_passed", 0) or 0) + (o.get("n_failed", 0) or 0)
        out.append((f"{name} ran the scripts", ran > 0,
                    o.get("note") or f"{ran} scripts executed"))
        out.append((f"{name} had every script present", not o.get("absent_at_this_state"),
                    ", ".join(o.get("absent_at_this_state") or []) or "none absent"))

    # Without the port the base is merely missing the tests, and "0 failed" there
    # would read as the base being fine. This gate is what stops that.
    base = obs.get("base") or {}
    out.append(("the head's scripts were ported onto the base",
                bool(base.get("ported")),
                ", ".join(base.get("ported") or []) or "nothing was copied"))
    return out


def _leaf(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def table(obs: dict) -> str:
    rows = ["| state | scripts run | passed | failed | failing scripts | ported in |",
            "|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        failing = ", ".join(f"`{_leaf(f)}`" for f in o.get("failed", [])) or "none"
        ported = ", ".join(f"`{_leaf(p)}`" for p in o.get("ported", [])) or "none"
        rows.append(f"| {name} | {len(o.get('selected', []))} | {o.get('n_passed', 0)} "
                    f"| {o.get('n_failed', 0)} | {failing} | {ported} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    # Every ported script must fail against the old install.sh. One passing would
    # mean that assertion held before the change, so it is not evidence for it.
    ported = set(base.get("ported") or [])
    return bool(ported) and ported.issubset(set(base.get("failed") or []))


def head_is_fixed(head: dict) -> bool:
    return (head.get("n_failed", 1) == 0) and (head.get("n_passed", 0) > 0)
