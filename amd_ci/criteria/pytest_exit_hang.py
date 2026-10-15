#!/usr/bin/env python3
"""Criteria: does a pytest process that finished its tests fail to EXIT at the head but not the base?

Pairs with probes/pytest_exit_hang_probe.py. Regression mode, per selection.
"""

from __future__ import annotations

TITLE = "pytest process exit after the tests finish (hang at interpreter shutdown?)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm", "linux", "nvidia", "discrete_gpu"]


def _runs(o: dict) -> dict:
    return (o or {}).get("runs") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        runs = _runs(obs.get(name))
        reached = [t for t, r in runs.items() if r.get("pytest_done")]
        out.append((f"{name}: pytest finished in at least one selection", bool(reached),
                    f"{len(reached)}/{len(runs)} selections reached PYTEST_DONE"))
    return out


def table(obs: dict) -> str:
    rows = ["| selection | state | pytest result | wall s | exited | live threads at return |",
            "|---|---|---|---|---|---|"]
    tags = sorted(set(_runs(obs.get("base"))) | set(_runs(obs.get("head"))), key = lambda t: (t != "all", t))
    for tag in tags:
        for name in ("base", "head"):
            r = _runs(obs.get(name)).get(tag)
            if not r:
                continue
            thr = ", ".join(f"{t.get('name')}({'d' if t.get('daemon') else 'ND'})"
                            for t in (r.get("threads_at_return") or []) if isinstance(t, dict)) or "none"
            rows.append(f"| {tag} | {name} | {r.get('summary') or r.get('pytest_done') or 'no summary'} "
                        f"| {r.get('wall_s')} | {'HUNG' if r.get('hung_after_tests') else ('yes' if r.get('returncode') is not None else 'no')} "
                        f"| {thr} |")
    parts = ["\n".join(rows), ""]
    for name in ("head", "base"):
        for tag, r in _runs(obs.get(name)).items():
            if r.get("hung_after_tests"):
                parts.append(f"**{name} / {tag}: watchdog stacks (process alive after the tests)**\n")
                parts.append("```\n" + (r.get("watchdog_dump") or "")[-4000:] + "\n```\n")
    return "\n".join(parts)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b = {t for t, r in _runs(base).items() if r.get("hung_after_tests")}
    h = {t for t, r in _runs(head).items() if r.get("hung_after_tests")}
    new = sorted(h - b)
    if new:
        return True, "the process hangs after its tests finish at the head only: " + ", ".join(new)
    return False, ("no selection hangs at the head that exits at the base"
                   + (f"; hangs at both: {', '.join(sorted(h & b))}" if h & b else ""))
