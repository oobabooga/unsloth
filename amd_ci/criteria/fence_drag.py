#!/usr/bin/env python3
"""Criteria: does a native scrollbar-thumb drag over a long code fence stall or crash Edge on Windows?

The report (unslothai/unsloth#13919): with the line window on, dragging the thumb across a 3,200-line fence
moves the window on every frame and stalls Chrome, and on Windows can end in STATUS_BREAKPOINT. Base arms:
the checkout's default (window) and the review's scrollbar guard. Head arms: the PR's default (defer), the
window override, and guard plus window. Judged on Edge at 4x CPU throttle; the rest is reported.

Pairs with probes/fence_drag_probe.py.
"""

from __future__ import annotations

import statistics

TITLE = "Scrollbar-thumb drag over a long code fence on Windows, base versus head"
MODE = "differential"
NEEDS = ["windows"]

_JUDGE_CHANNEL = "msedge"


def _rows(state: dict, channel: str) -> list[dict]:
    for c in state.get("channels") or []:
        if c.get("channel") == channel:
            return c.get("rows") or []
    return []


def _cells(state: dict, channel: str, arm: str, throttle: float) -> list[dict]:
    return [r for r in _rows(state, channel) if r.get("arm") == arm and r.get("throttle") == throttle
            and "error" not in r]


def _med(rows: list[dict], *path: str) -> float | None:
    vals = []
    for r in rows:
        v = r
        for p in path:
            v = (v or {}).get(p) if isinstance(v, dict) else None
        if isinstance(v, (int, float)):
            vals.append(v)
    return statistics.median(vals) if vals else None


def _crashes(state: dict, channel: str, arm: str) -> int:
    n = 0
    for r in _rows(state, channel):
        if r.get("arm") != arm:
            continue
        n += int(bool(r.get("crashed"))) + int(r.get("renderer_crashes") or 0) + int(r.get("browser_disconnects") or 0)
    return n


def _arms(name: str) -> list[str]:
    return ["default", "guard"] if name == "base" else ["default", "window", "guard_window"]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        st = obs.get(name) or {}
        out.append((f"{name}: built, served and measured", not st.get("error"),
                    str(st.get("error") or "ok")[:300]))
        for arm in _arms(name):
            rows = _cells(st, _JUDGE_CHANNEL, arm, 4.0)
            ran = [r for r in rows if (r.get("drag") or {}).get("ran")]
            out.append((f"{name}/{arm}: the thumb drag scrolled the thread on Edge at 4x", len(ran) >= 2,
                        f"{len(ran)} of {len(rows)} repetitions dragged the fence's height"))
    return out


def table(obs: dict) -> str:
    lines = ["| browser | state | arm | CPU | drag worst frame | drag frames over 50 ms | drag jank | window DOM swaps "
             "| first-pass worst frame | code spans after traversal | DOM nodes | theme change | crashes |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        st = obs.get(name)
        if not st:
            continue
        for c in st.get("channels") or []:
            ch = c.get("channel")
            for arm in _arms(name):
                for thr in (1.0, 4.0):
                    rows = _cells(st, ch, arm, thr)
                    if not rows:
                        continue

                    def f(*p, fmt = "{:.0f}"):
                        v = _med(rows, *p)
                        return "n/a" if v is None else fmt.format(v)

                    lines.append(
                        f"| {ch} | {name} | {arm} | {thr:g}x | {f('drag', 'worst_frame_ms')} ms | "
                        f"{f('drag', 'frames_over_50')} | {f('drag', 'jank_ms')} ms | {f('drag', 'pre_mutations')} | "
                        f"{f('first_pass', 'worst_frame_ms')} ms | {f('after_traversal', 'code_spans')} | "
                        f"{f('after_traversal', 'dom_nodes')} | {f('theme', 'theme_ms_median')} ms | "
                        f"{_crashes(st, ch, arm)} |")
    lines += ["", "Medians over repetitions. Crashes count renderer crashes and browser disconnects across the measured "
              "cells and the rapid-drag stress loop. Jank = time past one frame in frames over 50 ms."]
    return "\n".join(lines)


def _jank(state: dict, arm: str) -> float | None:
    return _med(_cells(state, _JUDGE_CHANNEL, arm, 4.0), "drag", "jank_ms")


def base_shows_defect(base: dict) -> bool:
    plain, guard = _jank(base, "default"), _jank(base, "guard")
    stalls = plain is not None and guard is not None and plain > 200 and plain > 2 * guard
    return stalls or _crashes(base, _JUDGE_CHANNEL, "default") > 0


def head_is_fixed(head: dict) -> bool:
    # Absolute, since this hook sees one state: the base guard arm on Linux stays well under this.
    j = _jank(head, "default")
    return j is not None and j <= 200 and _crashes(head, _JUDGE_CHANNEL, "default") == 0
