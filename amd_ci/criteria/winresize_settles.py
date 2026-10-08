#!/usr/bin/env python3
"""Criteria: does the Desktop WebView end at the window's client size after fast
resizes (PR 12542)?

Defect: after a fast drag the WebView (wry container or WebView2 widget) stays at
an older size, leaving a blank strip. Base shows it when any trial is still off
1.5 s after the resize; head is fixed when none is. Drags that never resized the
window, or a session without real input, prove nothing and fail a gate.
"""

from __future__ import annotations

TITLE = "Desktop WebView bounds after fast resize (Windows)"
MODE = "differential"
NEEDS = ["windows"]


def _states(obs: dict):
    return [(n, v) for n, v in obs.items() if not n.startswith("_")]


def _trials(state: dict) -> list[dict]:
    return [t for rec in state.get("launches") or [] for t in rec.get("trials") or []]


def _moved_drags(state: dict) -> int:
    return sum(1 for t in _trials(state)
               if t["kind"] == "drag" and t.get("drag") and t["drag"]["start"] != t["drag"]["end"])


def _bad(state: dict) -> int:
    return sum(1 for t in _trials(state) if t["settled_mismatch"])


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, v in _states(obs):
        launches = v.get("launches") or []
        ready = [r for r in launches if r.get("ready") in ("ready", "forced_show") and not r.get("error")]
        out.append((f"{name}: app window + WebView came up", bool(launches) and len(ready) == len(launches),
                    v.get("error") or f"{len(ready)}/{len(launches)} launches ready "
                    f"({', '.join(str(r.get('ready') or r.get('error')) for r in launches)})"))
        inter = [r for r in launches if r.get("cursor_ok") and r.get("input_desktop")]
        out.append((f"{name}: interactive desktop (input reaches it)", bool(launches) and len(inter) == len(launches),
                    f"cursor_ok/input_desktop on {len(inter)}/{len(launches)} launches"))
        moved = _moved_drags(v)
        out.append((f"{name}: real drags resized the window", moved > 0, f"{moved} drags moved the window"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | commit | trials | off after 1.5 s | off at 150 ms | drags that resized | worst lag during a drag (px) |",
            "|---|---|---|---|---|---|---|"]
    for name, v in _states(obs):
        ts = _trials(v)
        lag = max([abs(t["drag"]["worst_during"][0]) + abs(t["drag"]["worst_during"][1])
                   for t in ts if t.get("drag")] or [0])
        rows.append(f"| {name} | {str(v.get('sha'))[:9]} | {len(ts)} | {_bad(v)} | "
                    f"{sum(1 for t in ts if t['early_mismatch'])} | {_moved_drags(v)} | {lag} |")
    return "\n".join(rows)


def base_shows_defect(state: dict) -> tuple[bool, str]:
    n = _bad(state)
    return n > 0, f"{n}/{len(_trials(state))} trials left the WebView off the client size"


def head_is_fixed(state: dict) -> tuple[bool, str]:
    n = _bad(state)
    return n == 0, f"{n}/{len(_trials(state))} trials left the WebView off the client size"
