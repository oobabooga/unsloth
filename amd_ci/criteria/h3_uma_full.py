#!/usr/bin/env python3
"""Criteria: MiniMax-H3 GGUF, Vulkan build, integrated AMD (UMA), nothing held (the default, healthy host).

Regression mode. Head is worse when any of its renders fails where base's all pass, or when its argv differs from base
and its median wall time exceeds base's median by more than max(5 %, base's own spread). Byte-identical argv is
reported as such (the timing then compares one command with itself)."""

from __future__ import annotations

import statistics

TITLE = "MiniMax-H3 Vulkan render on gfx1151, full GTT (no holder): does the reserve cost the healthy host? (PR 13786)"
MODE = "regression"
NEEDS = ["gpu", "vulkan", "windows"]
SIZE = "864x480"


def _e(s):
    return ((s.get("auto") or {}).get(SIZE)) or {}


def _secs(s):
    return [r["secs"] for r in _e(s).get("renders", []) if r.get("ok")]


def gates(obs: dict) -> list:
    out = []
    for name in ("base", "head"):
        s = obs.get(name) or {}
        bad = s.get("import_error") or s.get("probe_error")
        out.append((f"{name}: tree imported and probe completed", not bad, bad or ""))
        out.append((f"{name}: GTT mostly free (> 50 GiB)", (s.get("readings_start") or {}).get("gtt_free_gib", 0) > 50,
                    str(s.get("readings_start"))))
        out.append((f"{name}: at least 3 renders ran", len(_e(s).get("renders", [])) >= 3, str(len(_e(s).get("renders", [])))))
    h = obs.get("head") or {}
    out.append(("head: the new UMA read engaged", h.get("uma_reading") is True, str(h.get("uma_reading"))))
    return out


def table(obs: dict) -> str:
    rows = ["| state | resident | flags | renders ok | secs per run | median | spread | peak GTT used | sd.cpp budget line |", "|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        s = obs.get(name) or {}
        e = _e(s)
        la = e.get("launch") or {}
        rs = e.get("renders", [])
        secs = [r.get("secs") for r in rs]
        ok = _secs(s)
        med = statistics.median(ok) if ok else None
        spread = (max(ok) - min(ok)) if ok else None
        budget = next((x for x in (rs[0].get("log_highlights", []) if rs else []) if "auto-detected" in x), "-")
        rows.append(f"| {name} | {la.get('resident')} | `{' '.join(la.get('flags') or [])}` | {len(ok)}/{len(rs)} | {secs} | {med} | {spread} | "
                    f"{[r.get('peak_gtt_used_gib') for r in rs]} | {budget[:160]} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict):
    b, h = _secs(base), _secs(head)
    nb, nh = len(_e(base).get("renders", [])), len(_e(head).get("renders", []))
    if len(h) < nh and len(b) == nb:
        return True, f"head failed {nh - len(h)} of {nh} renders, base none"
    same = (_e(base).get("launch") or {}).get("flags") == (_e(head).get("launch") or {}).get("flags")
    if not b or not h:
        return False, "no successful renders on one side; nothing to time"
    mb, mh = statistics.median(b), statistics.median(h)
    band = max(0.05 * mb, max(b) - min(b))
    detail = (f"argv {'byte-identical' if same else 'differs'}; median base {mb:.1f}s vs head {mh:.1f}s "
              f"(delta {mh - mb:+.1f}s, noise band {band:.1f}s)")
    return (not same and mh - mb > band), detail
