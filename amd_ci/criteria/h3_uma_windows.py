#!/usr/bin/env python3
"""Criteria: on Windows (no amdgpu sysfs, so no GTT reading) the PR must change nothing for MiniMax-H3 on the Vulkan
build. Regression mode, fit only: head is worse when any argv, refusal or load-fit verdict differs from base for Auto at
864x480x124 and 1344x768x124 or Low VRAM at 864x480x124, all computed by each tree's own Studio code on this box's
real sd-cli listing and live host readings."""

from __future__ import annotations

TITLE = "MiniMax-H3 Vulkan launch flags on Windows 11 gfx1151: base vs head (PR 13786)"
MODE = "regression"
NEEDS = ["gpu", "vulkan", "windows"]
CELLS = (("auto", "864x480"), ("auto", "1344x768"), ("low_vram", "864x480"))


def _cell(s: dict, mode: str, size: str) -> dict:
    if mode == "auto":
        return ((s.get("auto") or {}).get(size) or {}).get("launch") or {}
    return ((s.get("explicit") or {}).get(mode) or {}).get("launch_864x480") or {}


def gates(obs: dict) -> list:
    out = []
    for name in ("base", "head"):
        s = obs.get(name) or {}
        bad = s.get("import_error") or s.get("probe_error")
        out.append((f"{name}: tree imported and probe completed", not bad, bad or s.get("imported_from", "")))
        out.append((f"{name}: Studio reads the Vulkan AMD device from the real sd-cli listing", s.get("listed_class") == "vulkan",
                    f"{s.get('listed_class')} / {s.get('listed_desc')} / check_gtt={s.get('check_gtt')}"))
        for mode, size in CELLS:
            c = _cell(s, mode, size)
            out.append((f"{name}: {mode} {size} launch evaluated", "flags" in c, str(c)[:200]))
    h = obs.get("head") or {}
    out.append(("head: the UMA read ran on the real listing (a value, not None)", h.get("uma_reading") is not None, str(h.get("uma_reading"))))
    return out


def table(obs: dict) -> str:
    rows = ["| cell | base flags / refusal | head flags / refusal | identical |", "|---|---|---|---|"]
    b, h = obs.get("base") or {}, obs.get("head") or {}
    for mode, size in CELLS:
        cb, ch = _cell(b, mode, size), _cell(h, mode, size)
        fmt = lambda c: f"`{' '.join(c.get('flags') or [])}` / {(c.get('refusal') or '-')[:80]}"  # noqa: E731
        rows.append(f"| {mode} {size} | {fmt(cb)} | {fmt(ch)} | {cb.get('flags') == ch.get('flags') and cb.get('refusal') == ch.get('refusal')} |")
    rows.append(f"| auto load fit | {b.get('auto_load_fit')} | {h.get('auto_load_fit')} | {b.get('auto_load_fit') == h.get('auto_load_fit')} |")
    rows.append(f"\nhead uma_reading={h.get('uma_reading')} check_gtt={h.get('check_gtt')} sizes={h.get('sizes_source')}")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict):
    diffs = [f"{m} {s}" for m, s in CELLS
             if (_cell(base, m, s).get("flags"), _cell(base, m, s).get("refusal")) != (_cell(head, m, s).get("flags"), _cell(head, m, s).get("refusal"))]
    if base.get("auto_load_fit") != head.get("auto_load_fit"):
        diffs.append("auto load fit")
    if diffs:
        return True, "head differs from base on Windows in: " + ", ".join(diffs)
    return False, f"all {len(CELLS)} launch cells and the load fit are byte-identical base vs head"
