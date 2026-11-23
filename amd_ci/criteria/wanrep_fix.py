#!/usr/bin/env python3
"""Criteria for probes/wanrep_probe.py --arms proof --states base,head (Wan2.2-TI2V-5B reduction-config pin on gfx1151).

Per state, 4 fresh processes per config: P = INT8 attention off, A = Studio default (Auto). Each process renders one
prompt + seed 3 times (render 0 compiles, 1-2 steady).
  base_shows_defect: in some config, two base processes produced different frames.
  head_is_fixed:     every head config's processes are bit-identical (all renders), the pin engaged in every head
                     process (reduction_pins has a True), and the head's steady render median is within the base A/A
                     spread of the base median (max(base per-process median) - min(...), floor 1 %) per config.
Gates: every arm of both states rendered 3 times."""
from __future__ import annotations

import statistics

TITLE = "gfx1151 Linux: Wan2.2-TI2V-5B renders bit-identical across processes (reduction-config pin), base vs head"
MODE = "differential"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "windows", "windows_rocm_wddm"]
CONFIGS = ("P", "A")
_BASE: dict = {}


def _groups(o: dict) -> dict:
    return (o or {}).get("groups") or {}


def gates(obs: dict) -> list:
    # head_is_fixed only receives the head observation: keep the base for its speed bound (gates run first).
    _BASE.clear()
    _BASE.update(obs.get("base") or {})
    out = []
    for st in ("base", "head"):
        g = _groups(obs.get(st))
        for c in CONFIGS:
            x = g.get(c) or {}
            ok = bool(x.get("ok")) and all(x["ok"]) and len(x["ok"]) >= 3
            out.append((f"{st} {c}: every process rendered", ok, f"{x.get('tags')} ok={x.get('ok')}"))
    return out


def _variants(x: dict) -> set:
    return {tuple(f) for f in x.get("frames") or [] if f}


def _proc_medians(x: dict) -> list:
    return [statistics.median(s) for s in x.get("steady_s") or [] if s]


def base_shows_defect(base: dict):
    g = _groups(base)
    seen = {c: len(_variants(g.get(c) or {})) for c in CONFIGS}
    return any(v > 1 for v in seen.values()), f"distinct base frame sets per config: {seen}"


def _speed(base: dict, head: dict, c: str):
    b = _proc_medians(_groups(base).get(c) or {})
    h = _proc_medians(_groups(head).get(c) or {})
    if not b or not h:
        return False, f"{c}: no timings"
    bm, hm = statistics.median(b), statistics.median(h)
    spread = max(max(b) - min(b), 0.01 * bm)
    return hm <= bm + spread, f"{c}: base median {bm:.3f}s (A/A spread {spread:.3f}s, per process {b}), head median {hm:.3f}s (per process {h})"


def head_is_fixed(head: dict):
    g = _groups(head)
    lines, ok = [], True
    for c in CONFIGS:
        x = g.get(c) or {}
        v = _variants(x)
        frames_one = len(v) == 1 and all(len(f) == 1 for f in v)
        pins = x.get("pins") or []
        engaged = bool(pins) and all(p and any(p) for p in pins)
        sp_ok, sp = _speed(_BASE, head, c) if _BASE else (False, "base timings unavailable")
        ok = ok and frames_one and engaged and sp_ok
        lines.append(f"{c}: {len(v)} frame set(s) {sorted(v)}, pin engaged {engaged} ({pins}); {sp}")
    return ok, "; ".join(lines)


def table(obs: dict) -> str:
    rows = ["| state | config | process | frames | first render s | steady s | reduction pins |", "|---|---|---|---|---|---|---|"]
    for st in ("base", "head"):
        for c in CONFIGS:
            x = _groups(obs.get(st)).get(c) or {}
            for i, t in enumerate(x.get("tags") or []):
                rows.append(f"| {st} | {c} | {t} | {x['frames'][i]} | {x['first_s'][i]} | {x['steady_s'][i]} | {x['pins'][i]} |")
    for c in CONFIGS:
        rows.append(f"\nspeed {_speed(obs.get('base') or {}, obs.get('head') or {}, c)[1]}")
    return "\n".join(rows)
