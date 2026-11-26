#!/usr/bin/env python3
"""Criteria for probes/a12_probe.py: round-5 A1/A2 gfx1151 measurements. Arms differ by config / env / tree inside
the head step, so nothing is judged between states: the gates prove each requested item ran on gfx1151 with
enough arms; KEEP / KILL is decided from the observations by the reader (outputs/search_nov25c/measure_a1a2.md)."""

from __future__ import annotations

import statistics as st

TITLE = "gfx1151: round-5 A1/A2 (K [D,BN], W8A8 stages, thenoise layout/padding, LLC head groups, SDPA compile, attention race, AITER conv3d)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "windows", "multi_gpu"]
MIN_OK = {"aconv": 1, "lift": 10, "w8bo": 6, "fa": 6, "tune_bench": 3, "blk": 3, "lint": 6, "hgs": 60, "kt": 40, "hg_int8": 20, "hg_sdpa": 20, "w8a8": 3, "lin": 3, "sdc": 4, "race": 6, "conv3d": 3}


def _ok(rs):
    return [r for r in rs or [] if not r.get("error") and r.get("rc") == 0 and not r.get("missing")]


def gates(obs: dict) -> list:
    h = obs.get("head") or {}
    arch = (h.get("env") or {}).get("arch") or ""
    items = ((h.get("args") or {}).get("items") or "").split(",")
    out = [("gfx1151", arch.startswith("gfx1151"), arch)]
    keys = {"aconv": [], "lift": ["lift"], "w8bo": ["w8bo"], "fa": ["fa"], "tune": ["tune_bench"], "blk": ["blk"], "lint": ["lint"], "hgs": ["hgs"], "kt": ["kt"], "hg": ["hg_int8", "hg_sdpa"], "w8a8": ["w8a8"], "lin": ["lin"], "sdc": ["sdc"],
            "race": ["race"], "conv3d": ["conv3d"]}
    for item in items:
        for k in keys.get(item, []):
            n = len(_ok(h.get(k)))
            out.append((f"{k} arms ok", n >= MIN_OK.get(k, 1), f"{n} ok of {len(h.get(k) or [])}"))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    return False, "measurement only (arms differ by config / env / tree inside the head step)"


def table(obs: dict) -> str:
    h = obs.get("head") or {}
    L = [f"env {h.get('env')}", f"item seconds {h.get('item_s')}", ""]
    for key in ("kt", "hg_int8", "hg_sdpa", "hgs"):
        by: dict = {}
        for r in _ok(h.get(key)):
            by.setdefault((r.get("shape"), r.get("label")), []).append(r.get("med_ms"))
        for (shape, label), ms in sorted(by.items()):
            L.append(f"{key} {shape} {label}: med ms {[round(x, 2) for x in ms]} -> {round(st.median(ms), 2)}")
        L.append(f"{key} cmp {h.get(key + '_cmp')}")
    for key in ("w8a8", "lin", "lint", "blk", "sdc", "race", "conv3d"):
        L.append(f"{key}: {len(_ok(h.get(key)))} ok runs; errors {[r.get('error', '')[-200:] for r in h.get(key) or [] if r.get('error')]}")
    return "\n".join(L)
