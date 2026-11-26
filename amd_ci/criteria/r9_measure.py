#!/usr/bin/env python3
"""Criteria for probes/r9_probe.py: round-9 gfx1151 measurements on the head state (arms pick main or the patched
tree themselves). Nothing is judged between states: the gates prove each listed item ran on gfx1151 (a gated-off
conditional step counts as ran when its reason is recorded), and the table carries the numbers plus one verdict line
per item against its pre-registered kill line. Keep / kill is read from those lines."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TITLE = "gfx1151 round 9: C6-3 / C6-1 / C6-4 / M4 / M3"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "windows", "multi_gpu"]
JUDGES = ("r9_judge", "r9_judge_m3")


def _judges():
    out, errs = [], {}
    for name in JUDGES:
        try:
            out.append(importlib.import_module(name))
        except ModuleNotFoundError as exc:
            if exc.name != name:
                errs[name] = repr(exc)[:300]
        except Exception as exc:  # noqa: BLE001
            errs[name] = repr(exc)[:300]
    return out, errs


def gates(obs: dict) -> list:
    h = obs.get("head") or {}
    arch = (h.get("env") or {}).get("arch") or ""
    items = [x for x in ((h.get("args") or {}).get("items") or "").split(",") if x]
    out = [("gfx1151", arch.startswith("gfx1151"), arch), ("items listed", bool(items), ",".join(items))]
    js, errs = _judges()
    for name, err in errs.items():
        out.append((f"judge {name} imports", False, err))
    for j in js:
        try:
            out.extend(j.gates(h, items))
        except Exception as exc:  # noqa: BLE001
            out.append((f"judge {j.__name__} gates", False, repr(exc)[:300]))
    for it in items:
        if h.get(f"{it}_error"):
            out.append((f"{it} no exception", False, str(h[f"{it}_error"])[-300:]))
    me = h.get("module_errors") or {}
    for mod, err in me.items():
        out.append((f"probe module {mod} imports", not any(mod.endswith(s) for s in _needed_mods(items)), err[-200:]))
    return out


def _needed_mods(items):
    need = []
    if set(items) & {"smoke", "c63", "m4", "c61_knee", "c61_krea"}:
        need.append("r9_items")
    if "m3" in items:
        need.append("r9_m3")
    if set(items) & {"c64_decode"}:
        need.append("r9_c64")
    return need


def head_is_worse(base: dict, head: dict) -> tuple:
    return False, "measurement only (head state; arms differ by env / tree, numbers in the table)"


def table(obs: dict) -> str:
    h = obs.get("head") or {}
    L = [f"env {h.get('env')}", f"item seconds {h.get('item_s')}", ""]
    js, _ = _judges()
    for j in js:
        try:
            L.extend(j.table(h))
        except Exception as exc:  # noqa: BLE001
            L.append(f"judge {j.__name__} table error {exc!r}")
    return "\n".join(f"- {x}" if x and not x.startswith("-") else x for x in L)
