#!/usr/bin/env python3
"""Criteria: with two GPUs visible, does a model loaded alongside go to the GPU the first one is NOT on (PR 11591)?

Wiring evidence only: the second GPU is the real one renumbered by lib/device_multiplier.py, so this
proves Studio's selection and masking, never throughput. Base cannot keep two models at all.
"""

from __future__ import annotations

import re

TITLE = "Studio puts a model loaded alongside on a second GPU (gfx1151, 1 extra HIP device spoofed)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "nvidia", "multi_gpu", "windows", "mlx"]


def _probe(state: dict) -> dict:
    return state.get("probe") or {}


def _checks(state: dict) -> dict:
    return _probe(state).get("checks") or {}


def _selected(state: dict, alongside: bool):
    for p in (_probe(state).get("info") or {}).get("placements") or []:
        if p.get("alongside") is alongside and p.get("lines"):
            m = re.search(r"selected: (\[[^\]]*\]|None)", p["lines"][-1])
            return m.group(1) if m else None
    return None


def _gpus_seen(state: dict) -> int:
    for p in (_probe(state).get("info") or {}).get("placements") or []:
        for line in p.get("lines") or []:
            m = re.search(r"GPUs free: (\[.*?\])(?:, selected|$)", line)
            if m:
                return len(re.findall(r"\(\s*\d+\s*,", m.group(1)))
    return 0


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        out.append((f"{name}: install.sh --local succeeded", v.get("install_rc") == 0 and v.get("cli_exists"),
                    f"rc={v.get('install_rc')} backend={v.get('llama_backend_forced')} spoof={v.get('spoofed_devices')}"))
        c = _checks(v).get("load_A") or {}
        out.append((f"{name}: a GGUF loads and serves", bool(c.get("ok")), c.get("detail", "no probe")))
    head = obs.get("head") or {}
    n = _gpus_seen(head)
    out.append(("head: Studio saw at least two GPUs (spoof reached the placement path)", n >= 2, f"{n} GPUs in its free list"))
    return out


def table(obs: dict) -> str:
    names = [n for n in obs if not n.startswith("_")]
    keys = sorted({k for n in names for k in _checks(obs[n])})
    rows = ["| check | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in keys:
        cells = []
        for n in names:
            c = _checks(obs[n]).get(k)
            cells.append("-" if c is None else ("PASS" if c["ok"] else "FAIL: " + c["detail"][:80]))
        rows.append(f"| {k} | " + " | ".join(cells) + " |")
    rows.append("| first model on | " + " | ".join(str(_selected(obs[n], False)) for n in names) + " |")
    rows.append("| alongside model on | " + " | ".join(str(_selected(obs[n], True)) for n in names) + " |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    c = _checks(base)
    return bool(c.get("load_A", {}).get("ok")) and not c.get("status_lists_both", {}).get("ok", False)


def head_is_fixed(head: dict) -> bool:
    c = _checks(head)
    first, second = _selected(head, False), _selected(head, True)
    return (bool(c) and all(v["ok"] for v in c.values()) and first not in (None, "None")
            and second not in (None, "None") and first != second)
