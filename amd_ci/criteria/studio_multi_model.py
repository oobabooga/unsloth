#!/usr/bin/env python3
"""Criteria: can Studio keep two GGUFs loaded and answer each by name (PR 11591)?

Base cannot: an `alongside` load replaces the first model, so /status lists one. Head must pass
every multi_model_probe check on this GPU (load, alongside, both answer, unload one, plain load
replaces).
"""

from __future__ import annotations

TITLE = "Studio serves two GGUF models at once on gfx1151 (Windows 11)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "nvidia", "multi_gpu", "windows", "mlx"]


def _checks(state: dict) -> dict:
    return (state.get("probe") or {}).get("checks") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        out.append((f"{name}: installer --local succeeded", v.get("install_rc") == 0 and v.get("cli_exists"),
                    f"rc={v.get('install_rc')} {v.get('install_s')}s"))
        c = _checks(v).get("load_A") or {}
        out.append((f"{name}: a GGUF loads and serves on this GPU", bool(c.get("ok")), c.get("detail", "no probe")))
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
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    c = _checks(base)
    return bool(c.get("load_A", {}).get("ok")) and not c.get("status_lists_both", {}).get("ok", False)


def head_is_fixed(head: dict) -> bool:
    c = _checks(head)
    return bool(c) and all(v["ok"] for v in c.values())
