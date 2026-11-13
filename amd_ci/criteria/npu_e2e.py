#!/usr/bin/env python3
"""Criteria: with a real FastFlowLM model generating on the NPU, does the head's live monitor read
the NPU busy (agreeing with an independent counter read) and size the model in system RAM?
Pairs with probes/npu_e2e_probe.py."""

from __future__ import annotations

TITLE = "AMD NPU live monitor, real FastFlowLM load on the NPU"
MODE = "differential"
NEEDS = ["windows"]


def _busy(samples: list[dict], key: str) -> list[float]:
    return [s[key] for s in samples if isinstance(s.get(key), (int, float))]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    head = obs.get("head") or {}
    gens = head.get("generations") or []
    raw = _busy(head.get("samples") or [], "raw")
    return [
        ("head drove a real NPU load", bool(gens) and all(g["status"] == 200 for g in gens),
         head.get("error") or f"{len(gens)} generations: {gens[:3]}"),
        ("independent counter saw the NPU busy", bool(raw) and max(raw) > 1.0,
         f"raw max={max(raw) if raw else None} over {len(raw)} samples"),
    ]


def table(obs: dict) -> str:
    head = obs.get("head") or {}
    s = head.get("samples") or []
    ours, raw = _busy(s, "ours"), _busy(s, "raw")
    rows = [
        "| quantity | value |", "|---|---|",
        f"| base has npu_load_percent | {(obs.get('base') or {}).get('has_npu_load_percent')} |",
        f"| LUID | {head.get('luid')} |",
        f"| catalog size (MiB) | {head.get('catalog_mib')} |",
        f"| resident (MiB, on_gpu) loaded idle | {head.get('resident_loaded')} |",
        f"| reading, loaded idle | {head.get('idle_loaded_reading')} |",
        f"| head reading during generation: n / max / median | {len(ours)} / {max(ours) if ours else None} / {sorted(ours)[len(ours) // 2] if ours else None} |",
        f"| independent reading: n / max / median | {len(raw)} / {max(raw) if raw else None} / {sorted(raw)[len(raw) // 2] if raw else None} |",
        f"| reading after unload | {head.get('after_unload_reading')} |",
        f"| resident after unload | {head.get('after_unload_resident')} |",
        f"| generations | {head.get('generations')} |",
    ]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return base.get("has_npu_load_percent") is False


def head_is_fixed(head: dict) -> bool:
    s = head.get("samples") or []
    ours, raw = _busy(s, "ours"), _busy(s, "raw")
    if not ours or not raw or max(ours) <= 1.0:
        return False
    # Same source as Task Manager: the peaks agree to within a few points.
    if abs(max(ours) - max(raw)) > max(10.0, 0.25 * max(raw)):
        return False
    resident = head.get("resident_loaded") or [None, None]
    floor = head.get("catalog_mib") or 0
    if not resident[0] or resident[0] < floor or resident[1] is not False:
        return False
    after = head.get("after_unload_resident") or [None, None]
    return after[0] is None
