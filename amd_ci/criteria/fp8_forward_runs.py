#!/usr/bin/env python3
"""Criteria: does unsloth's FP8 block-quant forward run and stay accurate on this AMD GPU?

Defect: on ROCm archs without OCP fp8 (gfx1151 / gfx11xx, gfx90a), the forward
reached the fp8e4nv Triton kernels, which cannot compile there. Fixed: the
forward takes the dequant path and matches a float32 dequant reference.

Pairs with probes/fp8_forward_probe.py.
"""

from __future__ import annotations

TITLE = "FP8 block-quant forward on gfx1151, base versus head"
MODE = "differential"
NEEDS = ["gpu", "rocm"]
TOL = 0.08


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name} probe reached the GPU", bool(o.get("arch")), f"arch={o.get('arch')} hip={o.get('hip')}"))
        out.append((f"{name} ran its own checkout", str(o.get("unsloth_file", "")).startswith(str(o.get("checkout", "?"))),
                    str(o.get("unsloth_file"))))
    return out


def table(obs: dict) -> str:
    rows = ["| state | arch | triton | lacks_kernel | ok | rel_err | error |", "|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        rows.append(f"| {name} | {o.get('arch')} | {o.get('triton')} | {o.get('lacks_kernel')} | {o.get('ok')} "
                    f"| {o.get('rel_err')} | {str(o.get('error', ''))[:160]} |")
    return "\n".join(rows)


def _good(state: dict) -> bool:
    return bool(state.get("ok")) and bool(state.get("finite")) and (state.get("rel_err") or 1.0) < TOL


def base_shows_defect(base: dict) -> bool:
    return not _good(base)


def head_is_fixed(head: dict) -> bool:
    return _good(head)
