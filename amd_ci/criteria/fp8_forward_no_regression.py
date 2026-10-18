#!/usr/bin/env python3
"""Criteria: does the head's FP8 block-quant forward still run as well as the base on this AMD GPU?

Regression form of fp8_forward_runs.py, for a host where the base has no defect (ROCm Triton 3.5+
compiles fp8e4nv on every arch): the head must still run, stay finite and stay accurate.

Pairs with probes/fp8_forward_probe.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("fp8_forward_runs", Path(__file__).with_name("fp8_forward_runs.py"))
_runs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runs)
TOL, gates, table = _runs.TOL, _runs.gates, _runs.table

TITLE = "FP8 block-quant forward on gfx1151, no regression"
MODE = "regression"
NEEDS = ["gpu", "rocm"]


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    def good(s):
        return bool(s.get("ok")) and bool(s.get("finite")) and (s.get("rel_err") or 1.0) < TOL

    if good(base) and not good(head):
        return True, f"base ran (rel_err {base.get('rel_err')}), head did not: {head.get('error') or head.get('rel_err')}"
    path = "same kernel path" if base.get("lacks_kernel") == head.get("lacks_kernel") else "kernel path changed"
    return False, f"head runs, rel_err {head.get('rel_err')} vs base {base.get('rel_err')} ({path})"
