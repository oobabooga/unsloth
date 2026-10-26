#!/usr/bin/env python3
"""Criteria: the SAME Windows observations as zoo_win_tests.py, judged as a clearly labelled CPU run.

Used (via lib/redecide.py, no re-probing) when the Windows torch exposes no GPU: it then still answers "does the head
import and run the fused linear torch fallback on Windows". Gates: everything in zoo_win_tests.common_gates, plus the
GPU must really be ABSENT (so a GPU host can never be reported as a CPU pass) and >= 1 cpu-parametrised test passed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("zoo_win_tests", Path(__file__).with_name("zoo_win_tests.py"))
_g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g)

TITLE = "CPU RUN ON WINDOWS (no GPU visible to torch): unsloth-zoo fused-linear unit tests, zoo base vs head"
MODE = "regression"
NEEDS = ["windows", "gpu", "rocm", "discrete_gpu", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]
table = _g.table
head_is_worse = _g.head_is_worse


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    head = obs.get("head") or {}
    t = head.get("torch") or {}
    out = _g.common_gates(obs)
    out.append(("head: labelled CPU run (torch sees NO GPU)", t.get("is_available") is False,
                f"is_available={t.get('is_available')} torch={t.get('version')} hip={t.get('hip')}"))
    n_pass, n_fail = head.get("n_passed") or 0, (head.get("n_failed") or 0) + (head.get("n_errors") or 0)
    out.append(("head: tests ran on CPU", n_pass + n_fail > 0, f"{n_pass} passed, {n_fail} failed/errored"))
    return out
