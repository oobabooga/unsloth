#!/usr/bin/env python3
"""PR 9672's arch/routing suites, base versus head, on a Windows gfx1151 box.

The judging is `pytest_no_regression`'s and is reused verbatim: compare FAILING
TEST IDS, never counts, and refuse to call a suite that errored at setup, exited
without running, or collected nothing a result.

What is added here is `NEEDS`. The shared module declares `NEEDS = []`, which
renders a report that bounds nothing, and this PR is precisely the shape that
misleads: its largest hunk is `install.sh`, which cannot execute on a box with
no bash, and its new floors are about discrete parts this host does not have.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_shared_path = Path(__file__).resolve().parent / "pytest_no_regression.py"
_spec = importlib.util.spec_from_file_location("amd_ci_pytest_no_regression", _shared_path)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

TITLE = ("PR 9672 AMD arch / ROCm routing suites on Windows gfx1151 "
         "(tests/studio/install, base versus head)")
MODE = "regression"

# Authored. Every capability the CHANGE touches, not the ones this host has.
NEEDS = [
    "windows",            # met: this job is on the Windows half of the pool
    "linux",              # install.sh, the largest hunk, needs a POSIX shell
    "rocm",
    "gpu",
    "discrete_gpu",       # gfx1102 (RX 7600), gfx1200 / gfx1201 (RDNA 4)
    "multi_gpu_amd",      # the per-device AMD visible-mask indexing
    "amd_smi",
    "rocm_smi",
]

gates = _shared.gates
table = _shared.table
head_is_worse = _shared.head_is_worse
