#!/usr/bin/env python3
"""pytest_no_regression, with a NEEDS that bounds an unsloth-zoo checkpointing run.

The shared module declares `NEEDS = []`, which is right for it: it knows nothing about the
change it is judging. `untested_section` renders NEEDS minus the host, so an empty NEEDS
produces a report with no bounds at all, and that has already happened once. This wrapper
adds nothing but the declaration and the title; the judging is the shared module's, loaded
by path so there is no second copy of it to drift.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_path = Path(__file__).with_name("pytest_no_regression.py")
_spec = importlib.util.spec_from_file_location("amd_ci_pytest_no_regression", _path)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

TITLE = "unsloth-zoo PR 1316: the rest of the gradient-checkpointing suites, base vs head"
MODE = _m.MODE

# Same list as criteria/gc_offload_dtype.py: these suites cover the same code, so they
# inherit the same bounds. The offload path is GPU-only; it is reached on ROCm here but
# the same lines run on CUDA and XPU; the reported crash needs a GPU without bf16;
# double buffering is on by default only on a discrete GPU; the buffers are per-device;
# the PR's own evidence is Windows.
NEEDS = ["gpu", "rocm", "nvidia", "xpu", "no_bf16_gpu", "discrete_gpu", "multi_gpu", "windows"]

gates = _m.gates
table = _m.table
head_is_worse = _m.head_is_worse
