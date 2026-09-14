#!/usr/bin/env python3
"""PR 9672 install suites: same judgement as pytest_no_regression, real NEEDS.

The judgement is unchanged -- compare FAILED/ERROR ids by set, never by count --
so it is imported rather than restated. What this module adds is `NEEDS`.

`pytest_no_regression.NEEDS` is `[]`, which is honest for a generic suite runner
and useless for a specific one: `untested_section` renders the gaps as NEEDS
minus the host, so an empty NEEDS yields no gaps and the report bounds nothing.

PR 9672 indexes AMD visible-device masks BY DEVICE and floors gfx1102 / RDNA4 to
rocm6.4. The tests selected here (`tests/studio/install`) are pure-Python unit
tests over `install.sh`, `studio/setup.sh` and `studio/install_python_stack.py`:
they read the scripts and assert on the routing logic. Nothing in them installs a
wheel, and this leg deliberately masks the GPU, so the capabilities the CHANGE is
about are all untested regardless of how green the suite is. Declaring them is
what stops "4086 passed" reading as "the per-device mask works".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pytest_no_regression import (  # noqa: E402,F401
    MODE,
    gates,
    head_is_worse,
    table,
)

TITLE = "PR 9672 install suites (tests/studio/install), base versus head"

NEEDS = [
    # The change is entirely about the ROCm install route.
    "rocm",
    # Masked to "" in this job on purpose, so no device is exercised at all.
    "gpu",
    # The point of the change: one visible-device mask PER DEVICE. One GPU cannot
    # show an index being used for the wrong device.
    "multi_gpu",
    "multi_gpu_amd",
    # The rocm6.4 floor is for gfx1102 and RDNA4. This runner is gfx1151, an
    # integrated APU, so neither arch is present to route.
    "discrete_gpu",
    # install.sh also carries the NVIDIA and the Windows routes, which the same
    # edits sit next to.
    "nvidia",
    "windows",
]
