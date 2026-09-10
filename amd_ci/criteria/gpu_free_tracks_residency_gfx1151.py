#!/usr/bin/env python3
"""Criteria: on gfx1151 Linux, does the reported free VRAM track physical residency?

The same question `gpu_summary_sees_others` asks, bound to one part and to one
extra precondition, and kept in a separate file for a reason: that module is
generically named and reusable, and folding a `gfx1151` gate into it would make
every future caller inherit an assertion about silicon they are not testing.
This one extends it instead.

What it adds:

  the host is an integrated gfx1151 part
      A summary that tracks the driver on a discrete card says nothing about a
      unified pool. Anything else is INCONCLUSIVE rather than a quiet pass on
      the wrong hardware.

  the driver's own free tracks the held allocation
      The comparison is AGAINST the driver's free figure, so it is worth nothing
      where that figure does not itself respond to a resident allocation. On an
      integrated part the pool is system RAM and whether `hipMemGetInfo` free
      moves at all is exactly what is in doubt (ROCm/ROCm#5595,
      ggml-org/llama.cpp#18159). The fixture reads free before and after
      committing, so this is checked rather than assumed.

It also renders the kernel's own accounting (the TTM page ceiling and amdgpu's
GTT / VRAM totals and used counters) beneath the comparison. Which of those the
driver's reported total actually IS cannot be read off `mem_get_info`, and it is
the quantity a planner would be budgeting against.

Pairs with probes/gpu_summary_probe.py and probes/vram_holder_fixture.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Loaded by path, not imported by name: differential.py loads a criteria module
# with spec_from_file_location, so `criteria/` is not a package and never lands
# on sys.path. Going through the file keeps this working wherever it is invoked
# from, including the runner's checkout.
_BASE_PATH = Path(__file__).with_name("gpu_summary_sees_others.py")
_spec = importlib.util.spec_from_file_location(
    "amd_ci_criteria_gpu_summary_sees_others", _BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

GIB = base.GIB
TITLE = "Reported free VRAM against physical residency on gfx1151"
MODE = base.MODE
# Wider than the base module's, because this run is also a claim about which
# quantity the reported total is, which is a ROCm and integrated-GPU question.
NEEDS = ["gpu", "rocm", "integrated_gpu", "windows", "windows_rocm_wddm",
         "multi_gpu", "nvidia", "mig", "xpu", "mlx"]

# The part this is a claim about.
EXPECT_ARCH = "gfx1151"

# The base module keeps the holder size and tolerance in a module-level stash
# that its own gates() populates. Delegating to it below is what keeps the two
# in step; recomputing them here would be a second copy of the tolerance rule.
_CTX = base._CTX


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    # First, so _CTX is populated for everything below and for the base module's
    # own base_shows_defect / head_is_fixed.
    out = list(base.gates(obs))

    fixture = obs.get("_fixture") or {}
    states = _states(obs)
    held = _CTX["held"]

    # Read off the head where possible; every state ran on the same box.
    ref = states.get("head") or (next(iter(states.values())) if states else {})
    arch = str(ref.get("arch") or "")
    integrated = ref.get("is_integrated")
    out.append((f"the host is an integrated {EXPECT_ARCH} part",
                arch.startswith(EXPECT_ARCH) and bool(integrated),
                f"arch={arch or '-'} is_integrated={integrated} "
                f"torch={ref.get('torch_version', '-')} hip={ref.get('torch_hip', '-')}"))

    before = fixture.get("driver_free_before_gib")
    after = fixture.get("driver_free_gib")
    drop = fixture.get("driver_free_drop_gib")
    moved = drop is not None and held > 0 and drop >= 0.5 * held
    out.append(("the driver's own free tracks the held allocation",
                bool(moved),
                f"free {before if before is None else f'{before:.2f}'} -> "
                f"{after if after is None else f'{after:.2f}'} GiB, "
                f"drop {drop if drop is None else f'{drop:.2f}'} GiB against "
                f"{held:.2f} GiB held"))
    return out


def table(obs: dict) -> str:
    rows = [base.table(obs)]
    kernel = next((v.get("kernel") for v in _states(obs).values() if v.get("kernel")), None)
    if kernel:
        rows += ["", "Kernel's own view, for the record. Which of these the driver's "
                     "reported total actually is cannot be read off `mem_get_info`:",
                 "", "| source | GiB |", "|---|---|"]
        limit = kernel.get("ttm_pages_limit_gib")
        rows.append(f"| ttm pages_limit | {'-' if limit is None else f'{limit:.2f}'} |")
        for card, row in (kernel.get("cards") or {}).items():
            for name in ("mem_info_gtt_total", "mem_info_gtt_used",
                         "mem_info_vram_total", "mem_info_vram_used"):
                value = row.get(f"{name}_gib")
                if value is not None:
                    rows.append(f"| {card} {name} | {value:.2f} |")
    return "\n".join(rows)


# The comparison itself is the base module's, unchanged. This file narrows WHERE
# the answer counts, not WHAT counts as an answer.
base_shows_defect = base.base_shows_defect
head_is_fixed = base.head_is_fixed
