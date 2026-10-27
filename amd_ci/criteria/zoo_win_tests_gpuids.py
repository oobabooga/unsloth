#!/usr/bin/env python3
"""Criteria: zoo_win_tests.py, except the "a GPU test ran" gate also counts tests that can only run on a GPU (their
own skipif(not torch.cuda.is_available()) or a module-level one), named below by node-id prefix. The stock gate only
counts ids with "cuda" in their parameters, and these files pick the device inside the test body, so it read 0 on a
GPU host whose GPU-only tests passed. A GPU-only test counts only when it RAN (passed or failed), never when skipped.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_base_path = Path(__file__).resolve().parents[2] / "amd_ci" / "criteria" / "zoo_win_tests.py"
_spec = importlib.util.spec_from_file_location("zoo_win_tests", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

TITLE = _base.TITLE + " (GPU-only tests named by id)"
MODE = _base.MODE
NEEDS = _base.NEEDS
table = _base.table
head_is_worse = _base.head_is_worse

# Each is GPU-only in the zoo source at 9910e542 (skipif not cuda.is_available, module pytestmark, or "cuda" devices).
GPU_ONLY = (
    "tests/test_fused_linear_collapse.py::",  # pytestmark skipif(not torch.cuda.is_available())
    "tests/test_fused_linear_jsd.py::test_bf16_error_not_worse_than_trl_bf16",  # skipif not cuda
    "tests/test_chunked_logprob.py::test_fp32_output_gemm_only_where_fast",  # skipif not cuda; device="cuda"
    "tests/test_grpo_logprob_recompute.py::test_cuda_autocast_matches_plain_function",  # skipif not cuda
)


def _gpu_ran(o: dict) -> list:
    out = (o.get("outcomes") or {})
    return [k for k, v in out.items() if k.startswith(GPU_ONLY) and v in ("passed", "failed")]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    g = _base.gates(obs)
    head = obs.get("head") or {}
    ran = sorted(set(_base._cuda_ran(head)) | set(_gpu_ran(head)))
    return [(("head: >= 1 GPU test ran (cuda-parametrised or GPU-only by source)", len(ran) > 0,
              f"{len(ran)} ran: " + ", ".join(r.split("::")[-1] for r in ran[:6]))
             if name.startswith("head: >= 1 cuda-parametrised") else (name, ok, ev))
            for name, ok, ev in g]
