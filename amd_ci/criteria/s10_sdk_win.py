"""Criteria: AMD SDK wheel on Windows gfx1151 (torch 2.11.0+rocm7.14.0 from repo.amd.com/rocm/whl-multi-arch).

Run 45621361162 measured torch.version.hip = 7.14.60850 on this wheel, so the change is judged as a no-regression run:
  hip set (expected): every changed gate gives the same answer at base and head, and that answer is ROCm's.
  hip unset (the SDK-wheel premise): head gates give ROCm's answers and the fp32 gated-delta op finishes within 1e-4
    of fp64 through the torch path (unless Triton is missing); base differing is the expected fix, reported in detail.
Both: no test fails at the head that did not fail at the base, and the new SDK test file passes at the head.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s10_common as c  # noqa: E402

TITLE = "ROCm SDK-wheel detection (zoo fla / flex / fused-loss / moe gates + core flex gates), Windows gfx1151"
MODE = "regression"
NEEDS = ["rocm", "gpu", "windows"]


def gates(obs):
    out = c.common_gates(obs)
    for st in c.states(obs):
        o = obs[st]
        for k in ("pytest_zoo", "pytest_core"):
            r = o.get(k) or {}
            out.append((f"{st} {k} ran", r.get("rc") in (0, 1) and (r.get("n_passed") or 0) > 0,
                        f"rc {r.get('rc')} passed {r.get('n_passed')} {str(r.get('stderr_tail'))[:200]}"))
    return out


def _new_file_passes(head):
    z = head.get("pytest_zoo") or {}
    mine = lambda ids: [t for t in ids if "test_rocm_sdk_wheel_detection" in t]  # noqa: E731
    return bool(mine(z.get("passed") or [])) and not mine((z.get("failed") or []) + (z.get("errors") or []))


def head_is_worse(base, head):
    bad = []
    hip_set = bool(c.facts(base).get("hip")) and bool(c.facts(head).get("hip"))
    head_off = c.gate_mismatch(head, c.ROCM_ANSWERS)
    if head_off:
        bad.append(f"head gates off the ROCm answer {head_off}")
    if hip_set:
        diff = {k: (c.g(base).get(k), c.g(head).get(k)) for k in c.ROCM_ANSWERS if c.g(base).get(k) != c.g(head).get(k)}
        if diff:
            bad.append(f"gates differ base vs head on a hip-set wheel {diff}")
        mode = "hip set (7.14 SDK wheel): the change is a no-op here"
    else:
        op = head.get("op_fp32") or {}
        if op.get("not_run") and "Triton" not in op["not_run"]:
            bad.append(f"head op_fp32 not run: {op['not_run']}")
        elif not op.get("not_run") and not c.op_ok(op):
            bad.append("head op_fp32 did not finish within 1e-4 of fp64 through the torch path")
        mode = (f"hip UNSET (SDK premise holds): base gates off ROCm {c.gate_mismatch(base, c.ROCM_ANSWERS)}, "
                f"base op_fp32 died {c.op_dead(base.get('op_fp32') or {})}")
    new = c.pytest_regression(base, head)
    if new:
        bad.append("failing at the head only: " + ", ".join(new))
    if not _new_file_passes(head):
        bad.append("tests/test_rocm_sdk_wheel_detection.py did not pass at the head")
    if bad:
        return True, mode + "; " + "; ".join(bad)
    return False, mode + "; every changed gate answers ROCm at the head, no head-only test failures, new SDK test file passes"


def table(obs):
    return c.gates_table(obs) + "\n\n" + c.pytest_table(obs)
