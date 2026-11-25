"""Criteria: Linux ROCm gfx1151 (torch 2.11.0+rocm7.13.0, torch.version.hip set): the change must be a no-op.

Every changed gate already answers ROCm at the base (hip is set) and must answer the same at the head; no test that
did not fail at the base fails at the head (zoo tests in the checkout, core tests in s10_core/<state>); the tests new
at the head (test_rocm_sdk_wheel_detection.py, the *sdk* core tests) pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s10_common as c  # noqa: E402

TITLE = "ROCm SDK-wheel detection, Linux gfx1151 (hip set): no-regression"
MODE = "regression"
NEEDS = ["rocm", "gpu", "linux"]


def gates(obs):
    out = c.common_gates(obs)
    for st in c.states(obs):
        o = obs[st]
        out.append((f"{st} torch.version.hip is set (the Linux arm models a classic ROCm wheel)", bool(c.facts(o).get("hip")),
                    str(c.facts(o).get("hip"))))
        for k in ("pytest_zoo", "pytest_core"):
            r = o.get(k) or {}
            out.append((f"{st} {k} ran", r.get("rc") in (0, 1) and (r.get("n_passed") or 0) > 0,
                        f"rc {r.get('rc')} passed {r.get('n_passed')} {str(r.get('stderr_tail'))[:200]}"))
    head = obs.get("head") or {}
    hp = set((head.get("pytest_zoo") or {}).get("passed") or [])
    out.append(("head ran the new SDK test file", any("test_rocm_sdk_wheel_detection" in t for t in hp), f"{len(hp)} zoo passed"))
    return out


def head_is_worse(base, head):
    bad = []
    if c.gate_mismatch(base, c.ROCM_ANSWERS):
        bad.append(f"base gates off the ROCm answer {c.gate_mismatch(base, c.ROCM_ANSWERS)}")
    if c.gate_mismatch(head, c.ROCM_ANSWERS):
        bad.append(f"head gates off the ROCm answer {c.gate_mismatch(head, c.ROCM_ANSWERS)}")
    z = head.get("pytest_zoo") or {}
    if [t for t in (z.get("failed") or []) + (z.get("errors") or []) if "test_rocm_sdk_wheel_detection" in t]:
        bad.append("tests/test_rocm_sdk_wheel_detection.py failed at the head")
    new = c.pytest_regression(base, head)
    if new:
        bad.append("failing at the head only: " + ", ".join(new))
    if bad:
        return True, "; ".join(bad)
    return False, "every changed gate answers ROCm at both states; no test fails at the head that did not fail at the base"


def table(obs):
    return c.gates_table(obs) + "\n\n" + c.pytest_table(obs)
