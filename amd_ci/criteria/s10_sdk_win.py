"""Criteria: AMD SDK wheel on Windows gfx1151 (torch 2.11.0+rocm7.14.0 from repo.amd.com/rocm/whl-multi-arch).

Premise under test: that wheel leaves torch.version.hip unset, so the ROCm gates that read only hip switch off.
base shows the defect when hip is unset AND the hip-only gates answer "not ROCm" (s10_common.SDK_BASE_DEFECT).
If the wheel sets hip, base already sees ROCm: VOID, and the facts line says the premise does not hold here.
head is fixed when every changed gate answers ROCm, the fp32 gated-delta op (run only when hip is unset and Triton
imports) finishes fwd + bwd within 1e-4 of fp64 through the torch path, and no test regresses.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s10_common as c  # noqa: E402

TITLE = "ROCm SDK-wheel detection (zoo fla / flex / fused-loss / moe gates + core flex gates), Windows gfx1151"
MODE = "differential"
NEEDS = ["rocm", "gpu", "windows"]


_BASE: dict = {}


def gates(obs):
    _BASE["base"] = obs.get("base") or {}
    return c.common_gates(obs)


def base_shows_defect(base):
    f = c.facts(base)
    return (not f.get("hip")) and not c.gate_mismatch(base, c.SDK_BASE_DEFECT)


def head_is_fixed(head):
    if c.gate_mismatch(head, c.ROCM_ANSWERS):
        return False
    op = head.get("op_fp32") or {}
    if op.get("not_run"):
        if "Triton" not in op["not_run"]:
            return False
    elif not c.op_ok(op):
        return False
    return not c.pytest_regression(_BASE.get("base") or {}, head)


def state_is_fixed(name, obs):
    return head_is_fixed(obs) if name == "head" else None


def table(obs):
    out = [c.gates_table(obs), "", c.pytest_table(obs)]
    if "base" in obs and "head" in obs:
        out.append("")
        out.append(f"- head gates off the ROCm answer: {c.gate_mismatch(obs['head'], c.ROCM_ANSWERS)}")
        out.append(f"- base gates off the SDK defect pattern: {c.gate_mismatch(obs['base'], c.SDK_BASE_DEFECT)}")
        out.append(f"- base op_fp32 died: {c.op_dead(obs['base'].get('op_fp32') or {})}")
        out.append(f"- tests failing at head only: {c.pytest_regression(obs['base'], obs['head'])}")
    return "\n".join(out)
