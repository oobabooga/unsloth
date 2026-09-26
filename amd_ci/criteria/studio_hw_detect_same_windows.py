#!/usr/bin/env python3
"""Criteria: Studio hardware detection on a Windows 11 Strix Halo box (Radeon 8060S,
CPU torch), base versus head. Same judgement as studio_hw_detect_same.py, whose
helpers it reuses; only the non-vacuity gate on the base differs.

There is no ROCm torch on these boxes (unmeasured; not installed by this job), so
the expected verdict is a CPU device with a chat-only reason such as
torch_cpu_build, identical at both states. What makes the comparison non-vacuous
here is that the probe really ran on Windows with torch imported, the base's
unmasked detection completed, and the base inventory answered for AMD (the
DirectX registry / amd-smi path the PR's Windows code sits beside).

Pairs with probes/studio_hw_detect_probe.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "amd_ci_studio_hw_detect_same", Path(__file__).with_name("studio_hw_detect_same.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

TITLE = "Studio hardware detection on Windows 11 Strix Halo (CPU torch), base versus head (PR 11944)"
MODE = "regression"
NEEDS = list(_base.NEEDS)

table = _base.table
head_is_worse = _base.head_is_worse


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = _base.probe_gates(obs)
    b = _base.cell(obs.get("base") or {}, "unmasked")
    d, rt = _base.det(b), (b.get("raw_torch") or {})
    system = (b.get("platform") or {}).get("system")
    out.append(("base probe ran on Windows with torch importable",
                system == "Windows" and not rt.get("import_error"),
                f"system={system} torch={rt.get('version')} {rt.get('import_error') or ''}".strip()))
    out.append(("base unmasked detection completed",
                "detect" in b and not d.get("raised"),
                f"device={d.get('device')} reason={d.get('CHAT_ONLY_REASON')} "
                f"{d.get('raised') or ''}".strip()))
    inv = b.get("inventory") or {}
    out.append(("base physical inventory answered for AMD",
                "amd" not in (inv.get("unanswered") or []) and "inventory" in b,
                f"sources={inv.get('sources')} unanswered={inv.get('unanswered')} "
                f"devices={[str(x.get('vendor')) + ':' + str(x.get('name')) for x in inv.get('devices') or []]}"))
    return out
