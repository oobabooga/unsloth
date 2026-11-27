#!/usr/bin/env python3
"""Criteria for probes/wintiers_probe.py on the WINDOWS gfx1151 runners: the compiled RDNA int8 tiers, gated on the
regional compile probe instead of sys.platform.

Differential. Defect at the base: on Windows the regional compile probe passes and the RDNA W8A8 kernels are ready,
yet the base refuses the measured tiers (FLUX.3 Klein explicit int8 compiled, Qwen-Image-2.1 AUTO W8A8): its own
decisions say no. Fixed at the head: the head's decisions say yes, its Klein explicit-int8 arms run compiled with
W8A8 and are faster per step than the base arms by more than the A/A spread, within LPIPS 0.05 of the base render
(int8 W8A8 vs weight-only: Linux measured 0.012 against bf16), every family's head arms rendered, and Wan (not
touched by this change) keeps its tier and stays within its A/A spread or 3%."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wintiers_common as C  # noqa: E402

TITLE = "gfx1151 Windows: RDNA compile / int8 tiers gated on the compile probe (base vs head, Studio)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows", "exclusive_gpu", "nvidia", "multi_gpu"]
LPIPS_BOUND = 0.05


def gates(obs: dict) -> list:
    return C.common_gates(obs)


def table(obs: dict) -> str:
    return C.decisions_table(obs) + "\n\n" + C.arms_table(obs.get("head") or {})


def _decided(o: dict, fam: str, key: str) -> bool:
    return bool((((o.get("decisions") or {}).get("reasons_first") or {}).get(fam) or {}).get(key))


def base_shows_defect(base: dict) -> bool:
    d = base.get("decisions") or {}
    ready = d.get("platform") == "win32" and d.get("torch_compile_runtime_available") and d.get(
        "rocm_rdna_w8a8_ready") and (d.get("compile_probe") or {}).get("ok")
    refused = not _decided(base, "flux.3-klein", "explicit_int8") and not _decided(base, "qwen-image-2.1", "auto_w8a8")
    return bool(ready and refused)


def head_is_fixed(head: dict) -> bool:
    if not (_decided(head, "flux.3-klein", "explicit_int8") and _decided(head, "flux.3-klein", "explicit_w8a8")
            and _decided(head, "qwen-image-2.1", "auto_w8a8")):
        return False
    k = C.fam_arms(head, "klein")
    if k:  # a job may carry only some families; each present one is judged
        if not all(C.tier(r)["compiled"] and C.tier(r)["w8a8"] for t, r in k.items() if r.get("side") == "H"):
            return False
        t = C.timing(head, "klein")
        if not t or -t["delta"] <= t["spread"]:
            return False
        lp = C.pair(head, "klein", "H1", "B1").get("lpips_mean")
        if lp is None or lp > LPIPS_BOUND:
            return False
    if not C.families(head):
        return False
    w = C.fam_arms(head, "wan")
    if w:
        tiers = {C.tier_str(C.tier(r)) for r in w.values()}
        tw = C.timing(head, "wan")
        if len(tiers) != 1 or not tw or abs(tw["delta"]) > max(tw["spread"], 0.03 * tw["base"]):
            return False
    return True
