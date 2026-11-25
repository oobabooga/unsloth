#!/usr/bin/env python3
"""Criteria for probes/wintiers_probe.py on the LINUX gfx1151 runners: the change must leave Linux untouched.

Regression. The head is worse if its tier decisions differ from the base's, if any family's head arms engage a
different tier than its base arms, if a family's head per-step time is slower than the base by more than
max(A/A spread, 3%), or if the head render moves further from the base (H1 vs B1 LPIPS) than the larger A/A pair
plus 0.01."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wintiers_common as C  # noqa: E402

TITLE = "gfx1151 Linux: RDNA compile / int8 tiers unchanged by the Windows probe gate (base vs head, Studio)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "windows", "nvidia", "multi_gpu"]


def gates(obs: dict) -> list:
    return C.common_gates(obs)


def table(obs: dict) -> str:
    return C.decisions_table(obs) + "\n\n" + C.arms_table(obs.get("head") or {})


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bad = []
    rb = (base.get("decisions") or {}).get("reasons_after_probe")
    rh = (head.get("decisions") or {}).get("reasons_after_probe")
    if rb != rh:
        bad.append(f"decisions differ: base {rb} head {rh}")
    notes = []
    for fam in C.families(head):
        arms = C.fam_arms(head, fam)
        tb = {C.tier_str(C.tier(r)) for r in arms.values() if r.get("side") == "B"}
        th = {C.tier_str(C.tier(r)) for r in arms.values() if r.get("side") == "H"}
        if tb != th:
            bad.append(f"{fam}: tier base {sorted(tb)} head {sorted(th)}")
        t = C.timing(head, fam)
        if t:
            lim = max(t["spread"], 0.03 * t["base"])
            notes.append(f"{fam} {t['base']:.3f} -> {t['head']:.3f} s/step (limit +{lim:.3f})")
            if t["delta"] > lim:
                bad.append(f"{fam}: head slower by {t['delta']:.3f} s/step (> {lim:.3f})")
        hb = C.pair(head, fam, "H1", "B1").get("lpips_mean")
        aa = [x for x in (C.pair(head, fam, "B2", "B1").get("lpips_mean"),
                          C.pair(head, fam, "H2", "H1").get("lpips_mean")) if x is not None]
        if hb is not None and aa and hb > max(aa) + 0.01:
            bad.append(f"{fam}: H1 vs B1 LPIPS {hb} above the A/A {max(aa)} + 0.01")
    if bad:
        return True, "; ".join(bad)
    return False, "same decisions and tiers; " + "; ".join(notes)
