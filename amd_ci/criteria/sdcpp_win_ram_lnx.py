#!/usr/bin/env python3
"""Criteria for probes/sdcpp_win_ram_probe.py on Linux gfx1151: no regression. Linux already reads host RAM from
/proc/meminfo without psutil, so head must emit exactly the base's offload flags in both Python conditions, read the
same total RAM, and render them without failure."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdcpp_win_ram as W  # noqa: E402

TITLE = "sd.cpp offload CPU VAE tile, Linux no-regression for the Windows RAM fallback, gfx1151"
MODE = "regression"
NEEDS = W.NEEDS
gates = W.gates
table = W.table


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    notes = []
    for cond in ("nopsutil", "psutil"):
        fb, fh = W._ch(base, cond).get("flags"), W._ch(head, cond).get("flags")
        if fb != fh:
            return True, f"{cond}: head flags {fh} != base flags {fb}"
        rb = (W._ch(base, cond).get("state_readings") or {}).get("diffusion_small_host.host_ram_mib")
        rh = (W._ch(head, cond).get("state_readings") or {}).get("diffusion_small_host.host_ram_mib")
        if not (isinstance(rb, list) and isinstance(rh, list) and rb[0] == rh[0]):
            return True, f"{cond}: head total RAM {rh} vs base {rb}"
        for size in head.get("sizes") or []:
            r = W._run(head, W._arm(head, cond, size))
            if not W._ok(r):
                return True, f"{cond} {size}px: head render rc {r.get('rc')}"
        notes.append(f"{cond}: same flags {fh}, total RAM {rh[0]} MiB")
    return False, "; ".join(notes)
