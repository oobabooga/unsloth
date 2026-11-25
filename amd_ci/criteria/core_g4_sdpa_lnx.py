#!/usr/bin/env python3
"""Criteria: Linux ROCm control for the Windows-only wide-head SDPA guard in Unsloth core (judges
probes/core_g4_sdpa_probe.py). Regression mode: on Linux the guard must not install, and the head must be no worse
than the base (sdpa forward vs eager, and the 20-step LoRA sdpa-vs-eager loss gap)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_cg4", Path(__file__).with_name("core_g4_sdpa_common.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

TITLE = "gfx1151 Linux control: Gemma 4 sdpa vs eager through plain import unsloth (forward + 20-step LoRA)"
MODE = "regression"
NEEDS = _c.NEEDS


def gates(obs):
    out = _c.common_gates(obs)
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out.append(("both on Linux ROCm gfx1151",
                all((o.get("platform") == "linux" and o.get("hip") and o.get("arch") == "gfx1151") for o in (b, h)),
                f"{b.get('platform')} {b.get('hip')} {b.get('arch')}"))
    return out


def head_is_worse(base, head):
    if head.get("guard_installed"):
        return True, "the Windows-only guard installed on Linux"
    hf, hd = _c.fwd_wrong(_c.fwd(head))
    bf, _ = _c.fwd_wrong(_c.fwd(base))
    ht, htd = _c.train_wrong(head)
    bt, _ = _c.train_wrong(base)
    if (hf and not bf) or (ht and not bt):
        return True, f"head forward wrong {hf} ({hd}); head training wrong {ht} ({htd})"
    return False, f"guard not installed; head forward ({hd}); head training ({htd}); wide ops {_c.wide_ops(head)}"


def table(obs):
    return _c.table(obs)
