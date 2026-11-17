#!/usr/bin/env python3
"""Criteria: Windows ROCm wide-head SDPA guard in Unsloth core (judges probes/core_g4_sdpa_probe.py).

base_shows_defect: through plain `import unsloth`, Gemma 4's sdpa forward disagrees with eager (top-1 < 0.98 or
next-token loss > 0.1 nats worse) or the 20-step LoRA loss curve with sdpa departs from eager's.
head_is_fixed: the guard is installed, the wide-head attention ops in the sdpa forward are math only, the forward
matches eager and the sdpa loss curve tracks eager's.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_cg4", Path(__file__).with_name("core_g4_sdpa_common.py"))
_c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c)

TITLE = "gfx1151 Windows: Gemma 4 sdpa vs eager through plain import unsloth (forward + 20-step LoRA)"
MODE = "differential"
NEEDS = _c.NEEDS


def gates(obs):
    out = _c.common_gates(obs)
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out.append(("head: guard installed by import unsloth, base has none",
                bool(h.get("guard_installed")) and not b.get("guard_installed"),
                f"head {h.get('guard_installed')}, base {b.get('guard_installed')}"))
    out.append(("both on Windows ROCm gfx1151",
                all((o.get("platform") == "win32" and o.get("hip") and o.get("arch") == "gfx1151") for o in (b, h)),
                f"{b.get('platform')} {b.get('hip')} {b.get('arch')}"))
    return out


def base_shows_defect(base):
    fw, fd = _c.fwd_wrong(_c.fwd(base))
    tw, td = _c.train_wrong(base)
    return (fw or tw, f"forward wrong {fw} ({fd}); training wrong {tw} ({td}); wide ops {_c.wide_ops(base)}")


def head_is_fixed(head):
    fw, fd = _c.fwd_wrong(_c.fwd(head))
    tw, td = _c.train_wrong(head)
    ops = _c.wide_ops(head)
    math_only = bool(ops) and set(ops) == {"_scaled_dot_product_attention_math"}
    return (not fw and not tw and math_only and bool(head.get("guard_installed")),
            f"forward wrong {fw} ({fd}); training wrong {tw} ({td}); wide ops {ops}")


def table(obs):
    return _c.table(obs)
