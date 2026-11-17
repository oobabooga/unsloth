#!/usr/bin/env python3
"""Criteria: the Windows ROCm wide-head SDPA guard (judges probes/gemma4_sdpa_probe.py run with --studio-guard
--model-state all; base = origin/main, head = base + the guard).

base_shows_defect: some kernel arm at Gemma 4's global shapes is WRONG (forward or a gradient non-finite or > 0.05
relative L2 off fp64), or the real Gemma 4 forward with attn_implementation="sdpa" disagrees with "eager" (top-1
agreement < 0.98 or next-token loss > 0.1 nats worse).
head_is_fixed: the guard is installed, no kernel arm is WRONG, and the real sdpa forward matches eager.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_g4", Path(__file__).with_name("gemma4_sdpa.py"))
_g4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_g4)

TITLE = "gfx1151 Windows: wide-head SDPA guard at Gemma 4 global-attention shapes, and a real Gemma 4 forward"
MODE = "differential"
NEEDS = _g4.NEEDS


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        mf = o.get("model_forward") or {}
        measured = [a for a in (o.get("arms") or {}).values() if "out" in a]
        out.append((f"{name}: arms measured", len(measured) >= 12, f"{len(measured)}; {o.get('torch')} {o.get('arch')}"))
        out.append((f"{name}: real forward compared", "top1_agree" in mf, str(mf.get("error") or "")[:200]))
    h = obs.get("head") or {}
    out.append(("head: guard installed, base has none",
                bool(h.get("guard_installed")) and not (obs.get("base") or {}).get("guard_module_present"),
                f"head {h.get('guard_installed')}, base module {(obs.get('base') or {}).get('guard_module_present')}"))
    return out


def base_shows_defect(base):
    wrong = _g4._wrong_set(base)
    mw = _g4._model_wrong(base.get("model_forward") or {})
    return (bool(wrong) or mw, f"{len(wrong)} wrong arms; real forward wrong {mw}")


def head_is_fixed(head):
    wrong = _g4._wrong_set(head)
    mw = _g4._model_wrong(head.get("model_forward") or {})
    return (not wrong and not mw, f"{len(wrong)} wrong arms; real forward wrong {mw}")


def table(obs):
    parts = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        sub = {"head": o}
        parts.append(f"### {name}\n" + _g4.table(sub))
    return "\n\n".join(parts)
