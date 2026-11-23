#!/usr/bin/env python3
"""Criteria for probes/acc18200_probe.py (head = current main, base unmeasured).

Gates (INCONCLUSIVE when missing): every arm of every family rendered; the off and bf16 arms ran 0 INT8 attention
calls; scores exist. Worse (REGRESSION = the concern CONFIRMED) for a family where Auto engaged: mean LPIPS(Auto, bf16)
exceeds mean LPIPS(off, bf16) by more than the larger A/A spread (Auto-vs-Auto, off-vs-off LPIPS), plus 0.005."""
from __future__ import annotations

TITLE = "gfx1151 Linux: Auto INT8 attention vs off, each scored against the same platform's bf16 (current main)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "windows", "windows_rocm_wddm"]


def _fams(obs: dict) -> dict:
    return (obs.get("head") or {}).get("families") or {}


def _lp(fr: dict, a: str, b: str, name: str):
    s = (fr.get("scores") or {}).get(f"{name}_{a}|{name}_{b}") or {}
    return s.get("lpips_mean")


def _calls(arm: dict) -> int:
    return sum(x.get("int8_calls") or 0 for x in arm.get("renders") or [])


def gates(obs: dict) -> list:
    fams = _fams(obs)
    out = [("families observed", bool(fams), ",".join(fams) or "none")]
    for name, fr in fams.items():
        arms = fr.get("arms") or {}
        bad = [t for t, x in arms.items() if x.get("verdict") != "ok" or not x.get("renders")]
        out.append((f"{name}: every arm rendered", not bad and len(arms) == 5, f"failed: {bad}" if bad else "5 arms ok"))
        leak = {t: _calls(x) for t, x in arms.items() if t[0] in "OB" and _calls(x)}
        out.append((f"{name}: off / bf16 arms ran no INT8", not leak, str(leak or "0 calls")))
        out.append((f"{name}: scored", _lp(fr, "A1", "B1", name) is not None, "LPIPS A1 vs B1"))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    lines, worse = [], False
    for name, fr in (head.get("families") or {}).items():
        arms = fr.get("arms") or {}
        eng = {t: _calls(x) for t, x in arms.items()}
        ab = [_lp(fr, t, "B1", name) for t in ("A1", "A2")]
        ob = [_lp(fr, t, "B1", name) for t in ("O1", "O2")]
        aa = [_lp(fr, "A2", "A1", name), _lp(fr, "O2", "O1", name)]
        ao = [_lp(fr, "A1", "O1", name), _lp(fr, "A2", "O2", name)]
        if None in ab + ob + aa:
            lines.append(f"{name}: missing scores")
            continue
        noise = max(aa)
        d = sum(ab) / 2 - sum(ob) / 2
        bad = (eng.get("A1", 0) > 0) and d > noise + 0.005
        worse = worse or bad
        lines.append(f"{name}: INT8 calls {eng}; LPIPS Auto-vs-bf16 {ab}, off-vs-bf16 {ob}, Auto-vs-off {ao}, "
                     f"A/A {aa}; delta {d:.4f} vs noise {noise:.4f} -> {'FURTHER' if bad else 'not further'}")
    return worse, "; ".join(lines)
