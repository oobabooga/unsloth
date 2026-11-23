#!/usr/bin/env python3
"""Criteria for probes/wanrep_probe.py with an --arms-file (AMD coverage, Nov 23).

Groups are tag letters; each group is one shape/config, one fresh process per tag. Regression mode:
  head is worse when, in any TIMED group, the head's median of per-process steady medians exceeds the base median by
  more than the base A/A spread (max - min of the base per-process medians, floor 1 %), or when any group gives MORE
  distinct outputs across processes at the head than at the base (and more than one).
Gates: every arm of both states rendered every render; REQUIRE_PIN: the reduction pin engaged in every head process
and in no base process (the new path provably ran)."""
from __future__ import annotations

import statistics

TITLE = "gfx1151 Linux: Krea-2 Turbo 1024 px compiled steady-state speed, VAE decode and reproducibility, PR 19077 base vs head (timed, exclusive)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "windows", "windows_rocm_wddm"]
GROUPS = {"L": "1024x1024, 8 steps, Studio default, fresh compile cache, 5 renders (compile engages on render 3; renders 4-5 timed)"}        # letter -> description
TIMED = ("L",)          # letters whose steady renders are timed
REQUIRE_PIN = True


def _groups(o: dict) -> dict:
    return (o or {}).get("groups") or {}


def _distinct(x: dict) -> int:
    return len({tuple(f) for f in x.get("md5") or [] if f})


def _within_ok(x: dict) -> bool:
    return all(f is not None and len(f) == 1 for f in x.get("md5") or [])


def _proc_medians(x: dict) -> list:
    # steady_s = renders 1..n-1; Studio engages its deferred compile on generation 3 (index 2), so steady compiled
    # renders are index >= 3 = steady_s[2:].
    return [statistics.median(s[2:]) for s in x.get("steady_s") or [] if len(s) > 2]


def gates(obs: dict) -> list:
    out = []
    for st in ("base", "head"):
        g = _groups(obs.get(st))
        for c in GROUPS:
            x = g.get(c) or {}
            ok = bool(x.get("ok")) and all(x["ok"])
            out.append((f"{st} {c}: every process rendered", ok, f"{x.get('tags')} ok={x.get('ok')} err={[e for e in x.get('errors') or [] if e][:1]}"))
    if REQUIRE_PIN:
        for st, want in (("base", False), ("head", True)):
            pins = [p for c in GROUPS for p in (_groups(obs.get(st)).get(c) or {}).get("pins") or []]
            eng = [bool(p) and any(p) for p in pins]
            ok = bool(eng) and all(e == want for e in eng)
            out.append((f"{st}: reduction pin engaged = {want} in every process", ok, str(pins)))
    return out


def speed(base: dict, head: dict, c: str):
    b = _proc_medians(_groups(base).get(c) or {})
    h = _proc_medians(_groups(head).get(c) or {})
    if not b or not h:
        return None, f"{c}: no timings"
    bm, hm = statistics.median(b), statistics.median(h)
    spread = max(max(b) - min(b), 0.01 * bm)
    return hm <= bm + spread, (f"{c}: base median {bm:.3f}s (A/A spread {spread:.3f}s, per process {[round(v, 3) for v in b]}), "
                               f"head median {hm:.3f}s ({100 * (hm - bm) / bm:+.2f}%, per process {[round(v, 3) for v in h]})")


def head_is_worse(base: dict, head: dict):
    bad, lines = False, []
    for c in GROUPS:
        bd, hd = _distinct(_groups(base).get(c) or {}), _distinct(_groups(head).get(c) or {})
        bw, hw = _within_ok(_groups(base).get(c) or {}), _within_ok(_groups(head).get(c) or {})
        if hd > 1 and hd > bd:
            bad = True
        lines.append(f"{c} ({GROUPS[c]}): distinct outputs across processes base {bd} / head {hd}; "
                     f"within-process repeats identical base {bw} / head {hw}")
    for c in TIMED:
        ok, s = speed(base, head, c)
        if ok is False or ok is None:
            bad = True
        lines.append(s)
    return bad, "; ".join(lines)


def table(obs: dict) -> str:
    rows = ["| state | group | process | md5 (per render set) | load s | first render s | steady s | pins | compiled | status |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for st in ("base", "head"):
        for c in GROUPS:
            x = _groups(obs.get(st)).get(c) or {}
            for i, t in enumerate(x.get("tags") or []):
                stt = (x.get("status") or [None] * 99)[i] or {}
                rows.append(f"| {st} | {c} | {t} | {x['md5'][i]} | {x['load_s'][i]} | {x['first_s'][i]} | {x['steady_s'][i]} | "
                            f"{x['pins'][i]} | {x['blocks_compiled'][i]} | {stt.get('attention_backend')}/{stt.get('transformer_quant')} |")
    for c in GROUPS:
        rows.append(f"\n{c}: {GROUPS[c]}; spec {(_groups(obs.get('head')).get(c) or {}).get('spec')}")
    for c in TIMED:
        rows.append(f"\nspeed {speed(obs.get('base') or {}, obs.get('head') or {}, c)[1]}")
    return "\n".join(rows)
