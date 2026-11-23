#!/usr/bin/env python3
"""Criteria for probes/w8a8_train_probe.py: frozen int8 W8A8 training base (#18782), its backward cache (#18860)
and block swap over a frozen low-precision base (#18869) on gfx1151.

Regression mode, base = #18869's parent, head = #18869's merge commit (training files identical to origin/main).
Head is worse when an arm that completed at the base does not complete at the head, or a head bf16 arm (the
reference) trains to non-finite losses. Everything else (refusal messages, forced-gate arms, speed, loss deltas)
is reported in the table for a human to read: the probe observes, this module only judges the regression question.
"""

from __future__ import annotations

import math
import statistics

TITLE = "gfx1151: frozen int8 W8A8 / fp8 LoRA training bases and low-precision block swap (#18782 #18860 #18869)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "linux", "exclusive_gpu", "nvidia", "discrete_gpu", "windows"]


def _arms(o):
    return (o or {}).get("arms") or {}


def _finite(losses):
    return bool(losses) and all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses)


def _done(a):
    return bool(a.get("complete")) and _finite(a.get("losses")) and not a.get("errors")


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        host = o.get("host") or {}
        arch = str(host.get("arch") or "")
        out.append((f"{name}: probe ran on gfx1151 ROCm", arch.startswith("gfx1151") and bool(host.get("hip")),
                    f"arch={arch!r} hip={host.get('hip')} torch={host.get('torch')}"))
        a = _arms(o).get("bf16_c_a") or {}
        out.append((f"{name}: bf16 reference arm trained", _done(a),
                    f"steps={a.get('steps_seen')} errors={(a.get('errors') or [])[:1]} raised={a.get('raised')}"))
    return out


def _rel_vs(a, ref):
    la, lr = a.get("losses") or [], ref.get("losses") or []
    n = min(len(la), len(lr))
    if not n:
        return None
    return max(abs(la[i] - lr[i]) / max(abs(lr[i]), 1e-12) for i in range(n))


def table(obs):
    rows = ["| state | arm | outcome | engaged / fallback | s/step (med, step>=3) | peak alloc GiB | max rel loss vs bf16_c_a | breaks / recompiles |",
            "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        arms = _arms(obs.get(name))
        ref = arms.get("bf16_c_a") or {}
        for arm, a in arms.items():
            if a.get("skipped"):
                rows.append(f"| {name} | {arm} | skipped: {a['skipped']} | | | | | |")
                continue
            if _done(a):
                outcome = f"trained {a.get('steps_seen')} steps"
            else:
                msg = (a.get("errors") or [a.get("raised") or a.get("stage_error") or f"rc={a.get('_rc')}"])[0]
                outcome = "NOT trained: " + str(msg)[:220].replace("|", "/")
            eng = "; ".join(w[:140].replace("|", "/") for w in (a.get("warnings") or [])
                            if any(k in w for k in ("W8A8", "fp8", "block_swap", "int8")))[:300]
            bs = a.get("int8_bwd_stats_after")
            if isinstance(bs, dict) and bs.get("int8_bwd_hits"):
                eng += f"; int8_bwd hits={bs.get('int8_bwd_hits')} layers={bs.get('int8_bwd_layers')}"
            d = a.get("dynamo") if isinstance(a.get("dynamo"), dict) else {}
            rel = _rel_vs(a, ref) if arm != "bf16_c_a" and _done(a) and _done(ref) else None
            rows.append(f"| {name} | {arm} | {outcome} | {eng} | {a.get('s_per_step_median_from3')} | "
                        f"{a.get('max_allocated_gib')} | {'' if rel is None else f'{rel:.4g}'} | "
                        f"{d.get('graph_breaks')} / {a.get('_log_recompile_lines')} |")
    return "\n".join(rows)


def head_is_worse(base, head):
    bad = []
    ba, ha = _arms(base), _arms(head)
    for arm, b in ba.items():
        h = ha.get(arm)
        if h is None or h.get("skipped") or b.get("skipped"):
            continue
        if _done(b) and not _done(h):
            bad.append(f"`{arm}` trained at the base but not at the head")
    for arm in ("bf16_c_a", "bf16_c_b", "bf16_e"):
        h = ha.get(arm) or {}
        if h and not h.get("skipped") and h.get("losses") and not _finite(h.get("losses")):
            bad.append(f"`{arm}` non-finite losses at the head")
    if bad:
        return True, "; ".join(bad)
    aa = [x for x in (_rel_vs(ha.get("bf16_c_b") or {}, ha.get("bf16_c_a") or {}),) if x is not None]
    return False, ("no arm that trained at the base fails at the head"
                   + (f"; bf16 A/A max rel loss diff {statistics.fmean(aa):.4g}" if aa else ""))
