#!/usr/bin/env python3
"""Criteria (T7h.5e): unsloth #16071 fused LoRA order on gfx1151, base vs head. Pairs with probes/t7h5e_lora_probe.py.

#16071 deliberately changes bf16 rounding (the rank-r term is added inside the base GEMM's fp32 epilogue), so the head
with the flag on is judged against an fp32 reference run of the same scenario, not for equality with the base.

Gates (INCONCLUSIVE when any fails): every run finished from its own checkout on a HIP gfx1151 GPU; the base fp32
reference ran for every 16-bit model; in the head's count arms the fused order engaged with the flag on
(_lora_fused_ok True > 0) and never with UNSLOTH_LORA_FUSED_PASSES=0; the timed arms carry no counter wrapper.
REGRESSION when, for any model:
  - a head run raised or produced a non-finite loss / grad norm;
  - 16-bit model, flag on: the head's mean relative per-step loss distance to the fp32 reference exceeds the base's
    by more than REF_SLACK (relative);
  - the kill switch differs from base by more than KILL_TOL relative at any step (it runs main's order);
  - head peak memory > base * 1.05 or median step time (steps 3+) > base * STEP_TOL;
  - greedy tokens differ from base at a step whose top-2 margin is > 0.05 in both runs (not a near-tie);
  - the head-only test file tests/test_lora_fused_passes.py failed.
"""

from __future__ import annotations

import math
import os
import statistics

TITLE = "T7h.5e: unsloth #16071 fused LoRA order, LoRA SFT + generate on gfx1151 (base vs head vs kill switch vs fp32 ref)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "nvidia", "multi_gpu"]
KILL_TOL = 0.02
REF_SLACK = 0.10
STEP_TOL = 1.10


def _norm(p):
    return os.path.normcase(os.path.abspath(p)) if p else ""


def _sc(r):
    return (r or {}).get("scenario") or {}


def _med(tr):
    ms = (tr or {}).get("step_ms") or []
    return statistics.median(ms[3:]) if len(ms) > 3 else None


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-6)


def _ref_dist(losses, ref):
    d = [_rel(x, y) for x, y in zip(losses, ref)]
    return sum(d) / len(d) if d else None


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for st in ("base", "head"):
        s = obs.get(st) or {}
        runs = s.get("runs") or {}
        ok = bool(runs) and all(r.get("rc") == 0 and _sc(r).get("done") for r in runs.values())
        out.append((f"{st}: every run finished", ok,
                    "; ".join(f"{k} rc={r.get('rc')} err={str(_sc(r).get('error'))[:200]}" for k, r in runs.items())))
        own = bool(runs) and all(_norm(_sc(r).get("unsloth_file")).startswith(_norm(s.get("checkout"))) for r in runs.values())
        out.append((f"{st}: unsloth imported from its own checkout", own,
                    f"{sorted({str(_sc(r).get('unsloth_file')) for r in runs.values()})} vs {s.get('checkout')}"))
        t = [_sc(r).get("torch") or {} for r in runs.values()]
        out.append((f"{st}: HIP gfx1151", bool(t) and all(x.get("hip") and "gfx1151" in str(x.get("arch")) for x in t), f"{t[:1]}"))
        timed = [k for k in runs if k.split("|")[1] in ("flag1", "flag0", "fp32")]
        clean = all(not _sc(runs[k]).get("count_arm") for k in timed)
        out.append((f"{st}: timed / loss arms run without the counter wrapper", bool(timed) and clean, f"{timed}"))
    br = (obs.get("base") or {}).get("runs") or {}
    b16 = [k.split("|")[0] for k, r in br.items() if k.endswith("|flag1") and not _sc(r).get("load_in_4bit")]
    refs = all(_sc(br.get(f"{m}|fp32")).get("dtype") == "float32" and (_sc(br.get(f"{m}|fp32")).get("train") or {}).get("losses")
               for m in b16)
    out.append(("base: fp32 reference ran for every 16-bit model", bool(b16) and refs, f"{b16}"))
    hr = (obs.get("head") or {}).get("runs") or {}
    on = [r for k, r in hr.items() if k.endswith("|count1")]
    off = [r for k, r in hr.items() if k.endswith("|count0")]
    eng = bool(on) and all((_sc(r).get("fused_calls") or {}).get("fused_true", 0) > 0 for r in on)
    dis = bool(off) and all((_sc(r).get("fused_calls") or {}).get("fused_true", 1) == 0 and
                            (_sc(r).get("fused_calls") or {}).get("fused_false", 0) > 0 for r in off)
    out.append(("head: fused order engaged with the flag on, never with it off (count arms)", eng and dis,
                f"on {[_sc(r).get('fused_calls') for r in on]} off {[_sc(r).get('fused_calls') for r in off]}"))
    return out


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bad, notes = [], []
    br, hr = base.get("runs") or {}, head.get("runs") or {}
    for key, b in br.items():
        if not key.endswith("|flag1"):
            continue
        model = key.split("|")[0]
        bs = _sc(b)
        btr = bs.get("train") or {}
        bl = btr.get("losses") or []
        ref = (_sc(br.get(f"{model}|fp32")).get("train") or {}).get("losses")
        for flag in ("flag1", "flag0"):
            h = _sc(hr.get(f"{model}|{flag}"))
            htr = h.get("train") or {}
            if h.get("error") or not htr:
                bad.append(f"{model} {flag}: {str(h.get('error'))[:300]}")
                continue
            if not htr.get("finite"):
                bad.append(f"{model} {flag}: non-finite loss / grad norm")
            hl = htr.get("losses") or []
            d = max((_rel(x, y) for x, y in zip(hl, bl)), default = None)
            notes.append(f"{model} {flag}: max rel loss diff vs base {d}, final {hl[-1:]} vs {bl[-1:]}")
            if flag == "flag0" and (d is None or d > KILL_TOL):
                bad.append(f"{model} flag0: loss differs from base by {d} (> {KILL_TOL})")
            if flag == "flag1" and ref:
                dh, db = _ref_dist(hl, ref), _ref_dist(bl, ref)
                notes.append(f"{model} flag1: mean rel distance to fp32 ref head {dh} vs base {db}")
                if dh is None or db is None or dh > db * (1 + REF_SLACK):
                    bad.append(f"{model} flag1: further from the fp32 reference than base ({dh} vs {db})")
            if htr.get("peak_mib", 0) > btr.get("peak_mib", 0) * 1.05:
                bad.append(f"{model} {flag}: peak {htr.get('peak_mib'):.0f} vs base {btr.get('peak_mib'):.0f} MiB")
            hm, bm = _med(htr), _med(btr)
            notes.append(f"{model} {flag}: median step {hm} vs base {bm} ms, peak {htr.get('peak_mib', 0):.0f} vs {btr.get('peak_mib', 0):.0f} MiB")
            if not hm or not bm or hm > bm * STEP_TOL:
                bad.append(f"{model} {flag}: median step {hm} ms vs base {bm} ms")
            bg, hg = bs.get("generate") or {}, h.get("generate") or {}
            bt, ht = bg.get("tokens") or [], hg.get("tokens") or []
            for i, (x, y) in enumerate(zip(bt, ht)):
                if x != y:
                    bm_, hm_ = bg.get("margins") or [], hg.get("margins") or []
                    m = min(bm_[i] if i < len(bm_) else math.inf, hm_[i] if i < len(hm_) else math.inf)
                    notes.append(f"{model} {flag}: tokens diverge at {i} (margin {m})")
                    if m > 0.05:
                        bad.append(f"{model} {flag}: greedy tokens differ at step {i} with margin {m}")
                    break
    hp = head.get("head_only_pytest") or {}
    if not hp or hp.get("rc") != 0:
        bad.append(f"tests/test_lora_fused_passes.py did not pass at the head: {str(hp.get('tail'))[-600:]}")
    notes.append(f"head-only pytest rc={hp.get('rc')}: {str(hp.get('tail'))[-200:]}")
    if bad:
        return True, "; ".join(bad) + " | " + "; ".join(notes)
    return False, "; ".join(notes)


def table(obs: dict) -> str:
    rows = ["| state | run | dtype | final loss | median step ms | peak MiB | fused calls | dynamo (breaks / graphs) | tokens[:8] |",
            "|---|---|---|---|---|---|---|---|---|"]
    for st in ("base", "head", "merge"):
        for k, r in ((obs.get(st) or {}).get("runs") or {}).items():
            s = _sc(r)
            tr = s.get("train") or {}
            dy = s.get("dynamo_after_train") or {}
            rows.append(f"| {st} | {k} | {s.get('dtype')} | {(tr.get('losses') or [None])[-1]} | {_med(tr)} | "
                        f"{tr.get('peak_mib')} | {s.get('fused_calls')} | {dy.get('graph_break')} / {dy.get('unique_graphs')} | "
                        f"{((s.get('generate') or {}).get('tokens') or [])[:8]} |")
    return "\n".join(rows)
