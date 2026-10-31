#!/usr/bin/env python3
"""Criteria (T7h.5c): unsloth #16071 fused LoRA order on gfx1151, base vs head. Pairs with probes/t7h5c_lora_probe.py.

Gates (INCONCLUSIVE when any fails): every run finished from its own checkout on a HIP gfx1151 GPU, and at the head the
fused order engaged with the flag on (_lora_fused_ok True > 0) and never with UNSLOTH_LORA_FUSED_PASSES=0.
REGRESSION when, for any model:
  - a head run raised or produced a non-finite loss / grad norm;
  - head (fused) per-step loss differs from base by more than LOSS_TOL relative at any step (bf16 noise bound);
  - the kill switch differs from base by more than KILL_TOL (it runs main's order);
  - head peak memory > base * 1.05 or median step time > base * 1.10;
  - greedy tokens differ from base at a step whose top-2 margin is > 0.05 in both runs (not a near-tie);
  - the head-only test file tests/test_lora_fused_passes.py failed.
"""

from __future__ import annotations

import math
import os
import statistics

TITLE = "T7h.5c: unsloth #16071 fused LoRA order, bf16 LoRA SFT + generate on gfx1151 (base vs head vs kill switch)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "nvidia", "multi_gpu"]
LOSS_TOL = 0.02
KILL_TOL = 0.005


def _norm(p):
    return os.path.normcase(os.path.abspath(p)) if p else ""


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for st in ("base", "head"):
        s = obs.get(st) or {}
        runs = s.get("runs") or {}
        ok = bool(runs) and all(r.get("rc") == 0 and (r.get("scenario") or {}).get("done") for r in runs.values())
        out.append((f"{st}: every run finished", ok,
                    "; ".join(f"{k} rc={r.get('rc')} err={str((r.get('scenario') or {}).get('error'))[:200]}" for k, r in runs.items())))
        own = all(_norm((r.get("scenario") or {}).get("unsloth_file")).startswith(_norm(s.get("checkout"))) for r in runs.values()) and bool(runs)
        out.append((f"{st}: unsloth imported from its own checkout", own,
                    f"{[(r.get('scenario') or {}).get('unsloth_file') for r in runs.values()]} vs {s.get('checkout')}"))
        t = [(r.get("scenario") or {}).get("torch") or {} for r in runs.values()]
        out.append((f"{st}: HIP gfx1151", bool(t) and all(x.get("hip") and "gfx1151" in str(x.get("arch")) for x in t), f"{t[:1]}"))
    hr = (obs.get("head") or {}).get("runs") or {}
    on = [r for k, r in hr.items() if k.endswith("flag1")]
    off = [r for k, r in hr.items() if k.endswith("flag0")]
    eng = bool(on) and all(((r.get("scenario") or {}).get("fused_calls") or {}).get("fused_true", 0) > 0 for r in on)
    dis = bool(off) and all(((r.get("scenario") or {}).get("fused_calls") or {}).get("fused_true", 1) == 0 for r in off)
    out.append(("head: fused order engaged with the flag on, never with it off", eng and dis,
                f"on {[(r.get('scenario') or {}).get('fused_calls') for r in on]} off {[(r.get('scenario') or {}).get('fused_calls') for r in off]}"))
    return out


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-6)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    bad, notes = [], []
    br, hr = base.get("runs") or {}, head.get("runs") or {}
    for key, b in br.items():
        model = key.split("|")[0]
        bs = b.get("scenario") or {}
        btr = bs.get("train") or {}
        for flag, tol in (("flag1", LOSS_TOL), ("flag0", KILL_TOL)):
            h = (hr.get(f"{model}|{flag}") or {}).get("scenario") or {}
            htr = h.get("train") or {}
            if h.get("error") or not htr:
                bad.append(f"{model} {flag}: {str(h.get('error'))[:300]}")
                continue
            if not htr.get("finite"):
                bad.append(f"{model} {flag}: non-finite loss / grad norm")
            bl, hl = btr.get("losses") or [], htr.get("losses") or []
            d = max((_rel(x, y) for x, y in zip(hl, bl)), default = None)
            notes.append(f"{model} {flag}: max rel loss diff vs base {d}, final {hl[-1:]} vs {bl[-1:]}")
            if d is None or d > tol:
                bad.append(f"{model} {flag}: loss differs from base by {d} (> {tol})")
            if htr.get("peak_mib", 0) > btr.get("peak_mib", 0) * 1.05:
                bad.append(f"{model} {flag}: peak {htr.get('peak_mib'):.0f} vs base {btr.get('peak_mib'):.0f} MiB")
            hm = statistics.median((htr.get("step_ms") or [0])[3:] or [0])
            bm = statistics.median((btr.get("step_ms") or [0])[3:] or [0])
            notes.append(f"{model} {flag}: median step {hm:.0f} vs base {bm:.0f} ms, peak {htr.get('peak_mib', 0):.0f} vs {btr.get('peak_mib', 0):.0f} MiB")
            if bm and hm > bm * 1.10:
                bad.append(f"{model} {flag}: median step {hm:.0f} ms vs base {bm:.0f} ms")
            bg, hg = bs.get("generate") or {}, h.get("generate") or {}
            bt, ht = bg.get("tokens") or [], hg.get("tokens") or []
            for i, (x, y) in enumerate(zip(bt, ht)):
                if x != y:
                    m = min((bg.get("margins") or [math.inf])[i] if i < len(bg.get("margins") or []) else math.inf,
                            (hg.get("margins") or [math.inf])[i] if i < len(hg.get("margins") or []) else math.inf)
                    notes.append(f"{model} {flag}: tokens diverge at {i} (margin {m})")
                    if m > 0.05:
                        bad.append(f"{model} {flag}: greedy tokens differ at step {i} with margin {m}")
                    break
    hp = head.get("head_only_pytest") or {}
    if hp and hp.get("rc") != 0:
        bad.append(f"tests/test_lora_fused_passes.py failed at the head: {str(hp.get('tail'))[-600:]}")
    notes.append(f"head-only pytest rc={hp.get('rc')}: {str(hp.get('tail'))[-200:]}")
    if bad:
        return True, "; ".join(bad) + " | " + "; ".join(notes)
    return False, "; ".join(notes)


def table(obs: dict) -> str:
    rows = ["| state | run | final loss | median step ms | peak MiB | fused calls | tokens[:8] |", "|---|---|---|---|---|---|---|"]
    for st in ("base", "head"):
        for k, r in ((obs.get(st) or {}).get("runs") or {}).items():
            s = r.get("scenario") or {}
            tr = s.get("train") or {}
            ms = tr.get("step_ms") or []
            rows.append(f"| {st} | {k} | {(tr.get('losses') or [None])[-1]} | "
                        f"{statistics.median(ms[3:]) if len(ms) > 3 else None} | {tr.get('peak_mib')} | "
                        f"{s.get('fused_calls')} | {((s.get('generate') or {}).get('tokens') or [])[:8]} |")
    return "\n".join(rows)
