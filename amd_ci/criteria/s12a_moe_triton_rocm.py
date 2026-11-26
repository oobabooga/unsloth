#!/usr/bin/env python3
"""Criteria (survey A12-2): on gfx1151 the zoo's sync-free Triton grouped GEMM replaces torch._grouped_mm's per-expert
loop for MoE LoRA training. Differential mode.

The "defect" is the base behaviour: every expert GEMM on the loop (GENERIC_CALLS gemm == 0 in the base reference run).
The head is "fixed" only when ALL hold, judged from the head state's interleaved matrix (base / head arms ABBA):
  * every head bf16 run engaged Triton (GENERIC_CALLS gemm > 0, fallback 0, no self-check disable) and every base
    run in the matrix did not (gemm == 0)
  * parity: per-step loss max rel vs the base reference <= max(5e-3, 3 x the base-vs-base value); step-0 LoRA grad
    rel L2 <= max(3e-2, 3 x the base-vs-base value) and cosine >= 0.999; step-1 cosine >= 0.99 (one AdamW step
    amplifies rounding differences in near-zero grads: Triton vs cuBLAS on B200 gave step-1 cosine 0.9986); bf16+fp16
  * speed: median of the head runs' median s/step < median of the base runs' x (1 - A/A spread - 0.05), A/A spread =
    (max - min) / median of the base runs' medians
"""

from __future__ import annotations

import statistics

TITLE = "A12-2: zoo Triton grouped GEMM vs torch._grouped_mm loop, MoE LoRA (Qwen3-30B-A3B layer shape), gfx1151"
MODE = "differential"
NEEDS = ["rocm", "gpu", "exclusive_gpu"]
LOSS_TOL, GRAD_TOL, COS_MIN, COS1_MIN, MARGIN = 5e-3, 3e-2, 0.999, 0.99, 0.05


def _runs(o):
    return (o or {}).get("runs") or {}


def _done(r):
    return bool(r) and r.get("stage") == "done" and r.get("ok") is True and bool(r.get("finite"))


def _gemm(r):
    return int(((r.get("counters") or {}).get("triton_gemm") or {}).get("gemm", 0) or 0)


def _fallback(r):
    return int(((r.get("counters") or {}).get("triton_gemm") or {}).get("fallback", 0) or 0)


def _loss_rel(r, ref):
    a, b = r.get("losses") or [], ref.get("losses") or []
    if not a or len(a) != len(b):
        return float("inf")
    return max(abs(x - y) / (abs(y) + 1e-12) for x, y in zip(a, b))


def _grad(r, key = "rel_l2", step = "step0"):
    p = r.get("parity") or {}
    vals = [v.get(key) for k, v in p.items() if k == step and isinstance(v, dict) and v.get(key) is not None]
    if not vals:
        return None
    return min(vals) if key == "cosine" else max(vals)


def _split(head_obs):
    rs = _runs(head_obs)
    base = [r for t, r in rs.items() if t.startswith("t") and r.get("arm") == "base"]
    head = [r for t, r in rs.items() if t.startswith("t") and r.get("arm") == "head"]
    return base, head, rs


def _stats(runs):
    meds = [r.get("median_step_s") for r in runs if _done(r) and r.get("median_step_s")]
    if not meds:
        return None, None, meds
    m = statistics.median(meds)
    return m, (max(meds) - min(meds)) / m, meds


def gates(obs):
    out = []
    b, h = obs.get("base") or {}, obs.get("head") or {}
    ref = _runs(b).get("base_ref_bf16") or {}
    out.append(("base reference bf16 run finished", _done(ref), ref.get("error") or str(ref.get("stage"))))
    for name, o in (("base", b), ("head", h)):
        for tag, r in _runs(o).items():
            f = str(r.get("unsloth_zoo_file") or "")
            ok = bool(f) and f.replace("\\", "/").startswith(str(r.get("checkout")).replace("\\", "/"))
            out.append((f"{name}/{tag} imported its arm's unsloth_zoo", ok, f or f"stage={r.get('stage')} rc={r.get('rc')} {str(r.get('error'))[:300]}"))
            out.append((f"{name}/{tag} ran on gfx1151", "gfx1151" in str(r.get("arch")), f"{r.get('device')} {r.get('arch')}"))
    bs, _, _ = _split(h)
    out.append(("head matrix has 3 finished base runs (A/A)", sum(_done(r) for r in bs) >= 3, f"{sum(_done(r) for r in bs)}"))
    return out


def base_shows_defect(base):
    ref = _runs(base).get("base_ref_bf16") or {}
    return _done(ref) and _gemm(ref) == 0


def head_is_fixed(head):
    base, hd, rs = _split(head)
    ok, why = True, []
    if not all(_done(r) for r in hd) or len(hd) < 3:
        return False
    if any(_gemm(r) == 0 or _fallback(r) or r.get("generic_disabled_reason") for r in hd):
        return False
    if any(_gemm(r) != 0 for r in base):
        return False
    aa_loss = max([_loss_rel(r, base[0]) for r in base[1:]] + [0.0]) if base else 0.0
    aa_grad = max([_grad(r) or 0.0 for r in base] + [0.0])
    for r in hd + [rs.get("head_fp16") or {}]:
        if not _done(r):
            return False
        g, c = _grad(r), _grad(r, "cosine")
        if g is None or g > max(GRAD_TOL, 3 * aa_grad) or c is None or c < COS_MIN:
            return False
        c1 = _grad(r, "cosine", "step1")
        if c1 is None or c1 < COS1_MIN:
            return False
    for r in hd:
        if _loss_rel(r, base[0]) > max(LOSS_TOL, 3 * aa_loss):
            return False
    bm, bs, _ = _stats(base)
    hm, _, _ = _stats(hd)
    if bm is None or hm is None:
        return False
    return hm < bm * (1 - bs - MARGIN)


def table(obs):
    rows = ["| state | run | arm | result | median s/step | step s (timed) | losses (first, last) | triton gemm/wgrad/fallback | grad rel L2 / cos (s0,s1) | peak GB |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    vers = None
    for name in ("base", "head"):
        for tag, r in _runs(obs.get(name)).items():
            vers = vers or r.get("versions")
            res = "ok" if _done(r) else f"FAIL: {str(r.get('error') or r.get('stage'))[:200]}"
            st = r.get("step_secs") or []
            ls = r.get("losses") or []
            tg = (r.get("counters") or {}).get("triton_gemm") or {}
            par = r.get("parity") or {}
            ps = ", ".join(f"{k}: {v.get('rel_l2', 0):.2e}/{v.get('cosine', 0):.6f}" for k, v in par.items())
            rows.append(f"| {name} | {tag} | {r.get('arm')}{'(T=0)' if r.get('env_triton') == '0' else ''} {r.get('dtype')} | {res} | "
                        f"{r.get('median_step_s') and round(r['median_step_s'], 4)} | {[round(x, 3) for x in st]} | "
                        f"{ls[:1] + ls[-1:]} | {tg.get('gemm')}/{tg.get('wgrad')}/{tg.get('fallback')} | {ps} | {r.get('peak_mem_gb')} |")
    h = obs.get("head") or {}
    base, hd, rs = _split(h)
    bm, bs, bmeds = _stats(base)
    hm, hs, hmeds = _stats(hd)
    rows.append("")
    if bm and hm:
        rows.append(f"base runs medians {[round(x, 4) for x in bmeds]} -> median {bm:.4f} s/step, A/A spread {bs:.2%}")
        rows.append(f"head runs medians {[round(x, 4) for x in hmeds]} -> median {hm:.4f} s/step, spread {hs:.2%}")
        rows.append(f"head / base = {hm / bm:.4f} (speedup {bm / hm:.3f}x); win threshold head < base x {1 - bs - MARGIN:.4f}")
    if base:
        rows.append(f"loss max rel head vs base ref: {[f'{_loss_rel(r, base[0]):.2e}' for r in hd]}; base vs base: {[f'{_loss_rel(r, base[0]):.2e}' for r in base[1:]]}")
    ref = _runs(obs.get("base")).get("base_ref_bf16") or {}
    for tag, r in (("base_ref_bf16", ref), ("t1_head", rs.get("t1_head") or {})):
        if r.get("profile_step"):
            rows.append(f"{tag} profiled step: {r['profile_step']}")
        rows.append(f"{tag} policy {r.get('triton_policy')} backend {r.get('moe_backend')} cap {r.get('capability')} disabled {r.get('generic_disabled_reason')}")
    if vers:
        rows += ["", "Versions: " + ", ".join(f"{k}={v}" for k, v in vers.items())]
    return "\n".join(rows)
