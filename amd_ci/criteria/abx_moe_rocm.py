#!/usr/bin/env python3
"""Criteria (gate 18 A/B): zoo #2849 x #2850 on gfx1151, six arms on zoo main in one job (see abx_probe.py).
Differential mode used as a run-completeness check only: "defect" = the base reference ran; "fixed" = every head
matrix run finished with finite losses on its own tree. Speed and parity are reported in the table, not judged here.
"""
from __future__ import annotations

import statistics

TITLE = "Gate 18 A/B: zoo #2849 x #2850, MoE LoRA (Qwen3-30B-A3B layer shape), gfx1151, six arms"
MODE = "differential"
NEEDS = ["rocm", "gpu", "exclusive_gpu"]
ARMS = "abcdef"


def _runs(o):
    return (o or {}).get("runs") or {}


def _done(r):
    return bool(r) and r.get("stage") == "done" and r.get("ok") is True and bool(r.get("finite"))


def gates(obs):
    out = []
    b, h = obs.get("base") or {}, obs.get("head") or {}
    ref = _runs(b).get("ref_a") or {}
    out.append(("base reference run finished", _done(ref), ref.get("error") or str(ref.get("stage"))))
    for name, o in (("base", b), ("head", h)):
        for tag, r in _runs(o).items():
            f = str(r.get("unsloth_zoo_file") or "").replace("\\", "/")
            ok = bool(f) and f.startswith(str(r.get("checkout")).replace("\\", "/"))
            out.append((f"{name}/{tag} imported its arm's unsloth_zoo", ok, f or f"stage={r.get('stage')} {str(r.get('error'))[:300]}"))
            out.append((f"{name}/{tag} ran on gfx1151", "gfx1151" in str(r.get("arch")), f"{r.get('device')} {r.get('arch')}"))
    out.append(("arm trees patched", not h.get("error"), str(h.get("trees"))[:300]))
    return out


def base_shows_defect(base):
    return _done(_runs(base).get("ref_a") or {})


def head_is_fixed(head):
    rs = _runs(head)
    return bool(rs) and all(_done(r) for r in rs.values())


def _loss_rel(r, ref):
    x, y = r.get("losses") or [], ref.get("losses") or []
    if not x or len(x) != len(y):
        return None
    return max(abs(p - q) / (abs(q) + 1e-12) for p, q in zip(x, y))


def table(obs):
    h = obs.get("head") or {}
    ref = _runs(obs.get("base")).get("ref_a") or {}
    rows = ["| run | arm | ok | median s/step | backend (env) | probe failure | triton gemm/wgrad/fallback | loss max rel vs ref_a | grad s0 relL2/cos | s1 cos |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, r in list(_runs(obs.get("base")).items()) + list(_runs(h).items()):
        tg = (r.get("counters") or {}).get("triton_gemm") or {}
        p = r.get("parity") or {}
        s0, s1 = p.get("step0") or {}, p.get("step1") or {}
        lr = _loss_rel(r, ref)
        rows.append(f"| {tag} | {r.get('arm')} | {'ok' if _done(r) else 'FAIL ' + str(r.get('error'))[:150]} | "
                    f"{r.get('median_step_s') and round(r['median_step_s'], 4)} | {r.get('moe_backend')} ({r.get('env_backend')}, T={r.get('env_triton')}) | "
                    f"{str(r.get('probe_failure'))[:60]} | {tg.get('gemm')}/{tg.get('wgrad')}/{tg.get('fallback')} | "
                    f"{lr if lr is None else f'{lr:.2e}'} | {s0.get('rel_l2', 0):.2e}/{s0.get('cosine', 0):.6f} | {s1.get('cosine', 0):.6f} |")
    rows.append("")
    rs = _runs(h)
    for arm in ARMS:
        meds = [r["median_step_s"] for t, r in rs.items() if t.startswith("r") and r.get("arm") == arm and _done(r) and r.get("median_step_s")]
        if arm == "a":
            meds += [r["median_step_s"] for t, r in rs.items() if t.startswith("aa") and _done(r) and r.get("median_step_s")]
        if meds:
            m = statistics.median(meds)
            rows.append(f"arm {arm}: medians {[round(x, 4) for x in meds]} -> {m:.4f} s/step, spread {(max(meds) - min(meds)) / m:.2%}")
    for tag, r in rs.items():
        if r.get("profile_step"):
            rows.append(f"{tag} profiled step: {r['profile_step']}")
    return "\n".join(rows)
