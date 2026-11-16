#!/usr/bin/env python3
"""Criteria: SDPA at Gemma 4's global-attention shapes (judges probes/gemma4_sdpa_probe.py).

A measurement, not a fix: regression mode over the PR's base and head (the PR does not touch this path, so the two
must agree). The verdict line says only that; the answer is the table: an arm is WRONG when its forward output or any
of dq / dk / dv is non-finite or further than 0.05 relative L2 from fp64 math (bf16 math sits near 0.003-0.01), and
the real model forward is WRONG when sdpa vs eager top-1 agreement drops below 0.98 or the sdpa next-token loss
exceeds eager's by more than 0.1 nats.
"""

from __future__ import annotations

TITLE = "gfx1151: torch SDPA at Gemma 4 global-attention shapes (head 512, causal, GQA) and a real Gemma 4 forward"
MODE = "regression"
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm", "linux", "nvidia", "discrete_gpu"]
TOL = 0.05


def _wrong(arm: dict) -> bool:
    if "refused" in arm or "reference_error" in arm:
        return False
    vals = [arm.get(k) for k in ("out", "dq", "dk", "dv")]
    return any(v == "nonfinite" or (isinstance(v, (int, float)) and v > TOL) for v in vals)


def _wrong_set(state: dict) -> set:
    return {k for k, a in ((state or {}).get("arms") or {}).items() if _wrong(a)}


def _model_wrong(mf: dict) -> bool:
    if not mf or "top1_agree" not in mf:
        return False
    return (not mf.get("finite") or mf["top1_agree"] < 0.98
            or (mf.get("nll_sdpa", 0) - mf.get("nll_eager", 0)) > 0.1)


def gates(obs: dict) -> list:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        arms = o.get("arms") or {}
        measured = [a for a in arms.values() if "out" in a]
        if o.get("kernels_skipped"):
            continue
        out.append((f"{name}: arms measured", len(measured) >= 24,
                    f"{len(measured)} measured, {sum('refused' in a for a in arms.values())} refused; "
                    f"{o.get('torch')} {o.get('arch')} {o.get('kernels_error', '')}"))
    mf = (obs.get("head") or {}).get("model_forward") or {}
    out.append(("head: real Gemma 4 forward compared", "top1_agree" in mf, str(mf.get("error") or mf)[:300]))
    return out


def head_is_worse(base: dict, head: dict):
    new = sorted(_wrong_set(head) - _wrong_set(base))
    if new:
        return True, f"arms wrong only at head: {new[:6]}"
    return False, (f"head no worse; WRONG arms at both: {len(_wrong_set(head))}; "
                   f"real forward {'WRONG' if _model_wrong((head or {}).get('model_forward') or {}) else 'ok'}")


def table(obs: dict) -> str:
    head = obs.get("head") or {}
    arms = head.get("arms") or {}
    rows = ["| arm (head state) | out | dq | dk | dv | verdict |", "|---|---|---|---|---|---|"]
    for k, a in arms.items():
        if "refused" in a or "reference_error" in a:
            rows.append(f"| {k} | {a.get('refused') or a.get('reference_error')} | | | | refused |")
        else:
            rows.append(f"| {k} | {a.get('out')} | {a.get('dq')} | {a.get('dk')} | {a.get('dv')} | "
                        f"{'WRONG' if _wrong(a) else 'ok'} |")
    mf = head.get("model_forward") or {}
    rows += ["", f"**Real forward** ({mf.get('model')}, {mf.get('tokens')} tokens, transformers {mf.get('transformers')}, "
                 f"{mf.get('class')}): sdpa vs eager rel_l2 {mf.get('rel_l2')}, top-1 agree {mf.get('top1_agree')}, "
                 f"KL mean / max {mf.get('kl_eager_sdpa_mean')} / {mf.get('kl_eager_sdpa_max')}, NLL eager "
                 f"{mf.get('nll_eager')} vs sdpa {mf.get('nll_sdpa')}; impl {mf.get('impl_eager')} / {mf.get('impl_sdpa')} "
                 f"-> {'WRONG' if _model_wrong(mf) else 'ok'} {str(mf.get('error') or '')[:200]}"]
    return "\n".join(rows)
