#!/usr/bin/env python3
"""Criteria for probes/dit_fp16_route_probe.py (PR 18355): on a real ROCm card the fp16 fallback never engages and
every DiT precision answer is the base's, in every env case. Gated on a ROCm torch with a visible GPU, the checkout's
own module, and head actually carrying the new predicate, so a blind run cannot pass."""
from __future__ import annotations

TITLE = "DiT precision routing on the real gfx1151 card, base versus head"
MODE = "regression"
NEEDS = ["gpu", "nvidia"]
KEYS = ("bf16_unsupported_reason", "effective_mixed_precision", "preflight")


def _states(obs):
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def gates(obs):
    out = []
    for name, o in _states(obs).items():
        cases = o.get("cases") or {}
        errs = [c for c, v in cases.items() if "error" in v]
        out.append((f"{name}: every env case ran", bool(cases) and not errs, ", ".join(errs) or f"{len(cases)} ran"))
        u = cases.get("unset") or {}
        out.append((f"{name}: torch is a ROCm build", bool(u.get("hip")), str(u.get("hip"))))
        out.append((f"{name}: a GPU is visible", bool(u.get("cuda_available")), str(u.get("arch"))))
        out.append((f"{name}: checkout's own module", f"/{name}" in str(u.get("file")) or name in str(u.get("file")), str(u.get("file"))))
        if name == "head":
            out.append(("head: carries dit_fp16_fallback", bool(u.get("has_fallback_fn")), str(u.get("has_fallback_fn"))))
    return out


def table(obs):
    rows = ["| state | env | family | fallback | bf16 refusal | mixed precision | nf4 preflight | bf16 preflight |",
            "|---|---|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        for c, v in (o.get("cases") or {}).items():
            for f, r in (v.get("fam") or {}).items():
                rows.append(f"| {name} | {c} | {f} | {r['fallback']} | {bool(r['bf16_unsupported_reason'])} | "
                            f"{r['effective_mixed_precision']} | {bool(r['preflight']['nf4'])} | {bool(r['preflight']['bf16'])} |")
    return "\n".join(rows)


def head_is_worse(base, head):
    bad = []
    for c, hv in (head.get("cases") or {}).items():
        bv = (base.get("cases") or {}).get(c) or {}
        for f, hr in (hv.get("fam") or {}).items():
            br = (bv.get("fam") or {}).get(f) or {}
            if hr.get("fallback"):
                bad.append(f"{c}/{f}: fp16 fallback engaged on ROCm")
            for k in KEYS:
                if hr.get(k) != br.get(k):
                    bad.append(f"{c}/{f}: {k} {br.get(k)!r} -> {hr.get(k)!r}")
        if hv.get("train_precision_modes") != bv.get("train_precision_modes"):
            bad.append(f"{c}: train_precision_modes moved")
    if bad:
        return True, "; ".join(bad[:20])
    return False, "fallback never engages on ROCm; every DiT precision answer equals the base in every env case"
