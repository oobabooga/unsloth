#!/usr/bin/env python3
"""Criteria: unsloth-zoo fused_linear (head) is not slower, not hungrier and not less accurate than zoo main (base).

Pairs, per workload (what a user gets by default on each side, plus the GRPO opt-in recompute for reference):
  ce        base old_zoo        vs head new_routed (and new direct, reported)
  dpo       base main1782       vs head new
  grpo      base main_old       vs head new
  grpo_ent  base main_recompute vs head new
Worse = at the largest token count: median step time > 1.10 x base (beyond both arms' A/A spread), or peak extra
allocated memory > 1.05 x base + 0.05 GiB; or at the smallest count the fp64 loss error > max(1.5 x base, 1e-6).
"""

from __future__ import annotations

TITLE = "fused_linear head vs zoo main: step time, memory, fp64 error per workload"
MODE = "regression"
NEEDS = ["gpu", "multi_gpu", "nvidia", "mig", "xpu", "mlx"]
# DPO is judged against the default path (dense: stock TRL logits); main1782 is an opt-in, reported in the table.
PAIRS = {"ce": ("old_zoo", "new_routed"), "dpo": ("dense", "new"), "grpo": ("main_old", "new"),
         "grpo_ent": ("main_recompute", "new")}


def _rows(state: dict) -> list:
    return [r for r in (state or {}).get("rows") or [] if r.get("head") == "frozen"]


def _get(state: dict, wl: str, arm: str, tokens: int):
    for r in _rows(state):
        if r["workload"] == wl and r["arm_label"] == arm and r["tokens"] == tokens:
            return r.get("stats") or {}
    return None


def _tokens(state: dict) -> list:
    return sorted({r["tokens"] for r in _rows(state)})


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = {n: v for n, v in obs.items() if not n.startswith("_")}
    errs = {n: v.get("error") for n, v in states.items() if v.get("error")}
    out = [("every state ran", not errs, "; ".join(f"{k}: {str(e)[-300:]}" for k, e in errs.items()))]
    missing = []
    for n, v in states.items():
        side = 0 if v.get("role") == "base" else 1
        for wl, pair in PAIRS.items():
            for t in _tokens(v):
                s = _get(v, wl, pair[side], t)
                if not s or not s.get("total_ms"):
                    missing.append(f"{n}/{wl}/{pair[side]}@{t}")
    out.append(("every paired arm measured", not missing, ", ".join(missing[:12])))
    gpu = [v.get("env", {}).get("gpu") for v in states.values()]
    out.append(("a GPU was used in every state", all(gpu), str(gpu)))
    return out


def table(obs: dict) -> str:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    lines = ["| workload | tokens | base arm | base ms | head arm | head ms | base GiB | head GiB | base loss err | head loss err | "
             "base grad err | head grad err |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for wl, (ba, ha) in PAIRS.items():
        arms = [ha] + (["new"] if wl == "ce" else [])
        for t in _tokens(base) or _tokens(head):
            b = _get(base, wl, ba, t) or {}
            for a in arms:
                h = _get(head, wl, a, t) or {}
                lines.append(f"| {wl} | {t} | {ba} | {b.get('total_ms')} | {a} | {h.get('total_ms')} | "
                             f"{b.get('peak_extra_alloc_gib')} | {h.get('peak_extra_alloc_gib')} | {b.get('loss_abs_err')} | "
                             f"{h.get('loss_abs_err')} | {b.get('grad_rel_err')} | {h.get('grad_rel_err')} |")
    return "\n".join(lines)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    toks = sorted(set(_tokens(base)) & set(_tokens(head)))
    if not toks:
        return True, "no common token count measured"
    big, small = toks[-1], toks[0]
    bad, notes = [], []
    for wl, (ba, ha) in PAIRS.items():
        b, h = _get(base, wl, ba, big), _get(head, wl, ha, big)
        if not b or not h or not b.get("total_ms") or not h.get("total_ms"):
            bad.append(f"{wl}: missing timing")
            continue
        noise = max(b.get("spread_pct") or 0, h.get("spread_pct") or 0) / 100
        ratio = h["total_ms"] / b["total_ms"]
        notes.append(f"{wl}@{big}: {b['total_ms']} -> {h['total_ms']} ms ({ratio:.2f}x), "
                     f"{b.get('peak_extra_alloc_gib')} -> {h.get('peak_extra_alloc_gib')} GiB")
        if ratio > 1.10 + noise:
            bad.append(f"{wl} slower {ratio:.2f}x")
        bm, hm = b.get("peak_extra_alloc_gib"), h.get("peak_extra_alloc_gib")
        if bm is not None and hm is not None and hm > 1.05 * bm + 0.05:
            bad.append(f"{wl} memory {bm} -> {hm} GiB")
        bs, hs = _get(base, wl, ba, small) or {}, _get(head, wl, ha, small) or {}
        if bs.get("loss_abs_err") is not None and hs.get("loss_abs_err") is not None:
            if hs["loss_abs_err"] > max(1.5 * bs["loss_abs_err"], 1e-6):
                bad.append(f"{wl} fp64 loss error {bs['loss_abs_err']:.3g} -> {hs['loss_abs_err']:.3g}")
    return bool(bad), "; ".join(bad or notes)
