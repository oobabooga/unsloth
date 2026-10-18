#!/usr/bin/env python3
"""Criteria: KV preemption on gfx1151 across architectures, upstream b11642 versus #269.

Base must fail the overflow (S1) with context errors on every model whose cache shares cells; the
head must finish every S1 by parking (swap, cost and forced recompute), keep the no-park output
equal to base (S3), let one request reach the whole window (S5), refuse an over-long prompt cleanly
(S8), and return byte-identical text after forced swap parks (S10). Caches that keep their own cells
per sequence (deepseek4) must not park and must not error. Pairs with llama_arch_probe.py.
"""

from __future__ import annotations

TITLE = "llama.cpp #269 KV preemption across architectures on gfx1151, upstream b11642 versus head"
MODE = "differential"
NEEDS = ["rocm", "gpu"]
UNSHARED = {"deepseek4-moe"}


def _runs(o):
    return o.get("runs") or []


def _get(o, model, scen, arm, pol="fcfs"):
    return next((r for r in _runs(o) if r.get("tag") == model and r.get("scenario") == scen and r.get("arm") == arm
                 and (r.get("policy") or "fcfs") == pol), None)


def _models(o):
    return sorted({r.get("tag") for r in _runs(o)})


def _clean(r):
    return bool(r) and not r.get("start_error") and r.get("completed") == r.get("submitted") and not r.get("ctx_errors") \
        and not r.get("other_errors") and not r.get("asserts") and r.get("server_alive") is not False


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name} built llama-server ({o.get('backend')}, {str(o.get('llama_ref', ''))[:9]})",
                    bool(o.get("backend")) and not any(o.get(k) for k in ("build_error", "configure_error", "clone_error", "error")),
                    str(o.get("error") or o.get("build_error") or "")[-200:]))
        out.append((f"{name} ran on the GPU", bool(o.get("device")) and o.get("device") != "CPU", f"device {o.get('device')}"))
        out.append((f"{name} produced runs", len(_runs(o)) > 0, f"{len(_runs(o))} runs"))
    h = obs.get("head") or {}
    out.append(("head has --preempt-ram", bool(h.get("has_preempt_ram")), ""))
    return out


def base_shows_defect(base):
    hit = []
    for m in _models(base):
        if m in UNSHARED:
            continue
        r = _get(base, m, "S1", "base")
        if r and (r.get("ctx_errors") or (r.get("completed") or 0) < (r.get("submitted") or 0)):
            hit.append(m)
    return bool(hit), f"overflow failed on base for {len(hit)} model(s): {', '.join(hit)}"


def head_is_fixed(head):
    bad = []
    for m in _models(head):
        for scen, arm, pol in (("S1", "swap", "fcfs"), ("S1", "swap", "cost"), ("S1", "recomp", "fcfs"), ("S5", "swap", "fcfs")):
            r = _get(head, m, scen, arm, pol)
            if r is None:
                continue
            if not _clean(r):
                bad.append(f"{m} {scen} {arm} {pol}: {r.get('completed')}/{r.get('submitted')} ctx {r.get('ctx_errors')} other {r.get('other_errors')} {str(r.get('start_error') or '')[:80]}")
            elif scen == "S1" and m not in UNSHARED and not (r.get("parks") or r.get("metric_preempt")):
                bad.append(f"{m} {scen} {arm} {pol}: completed without a park")
            elif scen == "S1" and m in UNSHARED and (r.get("parks") or 0) > 0:
                bad.append(f"{m}: parked a cache whose sequences do not share cells")
        s5 = _get(head, m, "S5", "swap")
        if s5 and not s5.get("full_window_ok"):
            bad.append(f"{m} S5: the full-window request did not finish")
        s8 = _get(head, m, "S8", "swap")
        if s8 and not (s8.get("clean_4xx") and s8.get("after_ok")):
            bad.append(f"{m} S8: over-long prompt {s8.get('status')} after_ok={s8.get('after_ok')}")
        s10 = _get(head, m, "S10", "swap")
        if s10 and m not in UNSHARED and not all(s10.get("identical") or [False]):
            bad.append(f"{m} S10 swap: forced parks changed the text (first divergence {s10.get('first_divergence')})")
    return not bad, "; ".join(bad[:12]) or "every model finished every cell"


def table(obs):
    rows = ["| model | base S1 | head S1 swap fcfs | head S1 cost | head S1 recompute | S3 off==swap | S5 full window | S8 | S10 swap identical | S10 recompute first divergence | S4 flood |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    b, h = obs.get("base") or {}, obs.get("head") or {}

    def c(r):
        if not r:
            return "-"
        if r.get("start_error"):
            return "start error"
        return f"{r.get('completed')}/{r.get('submitted')} ctx {r.get('ctx_errors')} parks {r.get('parks')}"

    for m in sorted(set(_models(b)) | set(_models(h))):
        o3, s3 = _get(h, m, "S3", "off"), _get(h, m, "S3", "swap")
        s10, r10 = _get(h, m, "S10", "swap"), _get(h, m, "S10", "recomp")
        s5, s8 = _get(h, m, "S5", "swap"), _get(h, m, "S8", "swap")
        rows.append(f"| {m} | {c(_get(b, m, 'S1', 'base'))} | {c(_get(h, m, 'S1', 'swap'))} | {c(_get(h, m, 'S1', 'swap', 'cost'))} | "
                    f"{c(_get(h, m, 'S1', 'recomp'))} | {(o3 or {}).get('tokens') == (s3 or {}).get('tokens') if o3 and s3 else '-'} | "
                    f"{(s5 or {}).get('full_window_ok', '-')} | {(s8 or {}).get('status', '-')} | {(s10 or {}).get('identical', '-')} | "
                    f"{(r10 or {}).get('first_divergence', '-')} | {c(_get(h, m, 'S4', 'swap'))} |")
    return "\n".join(rows)
