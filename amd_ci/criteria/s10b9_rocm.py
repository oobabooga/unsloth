#!/usr/bin/env python3
"""Criteria: saved attention AUTOMATIC mode (UNSLOTH_GC_SAVE_ATTENTION unset) on gfx1151, unsloth #19508 + zoo #2778.

Differential mode.
  Defect at base  : every base repeat (8192 tokens) trained with finite losses and saved attention kept and reused
                    NOTHING (stored == reused == 0): automatic mode is off on ROCm.
  Fixed at head   : every head repeat trained with finite losses, stored > 0 and reused > 0 in the timed steps,
                    saved_attention_active() was True inside checkpoints, and every head repeat's per-step losses are
                    bitwise equal to the base repeats' (hex floats). The short-sequence head repeat is reported only.
  merge           : engagement (stored, reused > 0) and finite losses.
Timing (median s/step, spread, A/A) is rendered and stated, not gated: the verdict is about engagement + equality.
Gates: env built; each repeat ran on gfx1151 and imported unsloth / unsloth_zoo from that arm's trees; base A/A ran.
"""
from __future__ import annotations

import statistics

TITLE = "Saved attention automatic mode on gfx1151: Qwen3-0.6B LoRA 8192 tokens, base vs head (unsloth #19508 + zoo #2778)"
MODE = "differential"
NEEDS = ["rocm", "gpu", "exclusive_gpu", "nvidia", "multi_gpu", "windows"]
KEYS = ("stored", "reused", "mismatch", "unmatched", "no_kernel", "too_big", "random", "declined_short",
        "declined_unmeasured", "declined_budget", "kept_bytes")


def _reps(o, prefix):
    return {k: v for k, v in (o.get("reps") or {}).items() if k.startswith(prefix) and k != "head_short"}


def _ok(r):
    return r.get("stage") == "done" and r.get("finite") and not r.get("error")


def _tree_ok(r):
    def n(x):
        return str(x or "").replace("\\", "/").lower()
    p = r.get("paths") or ["?", "?"]
    uf, zf = n(r.get("unsloth_file")), n(r.get("zoo_file"))
    return uf.startswith(n(p[0]).rstrip("/") + "/") and zf.startswith(n(p[1]).rstrip("/") + "/")


def _delta(r, key):
    f, w = r.get("stats_final") or {}, r.get("stats_after_warmup") or {}
    return int(f.get(key, 0) or 0) - int(w.get(key, 0) or 0)


def _engaged(r):
    f = r.get("stats_final") or {}
    return int(f.get("stored", 0) or 0) > 0 and int(f.get("reused", 0) or 0) > 0 and _delta(r, "reused") > 0


def _idle(r):
    f = r.get("stats_final") or {}
    return int(f.get("stored", 0) or 0) == 0 and int(f.get("reused", 0) or 0) == 0


def _med(r):
    t = r.get("timed_s") or []
    return statistics.median(t) if t else None


def _arm(reps):
    meds = [m for m in (_med(r) for r in reps.values()) if m is not None]
    if not meds:
        return None
    m = statistics.median(meds)
    return {"median": m, "min": min(meds), "max": max(meds), "spread_pct": 100 * (max(meds) - min(meds)) / m,
            "reps": meds}


def _base_hex(obs):
    b = _reps(obs.get("base") or {}, "base")
    return next((r.get("loss_hex") for r in b.values() if r.get("loss_hex")), None)


def gates(obs):
    out = []
    for st in ("base", "head"):
        o = obs.get(st) or {}
        env = (o.get("env") or {}).get("env") or {}
        out.append((f"{st} environment built", bool(env.get("cuda_available")) and not (o.get("env") or {}).get("error"),
                    {k: env.get(k) for k in ("torch", "hip", "arch", "transformers", "trl", "peft")}))
        reps = o.get("reps") or {}
        out.append((f"{st} repeats imported this arm's unsloth + unsloth_zoo",
                    bool(reps) and all(_tree_ok(r) for r in reps.values()),
                    {k: [r.get("unsloth_file"), r.get("zoo_file")] for k, r in reps.items() if not _tree_ok(r)} or "all"))
        out.append((f"{st} ran on gfx1151 (HIP)", bool(reps) and all(str(r.get("arch") or "").startswith("gfx1151")
                                                                 for r in reps.values()),
                    sorted({str(r.get("arch")) for r in reps.values()})))
    base = _reps(obs.get("base") or {}, "base")
    aa = _reps(obs.get("head") or {}, "aa")
    out.append(("base repeats + interleaved A/A repeats trained (finite)", len(base) >= 2 and len(aa) >= 2 and
                all(_ok(r) for r in [*base.values(), *aa.values()]),
                {k: r.get("error") or r.get("stage") for k, r in [*base.items(), *aa.items()] if not _ok(r)} or
                f"base={len(base)} aa={len(aa)}"))
    return out


def base_shows_defect(base):
    reps = _reps(base, "base")
    if not reps or not all(_ok(r) for r in reps.values()):
        return False, "base repeats did not all train"
    if all(_idle(r) for r in reps.values()):
        return True, "automatic mode kept nothing at base"
    return False, "base already keeps attention in automatic mode"


def head_is_fixed(head):
    reps = _reps(head, "head")
    if not reps or not all(_ok(r) for r in reps.values()):
        return False, "head repeats did not all train: " + str({k: r.get("error") for k, r in reps.items()})
    bad = [k for k, r in reps.items() if not _engaged(r)]
    if bad:
        return False, "head kept / reused nothing in " + ", ".join(bad) + ": " + str(
            {k: reps[k].get("stats_final") for k in bad})
    inactive = [k for k, r in reps.items() if not ((r.get("active_during_warmup") or {}).get("active", 0) > 0)]
    if inactive:
        return False, "saved_attention_active() never True inside a checkpoint in " + ", ".join(inactive)
    ref = head.get("_base_hex")
    if ref is None:
        aa = _reps(head, "aa")
        ref = next((r.get("loss_hex") for r in aa.values() if r.get("loss_hex")), None)
    diff = [k for k, r in reps.items() if r.get("loss_hex") != ref]
    if diff:
        return False, "per-step losses differ from base in " + ", ".join(diff)
    return True, "engaged, losses bitwise equal to base"


def state_is_fixed(name, o):
    reps = o.get("reps") or {}
    if not reps:
        return None
    r = next(iter(reps.values()))
    if not _ok(r):
        return False, f"{name} did not train: {r.get('error')}"
    if not _engaged(r):
        return False, f"{name} kept nothing: {r.get('stats_final')}"
    return True, f"{name} engaged: {r.get('stats_final')}"


def _stats_cell(r):
    f = r.get("stats_final") or {}
    s = " ".join(f"{k}={f[k]}" for k in KEYS if f.get(k))
    act = r.get("active_during_warmup") or {}
    return (s or "all zero") + f"; active {act.get('active', 0)}/{act.get('active', 0) + act.get('inactive', 0)}"


def table(obs):
    base, head = obs.get("base") or {}, obs.get("head") or {}
    ref = _base_hex(obs)
    if ref is not None:
        head["_base_hex"] = ref
    arms = {"base (block 1)": _reps(base, "base"), "head": _reps(head, "head"), "A/A base (interleaved with head)":
            _reps(head, "aa")}
    lines = ["| arm | n | median s/step | min..max of rep medians | spread % | peak alloc GB | peak reserved GB |",
             "|---|---|---|---|---|---|---|"]
    summ = {}
    for name, reps in arms.items():
        s = _arm(reps)
        summ[name] = s
        pk = [r.get("peak_alloc_gb") for r in reps.values() if r.get("peak_alloc_gb") is not None]
        pr = [r.get("peak_reserved_gb") for r in reps.values() if r.get("peak_reserved_gb") is not None]
        if s:
            lines.append(f"| {name} | {len(s['reps'])} | {s['median']:.4f} | {s['min']:.4f}..{s['max']:.4f} | "
                         f"{s['spread_pct']:.2f} | {min(pk):.3f}..{max(pk):.3f} | {min(pr):.3f}..{max(pr):.3f} |"
                         if pk else f"| {name} | {len(s['reps'])} | {s['median']:.4f} | - | - | - | - |")
        else:
            lines.append(f"| {name} | 0 | - | - | - | - | - |")
    lines.append("")
    b1, h, aa = summ["base (block 1)"], summ["head"], summ["A/A base (interleaved with head)"]
    if h and aa:
        d = 100 * (h["median"] - aa["median"]) / aa["median"]
        aad = 100 * (aa["median"] - b1["median"]) / b1["median"] if b1 else float("nan")
        noise = max(aa["spread_pct"], b1["spread_pct"] if b1 else 0, abs(aad))
        lines.append(f"Timing: head vs interleaved base {d:+.2f}% s/step; head vs base block 1 "
                     f"{(100 * (h['median'] - b1['median']) / b1['median']) if b1 else float('nan'):+.2f}%; "
                     f"A/A (interleaved base vs base block 1) {aad:+.2f}%; noise floor (max of A/A delta and base "
                     f"rep spreads) {noise:.2f}%; beyond noise: {'yes' if abs(d) > noise else 'NO'}")
        lines.append("")
    lines += ["| repeat | rc | engagement (stats_final; saved_attention_active() in warmup self_attn calls) | "
              "timed reused delta | median s/step | losses == base |", "|---|---|---|---|---|---|"]
    for st, o in (("base", base), ("head", head), ("merge", obs.get("merge") or {})):
        for k, r in (o.get("reps") or {}).items():
            m = _med(r)
            eq = "ref" if (st == "base" and k == "base0") else (
                "n/a (seq differs)" if k == "head_short" else str(r.get("loss_hex") == ref))
            lines.append(f"| {st}:{k} (seq {r.get('seq')}) | {r.get('rc')} | "
                         f"{r.get('error') or _stats_cell(r)} | {_delta(r, 'reused')} | "
                         f"{'-' if m is None else f'{m:.4f}'} | {eq} |")
    lines.append("")
    hr = next(iter(_reps(head, "head").values()), {}) if _reps(head, "head") else {}
    br = next(iter(_reps(base, "base").values()), {}) if _reps(base, "base") else {}
    for nm, r in (("base", br), ("head", hr)):
        lines.append(f"{nm} dispatch: {r.get('dispatch')} auto_active={r.get('auto_active_flag')}->"
                     f"{r.get('auto_active_flag_end')} rocm_table={r.get('auto_min_seq_rocm_table')} "
                     f"rocm_sdpa_is_kept={r.get('rocm_sdpa_is_kept')} auto_min_seq(cuda:0)={r.get('auto_min_seq_device')}")
    if ref:
        lines.append(f"base0 losses: {br.get('losses')}")
    return "\n".join(lines)
