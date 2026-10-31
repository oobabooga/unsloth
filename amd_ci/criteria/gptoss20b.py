#!/usr/bin/env python3
"""Criteria: real GPT-OSS 20b on gfx1151, zoo base vs head (merged flex-inference series). Pairs with
probes/gptoss20b_probe.py (all arms in the head invocation, base arms interleaved A B B A).

Non-vacuity gates (INCONCLUSIVE unless all hold): model downloaded; head tree == the expected zoo main tree; every arm
exited 0 with no error / OOM rows; each arm imported unsloth_zoo from its own state checkout; the GPU is an AMD
gfx1151 (Radeon 8060S); head generate called flex_attention_with_sink_inference, the eager kill switch and the base
did not.
head_is_worse (REGRESSION) if any of:
  - greedy tokens: head flex differs from head eager or from base at a step whose top-2 logit margin (on the arm that
    diverged from the reference) is >= NEAR_TIE, i.e. not a near-tie flip
  - per case: head median decode ms/token or prefill s above base median by more than max(5%, base A/A spread)
  - per case: head peak allocated memory above base by more than 5% + 0.25 GB
  - SFT: any non-finite loss, adapters did not move, head step time (median of steps 2..) above base by > 10%,
    or step-1 loss differs from base by > 2%
"""

from __future__ import annotations

import math
import os
import statistics

TITLE = "Real GPT-OSS 20b on gfx1151: zoo flex-inference series (#2153, #2161, #2163) base vs head"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "windows", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]
NEAR_TIE = 0.25
FLEX = "flex_attention_with_sink_inference"


def _head(obs):
    return obs.get("head") or {}


def _arms(obs):
    return _head(obs).get("arms") or {}


def _norm(p):
    return os.path.normcase(os.path.abspath(p)) if p else ""


def _rows(arm):
    return {(r["ctx"], r["batch"]): r for r in ((arm.get("result") or {}).get("rows") or [])}


def _flex_calls(calls):
    return (calls or {}).get(FLEX, 0)


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    h, arms = _head(obs), _arms(obs)
    out = [("model downloaded", (h.get("download") or {}).get("rc") == 0,
            f"{h.get('model')} 4bit={h.get('load_in_4bit')} {h.get('download')}"),
           ("head tree == expected zoo main tree", bool(h.get("head_tree")) and h.get("head_tree") == h.get("expected_tree"),
            f"head {h.get('head_tree')} expected {h.get('expected_tree')} base sha {h.get('base_sha')}"),
           ("all 7 arms present", len(arms) == 7, ", ".join(arms))]
    cos = {"head": h.get("checkout") or "", "base": os.path.join(os.path.dirname(h.get("checkout") or ""), "base")}
    for name, a in arms.items():
        r = a.get("result") or {}
        bad = [f"{k}: {v.get('error') or v.get('oom')}" for k, v in _rows(a).items() if v.get("error") or v.get("oom")]
        ok = a.get("rc") == 0 and not r.get("error") and not bad and (bool(r) or name.startswith("gen_"))
        out.append((f"{name}: exited 0, no error / OOM", ok,
                    f"rc={a.get('rc')} {r.get('error') or ''} {'; '.join(bad)} {str(a.get('tail'))[-300:] if not ok else ''}"))
        if r.get("zoo_file"):
            out.append((f"{name}: zoo from its own checkout", _norm(r["zoo_file"]).startswith(_norm(cos[a["state"]])),
                        f"{r['zoo_file']} vs {cos[a['state']]}"))
        if r.get("gpu"):
            out.append((f"{name}: AMD gfx1151 GPU", "8060S" in r["gpu"] or "gfx1151" in r["gpu"] or "Radeon" in r["gpu"],
                        r["gpu"]))
    g = {n: (arms.get(n) or {}).get("gen") or {} for n in ("bench_base_1", "bench_head_1", "gen_head_eager")}
    out.append(("head generate engaged flex inference", _flex_calls(g["bench_head_1"].get("calls")) > 0,
                str(g["bench_head_1"].get("calls"))))
    out.append(("eager kill switch: no flex inference calls", bool(g["gen_head_eager"]) and
                _flex_calls(g["gen_head_eager"].get("calls")) == 0, str(g["gen_head_eager"].get("calls"))))
    out.append(("base: no flex inference calls", bool(g["bench_base_1"]) and _flex_calls(g["bench_base_1"].get("calls")) == 0,
                str(g["bench_base_1"].get("calls"))))
    return out


def _tokens_cmp(ref, other):
    """First divergence per row: (row, step, margin_ref, margin_other) or None."""
    res = []
    for i, (a, b) in enumerate(zip(ref.get("tokens") or [], other.get("tokens") or [])):
        for s, (x, y) in enumerate(zip(a, b)):
            if x != y:
                mr = (ref.get("margins") or [[]])[s][i] if s < len(ref.get("margins") or []) else None
                mo = (other.get("margins") or [[]])[s][i] if s < len(other.get("margins") or []) else None
                res.append((i, s, mr, mo))
                break
    return res


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _perf(obs):
    arms = _arms(obs)
    out = {}
    for case in sorted(set(_rows(arms.get("bench_base_1") or {})) | set(_rows(arms.get("bench_head_1") or {}))):
        d = {}
        for st in ("base", "head"):
            rs = [_rows(arms.get(f"bench_{st}_{i}") or {}).get(case) or {} for i in (1, 2)]
            d[st] = {k: [r.get(k) for r in rs] for k in ("decode_ms_per_token", "prefill_s", "peak_alloc_gb")}
        out[case] = d
    return out


def _sft(obs):
    return {st: ((_arms(obs).get(f"sft_{st}") or {}).get("result") or {}) for st in ("base", "head")}


def table(obs: dict) -> str:
    h = _head(obs)
    lines = [f"Model `{h.get('model')}` load_in_4bit={h.get('load_in_4bit')}; base `{h.get('base_sha')}`, head tree "
             f"`{h.get('head_tree')}`", "",
             "| case | metric | base run 1 / 2 | head run 1 / 2 | head / base (median) |", "|---|---|---|---|---|"]
    for case, d in _perf(obs).items():
        for k in ("prefill_s", "decode_ms_per_token", "peak_alloc_gb"):
            b, hh = _med(d["base"][k]), _med(d["head"][k])
            ratio = f"{hh / b:.3f}" if b and hh else "-"
            lines.append(f"| {case[0]}x{case[1]} | {k} | {d['base'][k]} | {d['head'][k]} | {ratio} |")
    arms = _arms(obs)
    g = {n: (arms.get(n) or {}).get("gen") or {} for n in ("bench_base_1", "bench_head_1", "gen_head_eager")}
    lines += ["", f"Greedy tokens ({len((g['bench_head_1'].get('tokens') or [[]])[0])} per row, padded batch of 2, "
              f"pad lens {g['bench_head_1'].get('pad_lens')}):",
              f"- head flex vs head eager first divergence (row, step, margin eager, margin flex): "
              f"{_tokens_cmp(g['gen_head_eager'], g['bench_head_1']) or 'identical'}",
              f"- head flex vs base: {_tokens_cmp(g['bench_base_1'], g['bench_head_1']) or 'identical'}",
              f"- head eager vs base: {_tokens_cmp(g['bench_base_1'], g['gen_head_eager']) or 'identical'}"]
    s = _sft(obs)
    lines += ["", "| SFT | losses | step s | peak GB | lora_B after (min) |", "|---|---|---|---|---|"]
    for st, r in s.items():
        lines.append(f"| {st} | {[round(x, 4) for x in r.get('losses') or []]} | {r.get('step_s')} | "
                     f"{r.get('peak_alloc_gb')} | {r.get('lora_b_after_min')} |")
    lines.append("")
    lines.append("Arms: " + ", ".join(f"{n} rc={a.get('rc')} {a.get('s')}s dyn_fallback={a.get('dynamic_cache_fallback')} alias_warn={a.get('alias_warnings')}"
                                      for n, a in arms.items()))
    return "\n".join(lines)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    obs = {"head": head, "base": base}
    why = []
    arms = _arms(obs)
    g = {n: (arms.get(n) or {}).get("gen") or {} for n in ("bench_base_1", "bench_head_1", "gen_head_eager")}
    for ref in ("gen_head_eager", "bench_base_1"):
        for row, step, mr, mo in _tokens_cmp(g[ref], g["bench_head_1"]):
            m = min(x for x in (mr, mo) if x is not None) if (mr is not None or mo is not None) else None
            if m is None or m >= NEAR_TIE:
                why.append(f"tokens vs {ref}: row {row} diverges at step {step}, margins ref {mr} head {mo}")
    for case, d in _perf(obs).items():
        for k in ("decode_ms_per_token", "prefill_s"):
            b, hh = d["base"][k], d["head"][k]
            if None in b or None in hh:
                why.append(f"{case} {k}: missing {b} {hh}")
                continue
            spread = abs(b[0] - b[1]) / min(b)
            tol = max(0.05, spread)
            if _med(hh) > _med(b) * (1 + tol):
                why.append(f"{case} {k}: head {hh} vs base {b} (> {tol:.1%})")
        pb, ph = _med(d["base"]["peak_alloc_gb"]), _med(d["head"]["peak_alloc_gb"])
        if pb is not None and ph is not None and ph > pb * 1.05 + 0.25:
            why.append(f"{case} peak: head {ph} GB vs base {pb} GB")
    s = _sft(obs)
    for st, r in s.items():
        ls = r.get("losses") or []
        if not ls or any(not math.isfinite(x) for x in ls):
            why.append(f"sft {st}: losses {ls}")
        if not (r.get("lora_b_after_min") or 0) > 0:
            why.append(f"sft {st}: adapters did not move ({r.get('lora_b_after_min')})")
    sb, sh = s["base"].get("step_s") or [], s["head"].get("step_s") or []
    if len(sb) > 1 and len(sh) > 1 and _med(sh[1:]) > _med(sb[1:]) * 1.10:
        why.append(f"sft step time: head {sh} vs base {sb}")
    lb, lh = (s["base"].get("losses") or [None])[0], (s["head"].get("losses") or [None])[0]
    if lb and lh and abs(lh - lb) / abs(lb) > 0.02:
        why.append(f"sft step-1 loss: head {lh} vs base {lb}")
    return bool(why), "; ".join(why) or "head within tolerance on tokens, prefill, decode, memory and SFT"
