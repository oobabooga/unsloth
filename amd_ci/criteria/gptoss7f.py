#!/usr/bin/env python3
"""Criteria: zoo main after the Round 7f GPT-OSS decode merges (#2233, #2236, #2246) on gfx1151, base = the commit
before #2233. Pairs with probes/gptoss7f_probe.py (all arms in the head invocation, bench arms A B B A).

Non-vacuity gates (INCONCLUSIVE unless all hold): model downloaded; head tree == the expected zoo main tree; every
bench arm exited 0 with no error / OOM rows, imported unsloth_zoo from its own checkout, ran on an AMD gfx1151; the
head arms report the 7f engagement block (the levers exist at the head; launches through triton_fast_launch are
reported, not gated, since the routed kernels may not run on HIP); both pytest arms ran (rc 0/1, outcomes > 0), each imported its own zoo,
saw HIP gfx1151.
head_is_worse (REGRESSION) if any of:
  - greedy tokens: head differs from base at a step whose top-2 margin is >= NEAR_TIE
  - per case: head median decode ms/token or prefill s above base by more than max(5%, base A/A spread)
  - per case: head peak allocated memory above base by more than 5% + 0.25 GB
  - a head bench arm made bitsandbytes direct calls on ROCm (#2246 must stay off), or fast-launch / fused decode
    report enabled on HIP while their gates say HIP is excluded (a silent gate change)
  - a test that fails / errors at the head and does not fail at the base (new test files count: absent at the base)
"""

from __future__ import annotations

import os
import statistics

TITLE = "zoo main after the Round 7f GPT-OSS decode merges (#2233, #2236, #2246) on gfx1151, base = before #2233"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "windows", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]
NEAR_TIE = 0.25
BENCH = ("bench_base_1", "bench_head_1", "bench_head_2", "bench_base_2")


def _head(obs):
    return obs.get("head") or {}


def _arms(obs):
    return _head(obs).get("arms") or {}


def _norm(p):
    return os.path.normcase(os.path.abspath(p)) if p else ""


def _rows(arm):
    return {(r["ctx"], r["batch"]): r for r in ((arm.get("result") or {}).get("rows") or [])}


def _eng(arm):
    return (arm.get("result") or {}).get("engagement") or {}


def _pt(obs):
    return _head(obs).get("pytest") or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    h, arms = _head(obs), _arms(obs)
    out = [("model downloaded", (h.get("download") or {}).get("rc") == 0, f"{h.get('model')} {h.get('download')}"),
           ("head tree == expected zoo main tree", bool(h.get("head_tree")) and h.get("head_tree") == h.get("expected_tree"),
            f"head {h.get('head_tree')} expected {h.get('expected_tree')} base sha {h.get('base_sha')}"),
           ("all 4 bench arms present", all(n in arms for n in BENCH), ", ".join(arms))]
    cos = {"head": h.get("checkout") or "", "base": os.path.join(os.path.dirname(h.get("checkout") or ""), "base")}
    for name in BENCH:
        a = arms.get(name) or {}
        r = a.get("result") or {}
        bad = [f"{k}: {v.get('error') or v.get('oom')}" for k, v in _rows(a).items() if v.get("error") or v.get("oom")]
        ok = a.get("rc") == 0 and bool(r) and not bad
        out.append((f"{name}: exited 0, no error / OOM", ok,
                    f"rc={a.get('rc')} {'; '.join(bad)} {str(a.get('tail'))[-300:] if not ok else ''}"))
        out.append((f"{name}: zoo from its own checkout",
                    bool(r.get("zoo_file")) and _norm(r.get("zoo_file")).startswith(_norm(cos.get(a.get("state"), "?"))),
                    f"{r.get('zoo_file')} vs {cos.get(a.get('state'))}"))
        out.append((f"{name}: AMD gfx1151 GPU", any(s in str(r.get("gpu")) for s in ("8060S", "gfx1151", "Radeon")),
                    str(r.get("gpu"))))
    for name in ("bench_head_1", "bench_head_2"):
        e = _eng(arms.get(name) or {})
        out.append((f"{name}: 7f levers present at the head (engagement block)",
                    "fast_launch_enabled" in e and "fused_decode_enabled" in e and "bnb_direct_on" in e, str(e)))
    pt = _pt(obs)
    for st in ("base", "head"):
        p = pt.get(st) or {}
        n = sum(len(p.get(k) or []) for k in ("passed", "failed", "skipped", "errors"))
        out.append((f"pytest {st}: ran and collected", p.get("rc") in (0, 1) and n > 0,
                    f"rc={p.get('rc')} outcomes={n} {str(p.get('tail'))[-300:] if p.get('rc') not in (0, 1) else ''}"))
        env = p.get("env") or {}
        out.append((f"pytest {st}: own zoo checkout, HIP gfx1151",
                    _norm(env.get("zoo")).startswith(_norm(cos[st])) and bool(env.get("hip")) and bool(env.get("avail"))
                    and "gfx1151" in str(env.get("arch")), str(env)))
    return out


def _tokens_cmp(ref, other):
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


def _new_failures(obs):
    pt = _pt(obs)
    b, h = pt.get("base") or {}, pt.get("head") or {}
    bad_b = set(b.get("failed") or []) | set(b.get("errors") or [])
    return sorted((set(h.get("failed") or []) | set(h.get("errors") or [])) - bad_b)


def table(obs: dict) -> str:
    h, arms = _head(obs), _arms(obs)
    lines = [f"Model `{h.get('model')}`; base `{h.get('base_sha')}`, head tree `{h.get('head_tree')}`", "",
             "| case | metric | base run 1 / 2 | head run 1 / 2 | head / base (median) |", "|---|---|---|---|---|"]
    for case, d in _perf(obs).items():
        for k in ("prefill_s", "decode_ms_per_token", "peak_alloc_gb"):
            b, hh = _med(d["base"][k]), _med(d["head"][k])
            lines.append(f"| {case[0]}x{case[1]} | {k} | {d['base'][k]} | {d['head'][k]} | "
                         f"{f'{hh / b:.3f}' if b and hh else '-'} |")
    g = {n: (arms.get(n) or {}).get("gen") or {} for n in ("bench_base_1", "bench_head_1")}
    lines += ["", f"Greedy tokens, head vs base first divergence (row, step, margin base, margin head): "
              f"{_tokens_cmp(g['bench_base_1'], g['bench_head_1']) or 'identical'}", "",
              "Engagement per arm:"]
    for n in BENCH:
        lines.append(f"- {n}: {_eng(arms.get(n) or {})}")
    pt = _pt(obs)
    lines += ["", "| pytest | rc | passed | failed | errors | skipped | absent files | s |", "|---|---|---|---|---|---|---|---|"]
    for st in ("base", "head"):
        p = pt.get(st) or {}
        lines.append(f"| {st} | {p.get('rc')} | {len(p.get('passed') or [])} | {len(p.get('failed') or [])} | "
                     f"{len(p.get('errors') or [])} | {len(p.get('skipped') or [])} | {p.get('absent')} | {p.get('s')} |")
    for st in ("base", "head"):
        p = pt.get(st) or {}
        if p.get("failed") or p.get("errors"):
            lines.append(f"- {st} failing: {sorted((p.get('failed') or []) + (p.get('errors') or []))[:40]}")
    lines.append(f"- failing at head only: {_new_failures(obs) or 'none'}")
    lines.append("")
    lines.append("Arms: " + ", ".join(f"{n} rc={a.get('rc')} {a.get('s')}s" for n, a in arms.items()))
    return "\n".join(lines)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    obs = {"head": head, "base": base}
    why = []
    arms = _arms(obs)
    g = {n: (arms.get(n) or {}).get("gen") or {} for n in ("bench_base_1", "bench_head_1")}
    for row, step, mr, mo in _tokens_cmp(g["bench_base_1"], g["bench_head_1"]):
        m = min(x for x in (mr, mo) if x is not None) if (mr is not None or mo is not None) else None
        if m is None or m >= NEAR_TIE:
            why.append(f"tokens: row {row} diverges at step {step}, margins base {mr} head {mo}")
    for case, d in _perf(obs).items():
        for k in ("decode_ms_per_token", "prefill_s"):
            b, hh = d["base"][k], d["head"][k]
            if None in b or None in hh:
                why.append(f"{case} {k}: missing {b} {hh}")
                continue
            tol = max(0.05, abs(b[0] - b[1]) / min(b))
            if _med(hh) > _med(b) * (1 + tol):
                why.append(f"{case} {k}: head {hh} vs base {b} (> {tol:.1%})")
        pb, ph = _med(d["base"]["peak_alloc_gb"]), _med(d["head"]["peak_alloc_gb"])
        if pb is not None and ph is not None and ph > pb * 1.05 + 0.25:
            why.append(f"{case} peak: head {ph} GB vs base {pb} GB")
    for n in ("bench_head_1", "bench_head_2"):
        e = _eng(arms.get(n) or {})
        if (e.get("bnb_direct_calls") or {}).get("direct", 0) or e.get("bnb_direct_on"):
            why.append(f"{n}: bnb direct active on ROCm {e.get('bnb_direct_on')} {e.get('bnb_direct_calls')}")
        if e.get("fast_launch_enabled") or (e.get("fast_launch_stats") or {}).get("fast", 0):
            why.append(f"{n}: fast launch enabled on HIP {e.get('fast_launch_stats')}")
        if e.get("fused_decode_enabled") or e.get("rope_certified"):
            why.append(f"{n}: fused decode / rope_decode enabled on HIP ({e.get('fused_decode_enabled')}, "
                       f"{e.get('rope_certified')})")
    nf = _new_failures(obs)
    if nf:
        why.append(f"tests failing at head only: {nf[:20]}")
    return bool(why), "; ".join(why) or "head within tolerance on tokens, prefill, decode, memory, gates and tests"
