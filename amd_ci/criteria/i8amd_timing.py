#!/usr/bin/env python3
"""Criteria for probes/i8amd_probe.py: Studio's Triton INT8 attention on gfx1151, base vs head, with comfy-kitchen's
HIP INT8 kernel as the outside reference.

Defect (base): the INT8 attention suite has a test that fails on every ROCm host
(``test_the_attention_kernel_compiles_through_the_8bit_prefetch_fix``). Fixed (head): it passes, no test that passes at
the base fails at the head, every head row is engaged and inside the accuracy gate, no shape is slower than the base
beyond noise (eager and CUDA graph), and the end-to-end INT8 render engaged the kernel and stays within LPIPS of
native. Timing: rounds of base, head, ck, ck, head, base, one fresh process per arm; ratio per shape = median over
rounds of mean(head) / mean(base); noise = max(spread of the base processes, 2%)."""

from __future__ import annotations

import statistics as st
from collections import defaultdict

TITLE = "gfx1151: Studio Triton INT8 attention vs base and comfy-kitchen HIP INT8 (timing, exclusive GPU)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "discrete_gpu", "windows", "windows_rocm_wddm", "multi_gpu"]
DEFECT = "test_the_attention_kernel_compiles_through_the_8bit_prefetch_fix"
LPIPS_MAX = 0.10


def _rows(head: dict, mode: str) -> list:
    return [r for r in head.get("rows") or [] if r.get("mode") == mode]


def _cell(by_rep: dict, m: str, a: str, b: str) -> dict | None:
    ratios, xa, xb = [], [], []
    for x in by_rep.values():
        va = [r["t"][m] for r in x.get(a, []) if r["t"].get(m)]
        vb = [r["t"][m] for r in x.get(b, []) if r["t"].get(m)]
        xa += va
        xb += vb
        if va and vb:
            ratios.append(st.mean(va) / st.mean(vb))
    if not ratios:
        return None
    spread = (max(xb) - min(xb)) / st.median(xb) if len(xb) > 1 else 0.0
    return {"ratio": st.median(ratios), "noise": max(spread, 0.02), a: st.median(xa), b: st.median(xb)}


def timing(head: dict, arms = ("i8_base", "i8_head", "ck")) -> dict:
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in _rows(head, "time"):
        arm = r.get("arm") or r.get("pkg")
        if arm in arms and r.get("t") and "error" not in r["t"]:
            by[r["shape"]][r["rep"]][arm].append(r)
    out = {}
    for shape, reps in by.items():
        rows = [r for x in reps.values() for lst in x.values() for r in lst]
        cell = {"engaged": all(r.get("engaged") for r in rows), "arms": sorted({a for x in reps.values() for a in x})}
        for m in ("eager_ms", "graph_ms"):
            cell[m] = _cell(reps, m, arms[1], arms[0])
            if len(arms) > 2:
                cell[m + "_vs_ck"] = _cell(reps, m, arms[1], arms[2])
                cell[m + "_base_vs_ck"] = _cell(reps, m, arms[0], arms[2])
        out[shape] = cell
    return out


def acc(head: dict) -> dict:
    return {r["shape"]: r for r in _rows(head, "acc") if r.get("gain") == 1.0}


def _acc_ok(r: dict) -> bool:
    h = r.get("head") or {}
    return bool(r.get("engaged")) and (bool(r.get("bit_identical")) or (
        h.get("finite") and h.get("rel_l1", 1) <= 0.05 and h.get("worst_cos", 0) >= 0.999))


def _cases(s: dict) -> dict:
    return (s.get("pytest") or {}).get("cases") or {}


def _defect(cases: dict) -> list:
    return [v for k, v in cases.items() if k.endswith("::" + DEFECT)]


def _e2e(s: dict) -> dict:
    cells = (((s.get("e2e") or {}).get("obs") or {}).get("cells")) or {}
    return cells if isinstance(cells, dict) else {}


def _int8_calls(cell: dict) -> int:
    n = 0
    for r in cell.get("renders") or []:
        st_ = ((r.get("engaged") or {}).get("int8_attention")) or {}
        n = max(n, int(st_.get("int8") or 0))
    return n


def gates(obs: dict) -> list:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    t = timing(head)
    out = [("head rows timed (base, head, ck)", bool(t) and all(len(c["arms"]) == 3 for c in t.values()),
            f"{len(t)} shapes, arms {sorted({a for c in t.values() for a in c['arms']})}"),
           ("INT8 Triton path and ck engaged in every timed process", bool(t) and all(c["engaged"] for c in t.values()),
            "native calls 0 and triton calls > 0 per Studio row; ck ran"),
           ("accuracy rows", bool(acc(head)), f"{len(acc(head))} shapes")]
    for name, s in (("base", base), ("head", head)):
        p = s.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("passed", 0) > 0,
                    f"{p.get('passed')} passed {p.get('failed')} failed {p.get('skipped')} skipped"))
        if "e2e" not in base and "e2e" not in head:
            continue  # no --e2e-spec on this run (the Windows leg): nothing rendered, nothing to gate
        e = _e2e(s)
        native, i8 = e.get("i8e_wan_native") or {}, e.get("i8e_wan_int8") or {}
        out.append((f"{name} e2e native and INT8 renders", native.get("verdict") == "ok" and i8.get("verdict") == "ok",
                    f"native {native.get('verdict')} / int8 {i8.get('verdict')} ({(s.get('e2e') or {}).get('rc')})"))
    card = {k: v for k, v in _cases(head).items() if "on_the_card" in k and "test_diffusion_int8_attention_rocm" in k}
    out.append(("ROCm card tests ran at head", bool(card) and all(v == "passed" for v in card.values()),
                f"{sum(v == 'passed' for v in card.values())}/{len(card)} passed"))
    return out


def base_shows_defect(base: dict):
    d = _defect(_cases(base))
    return (bool(d) and all(v == "failed" for v in d), f"base {DEFECT}: {d}")


def head_problems(head: dict) -> list:
    bad = []
    hc = _cases(head)
    d = _defect(hc)
    if not d or any(v != "passed" for v in d):
        bad.append(f"{DEFECT}: {d}")
    bc = head.get("base_pytest_cases")
    if not isinstance(bc, dict):
        bad.append("no base test ids to compare with")
    else:
        bad += [f"new failure {k}" for k, v in hc.items() if v == "failed" and bc.get(k) != "failed"]
    for shape, c in timing(head).items():
        for m in ("eager_ms", "graph_ms"):
            x = c.get(m)
            if x and x["ratio"] > 1 + x["noise"]:
                bad.append(f"{shape} {m} head/base {x['ratio']:.3f} > 1 + {x['noise']:.3f}")
    for shape, r in acc(head).items():
        if not _acc_ok(r):
            bad.append(f"{shape} accuracy {r.get('head')}")
    i8 = _e2e(head).get("i8e_wan_int8") or {}
    if "e2e" not in head:
        return bad
    if _int8_calls(i8) <= 0:
        bad.append("e2e: INT8 attention served no call")
    if i8.get("lpips") is None or i8["lpips"] > LPIPS_MAX:
        bad.append(f"e2e LPIPS vs native {i8.get('lpips')}")
    return bad


def head_is_fixed(head: dict):
    bad = head_problems(head)
    return (not bad, "; ".join(bad[:12]) if bad else "defect test passes, nothing slower, accuracy kept")


def _f(x, k, nd = 3):
    v = (x or {}).get(k)
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def table(obs: dict) -> str:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    t, a = timing(head), acc(head)
    lines = ["Per-call ms (median of fresh-process medians); ratio = median over rounds; noise = base spread (min 2%).",
             "",
             "| shape | base eager | head eager | head/base | noise | ck eager | head/ck | base/ck | head/base graph |"
             " head/ck graph | bits | head rel L1 | head worst cos | ck rel L1 | ck worst cos |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for shape in sorted(t, key = lambda s: (t[s].get("eager_ms") or {}).get("i8_base", 0)):
        c = t[shape]
        e, g = c.get("eager_ms") or {}, c.get("graph_ms") or {}
        evc, gvc, ebc = c.get("eager_ms_vs_ck") or {}, c.get("graph_ms_vs_ck") or {}, c.get("eager_ms_base_vs_ck") or {}
        r = a.get(shape) or {}
        lines.append(f"| {shape} | {_f(e, 'i8_base')} | {_f(e, 'i8_head')} | {_f(e, 'ratio')} | {_f(e, 'noise')} | "
                     f"{_f(evc, 'ck')} | {_f(evc, 'ratio')} | {_f(ebc, 'ratio')} | {_f(g, 'ratio')} | {_f(gvc, 'ratio')} | "
                     f"{'same' if r.get('bit_identical') else 'diff'} | {_f(r.get('head'), 'rel_l1', 4)} | "
                     f"{_f(r.get('head'), 'worst_cos', 5)} | {_f(r.get('ck'), 'rel_l1', 4)} | {_f(r.get('ck'), 'worst_cos', 5)} |")
    ns = timing(head, ("i8_head_ns", "i8_nospec_ns"))
    if ns:
        lines += ["", "Sizes out of the attention compile key (i8_nospec) vs the head (sq / sk specialised on ROCm):", "",
                  "| shape | head eager | nospec eager | nospec/head | noise | nospec/head graph |", "|---|---|---|---|---|---|"]
        for shape, c in ns.items():
            e, g = c.get("eager_ms") or {}, c.get("graph_ms") or {}
            lines.append(f"| {shape} | {_f(e, 'i8_head_ns')} | {_f(e, 'i8_nospec_ns')} | {_f(e, 'ratio')} | "
                         f"{_f(e, 'noise')} | {_f(g, 'ratio')} |")
    var = [r for r in _rows(head, "variants")]
    if var:
        lines += ["", "Compiled variants over every shape, and the device AMDGCN of each attention variant:", ""]
        for r in var:
            isa = "; ".join(f"vgpr {x.get('vgpr_count')} waitcnt {x.get('s_waitcnt')} iu8 {x.get('wmma_iu8')} "
                            f"f16 {x.get('wmma_f16')} lds {x.get('shared')} scratch {x.get('private_segment_fixed_size')}"
                            for x in (r.get("attn_isa") or {}).values())
            lines.append(f"- {r.get('pkg')}: {r.get('compiled_variants')} | {isa}")
    for name, s in (("base", base), ("head", head)):
        e = _e2e(s)
        for tag in ("i8e_wan_native", "i8e_wan_int8"):
            c = e.get(tag) or {}
            lines.append(f"- e2e {name} {tag}: {c.get('verdict')} new_s {c.get('new_s')} steady_s {c.get('steady_s')} "
                         f"walls {c.get('walls')} LPIPS {c.get('lpips')} int8 calls {_int8_calls(c)}")
        p = s.get("pytest") or {}
        lines.append(f"- pytest {name}: {p.get('passed')} passed / {p.get('failed')} failed / {p.get('skipped')} skipped;"
                     f" failing {[k.split('::')[-1] for k, v in _cases(s).items() if v == 'failed'][:8]}")
    return "\n".join(lines)
