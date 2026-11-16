#!/usr/bin/env python3
"""Criteria: is the head's Triton INT8 attention on this AMD GPU no slower than the base at any shape, with the same
output (bit for bit, or inside the accuracy gate), and does no INT8 attention test that passes at the base fail at
the head? Judges probes/i8ts_probe.py.

Timing (head state's rows): ABBA rounds, one fresh process per arm. Per shape and metric ratio = median over rounds
of mean(head) / mean(base); noise = max(A/A spread of the base processes, 2%). Worse = eager or graph ratio above
1 + noise. Accuracy gate (gain 1, 3 heads x 256 query rows vs fp64): rel L1 <= 0.05, worst-row cosine >= 0.999."""

from __future__ import annotations

import statistics as st
from collections import defaultdict

TITLE = "gfx1151: Studio Triton INT8 attention, head vs base (ABBA timing, exclusive GPU)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "discrete_gpu", "windows", "windows_rocm_wddm", "multi_gpu"]


def _timing(head: dict, arm: str = "i8_head") -> dict:
    """The committed head (arm i8_head) against the base; arm i8_headx is a candidate config, shown, not judged."""
    by = defaultdict(list)
    for r in head.get("rows") or []:
        if (r.get("mode") == "time" and r.get("t") and "error" not in r["t"]
                and r.get("arm", r["pkg"]) in ("i8_base", arm)):
            by[r["shape"]].append(r)
    res = {}
    for shape, lst in by.items():
        cell = {"engaged": all(r.get("engaged") for r in lst)}
        for m in ("eager_ms", "graph_ms", "loop_ms", "host_us"):
            per = defaultdict(lambda: {"b": [], "h": []})
            for r in lst:
                if r["t"].get(m):
                    per[r["rep"]]["b" if r["pkg"] == "i8_base" else "h"].append(r["t"][m])
            ratios = [st.mean(x["h"]) / st.mean(x["b"]) for x in per.values() if x["b"] and x["h"]]
            base = [v for x in per.values() for v in x["b"]]
            if ratios:
                spread = (max(base) - min(base)) / st.median(base) if len(base) > 1 else 0.0
                cell[m] = {"ratio": st.median(ratios), "noise": max(spread, 0.02), "base": st.median(base),
                           "head": st.median([v for x in per.values() for v in x["h"]])}
        res[shape] = cell
    return res


def _acc(head: dict) -> dict:
    return {r["shape"]: r for r in head.get("rows") or [] if r.get("mode") == "acc" and r.get("gain") == 1.0
            and r.get("arm", "i8_head") == "i8_head"}


def _acc_ok(r: dict) -> bool:
    h = r.get("head") or {}
    return bool(r.get("engaged")) and (bool(r.get("bit_identical")) or (
        h.get("finite") and h.get("rel_l1", 1) <= 0.05 and h.get("worst_cos", 0) >= 0.999))


def gates(obs: dict) -> list:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    t = _timing(head)
    out = [("head rows timed", bool(t), f"{len(t)} shapes"),
           ("INT8 Triton path engaged in every timed process", bool(t) and all(c["engaged"] for c in t.values()),
            "native calls 0 and triton calls > 0 per row"),
           ("accuracy rows", bool(_acc(head)), f"{len(_acc(head))} shapes")]
    for name, s in (("base", base), ("head", head)):
        p = s.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("passed", 0) > 0, f"{p.get('passed')} passed {p.get('failed')} failed"
                                                                  f" {p.get('skipped')} skipped"))
    cases = (head.get("pytest") or {}).get("cases") or {}
    card = {k: v for k, v in cases.items() if "on_the_card" in k and "test_diffusion_int8_attention_rocm" in k}
    out.append(("ROCm card tests ran at head", bool(card) and all(v == "passed" for v in card.values()),
                f"{sum(v == 'passed' for v in card.values())}/{len(card)} passed"))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    bad = []
    for shape, c in _timing(head).items():
        for m in ("eager_ms", "graph_ms"):
            x = c.get(m)
            if x and x["ratio"] > 1 + x["noise"]:
                bad.append(f"{shape} {m} {x['ratio']:.3f} > 1 + {x['noise']:.3f}")
    for shape, r in _acc(head).items():
        if not _acc_ok(r):
            bad.append(f"{shape} accuracy {r.get('head')}")
    bc = (base.get("pytest") or {}).get("cases") or {}
    hc = (head.get("pytest") or {}).get("cases") or {}
    new_fail = [k for k, v in hc.items() if v == "failed" and bc.get(k) != "failed"]
    bad += [f"new test failure {k}" for k in new_fail]
    return (bool(bad), "; ".join(bad[:12]) if bad else "no shape slower beyond noise, accuracy kept, no new failures")


def table(obs: dict) -> str:
    head = obs.get("head") or {}
    t, acc = _timing(head), _acc(head)
    lines = ["| shape | eager ms base | eager ms head | ratio | graph ms base | graph ms head | ratio | noise | bits |"
             " rel L1 | worst cos |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for shape, c in t.items():
        e, g, a = c.get("eager_ms") or {}, c.get("graph_ms") or {}, acc.get(shape) or {}
        h = a.get("head") or {}
        lines.append(f"| {shape} | {e.get('base', 0):.3f} | {e.get('head', 0):.3f} | {e.get('ratio', 0):.3f} | "
                     f"{g.get('base', 0):.3f} | {g.get('head', 0):.3f} | {g.get('ratio', 0):.3f} | "
                     f"{e.get('noise', 0):.3f} | {'same' if a.get('bit_identical') else 'DIFF'} | "
                     f"{h.get('rel_l1', float('nan')):.4f} | {h.get('worst_cos', float('nan')):.5f} |")
    return "\n".join(lines)
