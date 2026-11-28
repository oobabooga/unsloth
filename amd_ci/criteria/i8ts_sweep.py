#!/usr/bin/env python3
"""Criteria: a config sweep of Studio's Triton INT8 attention on this AMD GPU (head checkout), with the head's output
compared against the base's at every protocol shape. Judges probes/i8ts_probe.py run with --steps variants,acc,sweep.

Regression = a shape whose head output differs from the base and fails the accuracy gate (gain 1: rel L1 <= 0.05,
worst-row cosine >= 0.999), or a new test failure. The sweep and the compiled-variant counts are reported, not judged."""

from __future__ import annotations

TITLE = "gfx1151: Studio Triton INT8 attention config sweep, head vs base output"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "discrete_gpu", "windows", "windows_rocm_wddm", "multi_gpu"]


def _rows(st: dict, mode: str) -> list:
    return [r for r in st.get("rows") or [] if r.get("mode") == mode]


def gates(obs: dict) -> list:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    acc = _rows(head, "acc")
    out = [("accuracy rows", len({r["shape"] for r in acc}) >= 17, f"{len(acc)} rows"),
           ("sweep rows", bool(_rows(head, "sweep")), f"{len(_rows(head, 'sweep'))} rows"),
           ("variant counts", bool(_rows(head, "variants")), f"{len(_rows(head, 'variants'))} rows")]
    for name, s in (("base", base), ("head", head)):
        p = s.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("passed", 0) > 0,
                    f"{p.get('passed')} passed {p.get('failed')} failed {p.get('skipped')} skipped"))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    bad = []
    for r in _rows(head, "acc"):
        h = r.get("head") or {}
        ok = r.get("engaged") and (r.get("bit_identical") or (r.get("gain") != 1.0) or (
            h.get("rel_l1", 1) <= 0.05 and h.get("worst_cos", 0) >= 0.999))
        if not ok:
            bad.append(f"{r['shape']} gain {r.get('gain')}")
    bc = (base.get("pytest") or {}).get("cases") or {}
    hc = (head.get("pytest") or {}).get("cases") or {}
    bad += [f"new test failure {k}" for k, v in hc.items() if v == "failed" and bc.get(k) != "failed"]
    return bool(bad), "; ".join(bad[:12]) if bad else "head output equals the base or passes the gate; no new failures"


def table(obs: dict) -> str:
    head = obs.get("head") or {}
    v = _rows(head, "variants")
    return "compiled variants: " + "; ".join(f"{r['pkg']} {r['compiled_variants']}" for r in v)
