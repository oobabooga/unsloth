#!/usr/bin/env python3
"""Criteria for probes/i8vs_probe.py: one batched INT8 vs SageAttention vs SDPA benchmark of the head tree on this AMD
GPU. Regression mode with the base unmeasured: the verdict judges the head alone.

Gates (INCONCLUSIVE when missing): the kernel grid ran, Studio INT8 engaged in at least one cell, every arm of every
cell carries a status (OK / NOT_RUN with reason / ERROR). Worse = an engaged INT8 cell outside Studio's own accuracy
gate against fp32 (cosine < 0.999 or rel-L1 > 0.05 on the probe inputs). Speed is reported, not judged."""
from __future__ import annotations

TITLE = "gfx1151: Studio INT8 attention vs SageAttention vs SDPA, batched benchmark (head tree)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "discrete_gpu", "windows", "windows_rocm_wddm", "multi_gpu"]


def _rows(st: dict) -> list:
    return [r for kb in (st.get("kernels") or {}).values() for r in (kb.get("rows") or [])]


def gates(obs: dict) -> list:
    head = obs.get("head") or {}
    rows = _rows(head)
    engaged = [r for r in rows if "int8" in (r.get("ms") or {})]
    unstated = [r for r in rows if not r.get("status")]
    return [("kernel rows", bool(rows), f"{len(rows)} rows"),
            ("int8 engaged", bool(engaged), f"{len(engaged)} cells with INT8 timed"),
            ("every arm has a status", not unstated, f"{len(unstated)} rows without status")]


def head_is_worse(base: dict, head: dict) -> tuple:
    bad = []
    for r in _rows(head):
        acc = ((r.get("acc") or {}).get("probe") or {}).get("int8")
        if acc and (acc["cos"] < 0.999 or acc["rel_l1"] > 0.05):
            bad.append(f"{r['dtype']} D{r['D']} H{r['H']} S{r['S']}: cos {acc['cos']} rel-L1 {acc['rel_l1']}")
    if bad:
        return True, "INT8 outside its accuracy gate: " + "; ".join(bad[:8])
    rows = _rows(head)
    n = sum(1 for r in rows if "int8" in (r.get("ms") or {}))
    return False, f"INT8 inside its accuracy gate in all {n} timed cells; numbers in bench_head/bench_attn.md"
