#!/usr/bin/env python3
"""Criteria: does the head's Triton INT8 attention on this AMD GPU give the base's output (bit for bit, or inside the
accuracy gate) at every protocol shape, and does no INT8 attention test that passes at the base fail at the head?
Judges probes/i8ts_probe.py run with --steps acc (no timing: the card may be shared).

Accuracy gate (gain 1, 3 heads x 256 query rows vs fp64): rel L1 <= 0.05, worst-row cosine >= 0.999."""

from __future__ import annotations

import json
from pathlib import Path

TITLE = "gfx1151: Studio Triton INT8 attention, head vs base output and tests (no timing)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "nvidia", "discrete_gpu", "windows", "windows_rocm_wddm", "multi_gpu"]
SHAPES = 17


def _acc(head: dict) -> dict:
    return {(r["shape"], r["gain"]): r for r in head.get("rows") or [] if r.get("mode") == "acc" and "gain" in r}


def _ok(r: dict) -> bool:
    h = r.get("head") or {}
    if not r.get("engaged"):
        return False
    if r.get("bit_identical"):
        return True
    return r.get("gain") == 1.0 and bool(h.get("finite")) and h.get("rel_l1", 1) <= 0.05 and h.get("worst_cos", 0) >= 0.999


def gates(obs: dict) -> list:
    base, head = obs.get("base") or {}, obs.get("head") or {}
    acc = _acc(head)
    shapes = {s for s, _ in acc}
    out = [("accuracy rows for every protocol shape", len(shapes) >= SHAPES, f"{len(shapes)} shapes, {len(acc)} rows"),
           ("INT8 Triton path engaged in both packages", bool(acc) and all(r.get("engaged") for r in acc.values()),
            "native calls 0 at base and head")]
    for name, s in (("base", base), ("head", head)):
        p = s.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("passed", 0) > 0,
                    f"{p.get('passed')} passed {p.get('failed')} failed {p.get('skipped')} skipped"))
    cases = (head.get("pytest") or {}).get("cases") or {}
    card = {k: v for k, v in cases.items() if "on_the_card" in k and "test_diffusion_int8_attention_rocm" in k}
    out.append(("ROCm card tests ran at head", bool(card) and all(v == "passed" for v in card.values()),
                f"{sum(v == 'passed' for v in card.values())}/{len(card)} passed"))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    bad = [f"{s} gain {g}: bits differ and gate fails {r.get('head')}" for (s, g), r in _acc(head).items() if not _ok(r)]
    bc = (base.get("pytest") or {}).get("cases") or {}
    hc = (head.get("pytest") or {}).get("cases") or {}
    bad += [f"new test failure {k}" for k, v in hc.items() if v == "failed" and bc.get(k) != "failed"]
    return bool(bad), "; ".join(bad[:12]) if bad else "same output bits at every shape and gain, no new test failures"


def table(obs: dict) -> str:
    lines = ["| shape | gain | bits | n diff | rel L1 head | rel L1 base | worst cos head | worst cos base |",
             "|---|---|---|---|---|---|---|---|"]
    for (s, g), r in sorted(_acc(obs.get("head") or {}).items()):
        h, b = r.get("head") or {}, r.get("base") or {}
        lines.append(f"| {s} | {g} | {'same' if r.get('bit_identical') else 'DIFF'} | {r.get('n_diff')} | "
                     f"{h.get('rel_l1', float('nan')):.4f} | {b.get('rel_l1', float('nan')):.4f} | "
                     f"{h.get('worst_cos', float('nan')):.5f} | {b.get('worst_cos', float('nan')):.5f} |")
    return "\n".join(lines)
