#!/usr/bin/env python3
"""Criteria: is the down-LoRA delta scattered back to token order (unsloth-zoo#887)?

JUDGES ONLY. Pairs with probes/moe_roworder_probe.py.

The thresholds are asymmetric on purpose. The bf16 noise floor for this shape is
around 0.008 relative, and the defect is a whole-row misplacement worth roughly 1.0,
so there are two orders of magnitude between them. Anything in the gap is neither,
and is deliberately not counted as fixed.
"""

from __future__ import annotations

TITLE = "unsloth-zoo#887: down-LoRA row order in the Triton grouped-GEMM MoE forward"
MODE = "differential"

# Declare what the CHANGE touches, not what the host happens to have: the gap list
# is NEEDS minus the host, so under-declaring here makes the report bound nothing.
# Only names capability.py can actually evaluate. An unrecognised name renders as
# "not available on this host" even when the host has it, which would make the gap
# list read as though the run measured less than it did.
NEEDS = ["rocm", "gpu", "nvidia", "multi_gpu", "windows", "xpu", "mlx"]

DEFECT = 0.05    # above this, rows are demonstrably misplaced
FIXED = 0.02     # at or below this, within the bf16 noise floor for this shape


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        err = o.get("error")
        out.append((f"{name} probe ran", err is None and o.get("worst") is not None,
                    err or f"worst rel err {o.get('worst')}"))
        # A run where the routing happened not to permute, or where the LoRA delta was
        # zero, would show no difference at either state and compare vacuously.
        out.append((f"{name} routing actually permutes rows", bool(o.get("routing_permutes")),
                    f"routing_permutes={o.get('routing_permutes')}, "
                    f"gather_is_permutation={o.get('gather_is_permutation')}"))
        out.append((f"{name} LoRA delta is non-zero", bool(o.get("delta_nonzero")),
                    f"delta_nonzero={o.get('delta_nonzero')}"))
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out.append(("base is the unfixed combination", bool(b.get("uses_plain_add")),
                f"base uses_plain_add={b.get('uses_plain_add')} "
                f"uses_index_add={b.get('uses_index_add')}"))
    out.append(("head carries the scatter", bool(h.get("uses_index_add")),
                f"head uses_index_add={h.get('uses_index_add')} "
                f"uses_plain_add={h.get('uses_plain_add')}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | forward | d lora_A | d lora_B | worst | device / arch |",
            "|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        if o.get("error"):
            rows.append(f"| {name} | - | - | - | - | error: {o['error'][:80]} |")
            continue
        rows.append(
            f"| {name} | {o.get('rel_forward', float('nan')):.4f} "
            f"| {o.get('rel_lora_A', float('nan')):.4f} "
            f"| {o.get('rel_lora_B', float('nan')):.4f} "
            f"| **{o.get('worst', float('nan')):.4f}** "
            f"| {o.get('device', '?')} / {o.get('arch', '?')} |")
    rows.append("")
    rows.append(f"Relative error against an independent fp32 per-token reference. "
                f"Defect threshold {DEFECT}, fixed threshold {FIXED}.")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    w = base.get("worst")
    if w is None:
        return False, f"base did not measure: {base.get('error')}"
    if w > DEFECT:
        return True, (f"base worst relative error {w:.4f} > {DEFECT}: forward "
                      f"{base.get('rel_forward'):.4f}, lora_A {base.get('rel_lora_A'):.4f}, "
                      f"lora_B {base.get('rel_lora_B'):.4f}")
    return False, (f"base worst relative error {w:.4f} <= {DEFECT}; the defect does not "
                   f"reproduce on this host, so the head result bounds nothing")


def head_is_fixed(head: dict) -> tuple[bool, str]:
    w = head.get("worst")
    if w is None:
        return False, f"head did not measure: {head.get('error')}"
    if w <= FIXED:
        return True, (f"head worst relative error {w:.4f} <= {FIXED}, within the bf16 "
                      f"noise floor")
    return False, (f"head worst relative error {w:.4f} > {FIXED}: rows still misplaced "
                   f"or only partly corrected")
