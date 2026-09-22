#!/usr/bin/env python3
"""Criteria: an offloaded activation must come back in the dtype it went out in.

Judges observations from probes/gc_offload_dtype_probe.py for unslothai/unsloth-zoo PR 1316.

The defect: `UnslothCheckpointFunction` handed the backward recompute the staging BUFFER,
whose dtype is whatever checkpointing was initialised with, instead of the activation. On a
FORCE_FLOAT32 family (qwen3_5, gemma3) loaded on a GPU with no bfloat16, that is a bfloat16
buffer holding float16 hidden states, so the recompute ran in bfloat16.

What this criteria does NOT claim. The report the PR is named after is an LLVM abort, and
that abort is a gfx10 property: Triton cannot lower bf16 there, so the wrong-dtype tensor
kills the process instead of raising. gfx1151 has bf16, so the abort is unreachable on this
host and `no_bf16_gpu` is declared as a gap. What is measurable here is the cause: which
dtype the recompute ran in and whether the bytes survived the round trip.

Three cases, and the third matters as much as the first:

  bf16_buffers_fp16_activation   the reported bug. Must recompute in float16.
  fp16_buffers_fp32_activation   gemma3's float32 activations over 16-bit buffers.
  bf16_buffers_bf16_activation   the CONTROL: what almost every user actually has. Must be
                                 unchanged, and must still be offloaded, or the "fix" is
                                 just a switched-off feature.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 1316: the dtype a checkpoint recompute runs in, base vs head"
MODE = "differential"

# Authored, not derived. Every capability the CHANGE touches, so the report bounds itself:
# the offload path is GPU-only; it is reached on ROCm here but the same code runs on CUDA
# and XPU; the reported crash needs a GPU without bf16; double buffering is only ON by
# default on a discrete GPU; the buffers are per-device; the PR's own evidence is Windows.
NEEDS = ["gpu", "rocm", "nvidia", "xpu", "no_bf16_gpu", "discrete_gpu", "multi_gpu", "windows"]

DEFECT_CASE = "bf16_buffers_fp16_activation"
CONTROL_CASE = "bf16_buffers_bf16_activation"
CASES = (DEFECT_CASE, "fp16_buffers_fp32_activation", CONTROL_CASE)


def _passes(o: dict, case: str) -> list[dict]:
    return ((o.get("cases") or {}).get(case) or {}).get("passes") or []


def _case_error(o: dict, case: str):
    return ((o.get("cases") or {}).get(case) or {}).get("error")


def _clean(o: dict, case: str) -> tuple[bool, str]:
    """Did every pass of `case` keep the activation's dtype and its exact bytes?"""
    if _case_error(o, case):
        return False, f"{case} raised: {str(_case_error(o, case)).strip().splitlines()[-1][:160]}"
    passes = _passes(o, case)
    if not passes:
        return False, f"{case} produced no passes"
    want = ((o.get("cases") or {}).get(case) or {}).get("activation_dtype")
    for p in passes:
        if p.get("recompute_dtypes") != [want]:
            return False, (f"{case} pass {p.get('pass')}: the recompute ran in "
                           f"{p.get('recompute_dtypes')}, the forward saw {want}")
        if not p.get("recompute_bitexact"):
            return False, f"{case} pass {p.get('pass')}: the recompute got different bytes back"
        if not p.get("grad_correct"):
            return False, f"{case} pass {p.get('pass')}: the gradient is wrong"
    return True, f"{case}: recompute in {want}, bytes identical, gradient exact"


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    """Non-vacuity. Each of these, if untrue, makes the comparison mean nothing."""
    out: list[tuple[str, bool, str]] = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name}: module imported", not o.get("import_error"),
                    str(o.get("import_error", "")).strip().splitlines()[-1][:180]
                    if o.get("import_error") else
                    f"gradient_checkpointing from {o.get('gc_file')} "
                    f"(unsloth: {o.get('unsloth_import', 'unrecorded')})"))
        # Without this both legs could be measuring the same installed copy, and the
        # differential would compare a state against itself.
        out.append((f"{name}: source came from ITS OWN checkout", bool(o.get("gc_file_in_checkout")),
                    f"{o.get('gc_file')} under {o.get('checkout')}"))
        out.append((f"{name}: a GPU was visible", bool(o.get("cuda_available")),
                    f"{o.get('device_name', '-')} {o.get('gcn_arch', '')} "
                    f"torch {o.get('torch_version')} hip {o.get('torch_hip')}"))
        # The whole defect only exists for an activation that is actually offloaded.
        # A run where nothing crossed the bus proves nothing at either state.
        ctrl = _passes(o, CONTROL_CASE)
        offloaded = bool(ctrl) and all(p.get("offloaded") for p in ctrl)
        out.append((f"{name}: the activation was really offloaded", offloaded,
                    "CPU_INDEX after forward: "
                    + ", ".join(str(p.get("cpu_index_after_forward")) for p in ctrl)))
    return out


def table(obs: dict) -> str:
    rows = ["| case | state | recompute ran in | forward saw | bytes identical | grad exact "
            "| offloaded | double buffer |", "|---|---|---|---|---|---|---|---|"]
    for case in CASES:
        for name in ("base", "head", "merge"):
            o = obs.get(name)
            if not o:
                continue
            entry = (o.get("cases") or {}).get(case) or {}
            if entry.get("error"):
                last = str(entry["error"]).strip().splitlines()[-1][:80]
                rows.append(f"| `{case}` | {name} | raised: `{last}` | | | | | |")
                continue
            for p in entry.get("passes") or []:
                rows.append(
                    f"| `{case}` | {name} pass {p.get('pass')} "
                    f"| `{', '.join(p.get('recompute_dtypes') or [])}` "
                    f"| `{entry.get('activation_dtype')}` "
                    f"| {'yes' if p.get('recompute_bitexact') else '**NO**'} "
                    f"| {'yes' if p.get('grad_correct') else '**NO**'} "
                    f"| {'yes' if p.get('offloaded') else 'no'} "
                    f"| {'on' if p.get('use_double_buffer') else 'off'} |")

    # Informational, deliberately not a gate: MINIMUM_SIZE moved from elements to bytes,
    # so the 2MB cutoff now means the same amount of memory in every width. A float32
    # activation of 3MB is below the old element cutoff and above the new byte one.
    extra = ["", "MINIMUM_SIZE and a 3 MB float32 activation over bfloat16 buffers "
                 "(an intended behaviour change, not a defect):", "",
             "| state | MINIMUM_SIZE | 3 MB float32 activation offloaded |", "|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        t = o.get("threshold") or {}
        extra.append(f"| {name} | {t.get('minimum_size_value')} | "
                     f"{'yes' if t.get('offloaded') else 'no'} |")

    # Stated rather than assumed: on a unified-memory APU (Strix Halo is one)
    # `_double_buffer_disabled()` turns GPU_BUFFERS_B off by default, and that buffer is
    # one of the lines this PR changes. The probe sets UNSLOTH_DISABLE_DOUBLE_BUFFER=0 so
    # the path is exercised wherever the run lands; it engages from the second pass, once
    # free memory clears the headroom. Whether it actually did is in the table.
    engaged = sorted({
        f"{name} pass {p.get('pass')}"
        for name in ("base", "head", "merge") if obs.get(name)
        for case in CASES for p in _passes(obs[name], case) if p.get("use_double_buffer")
    })
    note = ["", "`GPU_BUFFERS_B` (double buffering) is one of the buffers this PR changes, and "
                "`_double_buffer_disabled()` turns it OFF by default on a unified-memory APU. "
                "The probe sets `UNSLOTH_DISABLE_DOUBLE_BUFFER=0` to force it on regardless. "
                + (f"It was actually in use for: {', '.join(engaged)}."
                   if engaged else
                   "**It never engaged in this run**, so the `GPU_BUFFERS_B` line is UNTESTED here.")]
    return "\n".join(rows + extra + note)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    """The base must run the recompute on the wrong dtype, or there is nothing to fix."""
    ok, why = _clean(base, DEFECT_CASE)
    if ok:
        return False, f"the base handled the reported case correctly ({why})"
    return True, f"base: {why}"


def head_is_fixed(head: dict) -> tuple[bool, str]:
    """Every case clean, AND the control still offloads: a fix that silently stops
    offloading would pass a dtype check while costing exactly the VRAM the feature saves."""
    details = []
    for case in CASES:
        ok, why = _clean(head, case)
        if not ok:
            return False, why
        details.append(why)
    ctrl = _passes(head, CONTROL_CASE)
    if not all(p.get("offloaded") for p in ctrl):
        return False, "the control case stopped offloading, so the VRAM saving is gone"
    return True, "; ".join(details)
