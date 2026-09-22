#!/usr/bin/env python3
"""Criteria: does the FP8 expert dequant decline an illegal Triton tile on ROCm?

unsloth-zoo#1321 turns on Triton's maximum tensor numel. The open question the
CUDA runs could not answer is whether ROCm picks a different cap, which would
make the fix either useless (cap lower than ours) or needlessly slow (higher).

Defect at the base: driving the path with a per-expert per-tensor scale over a
2048 x 2048 tile RAISES, because BLOCK_SIZE derives to 2048 and the kernel
cannot compile. Fixed at the head: the same call DECLINES (returns None) so the
caller's vectorized fallback handles it, while a legal 1024 x 1024 tile is
still taken at both states.

Pairs with probes/fp8_tile_limit_probe.py.
"""

from __future__ import annotations

TITLE = "FP8 expert dequant tile limit on ROCm, base versus head"
MODE = "differential"

# Authored, not derived from the host: the change is a guard on a Triton path
# that runs on CUDA and ROCm alike, so the NVIDIA and Windows legs it does NOT
# reach have to be named. discrete_gpu because gfx1151 is an integrated part and
# the models that provoked this (Mistral-Small-4-119B) never run on one.
NEEDS = ["gpu", "rocm", "nvidia", "discrete_gpu", "windows"]

_NUMEL_MARKERS = ("maximum tensor numel", "exceeds triton")


def _kernel_present(o: dict) -> bool:
    return bool(o.get("unsloth_fp8_kernel"))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}

        out.append((f"{name}: GPU visible to torch", bool(o.get("gpu_available")),
                    f"arch={o.get('arch')} torch={o.get('torch_version')} "
                    f"hip={o.get('torch_hip')}"))

        # Without the kernel the path returns None for every input, so the base
        # would "decline" the oversized tile and the comparison would be between
        # two identical nothings. This is the gate that keeps the run honest.
        out.append((f"{name}: unsloth FP8 Triton kernel importable", _kernel_present(o),
                    o.get("unsloth_fp8_kernel_error")
                    or "unsloth.kernels.fp8.weight_dequant_block imported"))

        # A stubbed Triton reports its cap as a placeholder object rather than a
        # number, which would make every reading below meaningless.
        limits = o.get("triton_limits") or {}
        real = {k: v for k, v in limits.items() if v.get("type") == "int"}
        out.append((f"{name}: Triton reports a numeric tile cap", bool(real),
                    f"{limits or o.get('triton_error') or 'nothing read'}"))

        # The legal tile is the non-vacuity control: if 1024 x 1024 does not go
        # through at this state, "declined" at the head says nothing about the
        # guard and everything about the hardware.
        legal = (o.get("legal") or {}).get("outcome")
        out.append((f"{name}: a legal 1024x1024 tile is still taken", legal == "returned",
                    f"legal tile outcome={legal} "
                    f"{(o.get('legal') or {}).get('error', '')}".strip()))
    return out


def table(obs: dict) -> str:
    rows = [
        "| state | Triton cap read here | module cap | 2048x2048 tile | 1024x1024 tile |",
        "|---|---|---|---|---|",
    ]
    for name in ("base", "head", "merge"):
        o = obs.get(name)
        if not o:
            continue
        limits = o.get("triton_limits") or {}
        read = ", ".join(f"{k.split('.')[-1]}={v['value']}" for k, v in limits.items()) or "-"
        over = (o.get("oversized") or {}).get("outcome", "-")
        if over == "raised":
            # The interesting part of a CompilationError is its LAST line; the
            # first 90 characters are the kernel source echo, identical for
            # every failure. Newlines and pipes flattened or the row stops
            # being a table row.
            error = " ".join(str((o.get("oversized") or {}).get("error", "")).split())
            tail = error.split("ValueError")[-1] if "ValueError" in error else error[-120:]
            over = f"raised: `{tail.replace('|', '/')[:120]}`"
        legal = (o.get("legal") or {}).get("outcome", "-")
        rows.append(f"| {name} | {read} | {o.get('module_limit', '-')} | {over} | {legal} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    over = base.get("oversized") or {}
    outcome = over.get("outcome")
    error = str(over.get("error") or "")
    if outcome != "raised":
        return False, (
            f"the base did not fail on a 2048x2048 tile, it reported {outcome!r}. "
            "Without a base failure there is nothing for the head to fix"
        )
    # Any exception would satisfy "raised"; only Triton's numel refusal is the
    # defect this change is about. A ROCm OOM or a missing symbol dressed up as
    # the defect would make the head's decline look like a fix for it.
    if not any(m in error.lower() for m in _NUMEL_MARKERS):
        return False, (
            f"the base raised, but not Triton's tile limit: {error[:300]}. "
            "That is a different failure and this run cannot speak to it"
        )
    return True, f"base refuses the oversized tile as expected: {error[:300]}"


def head_is_fixed(head: dict) -> tuple[bool, str]:
    over = head.get("oversized") or {}
    outcome = over.get("outcome")
    if outcome != "declined":
        return False, (
            f"the head reported {outcome!r} for the oversized tile, expected it to "
            f"decline so the caller's vectorized fallback takes over. "
            f"{over.get('error', '')[:300]}"
        )
    legal = (head.get("legal") or {}).get("outcome")
    if legal != "returned":
        return False, (
            f"the head declines the oversized tile but also no longer takes a legal "
            f"1024x1024 one ({legal!r}), which would be a fix that costs the fast path"
        )
    return True, (
        "head declines the 2048x2048 tile and still takes the 1024x1024 one, so the "
        f"guard fires exactly at Triton's cap as read on this machine "
        f"({head.get('module_limit')})"
    )
