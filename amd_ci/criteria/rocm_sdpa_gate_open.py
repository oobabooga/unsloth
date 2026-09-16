#!/usr/bin/env python3
"""Criteria: does this checkout open ROCm's fused SDPA gate, and does that change anything?

unsloth#8819 (finetuning falls back to the quadratic MATH path on ROCm), #8225 (the same
gate on gfx1200 diffusion) and #9404 (`diffusion.attention.math_only ... kernels=math` on a
Radeon 8060S) all reduce to one question: with only `import unsloth` run, is
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL set, and does torch then hand out a sub-quadratic
attention kernel.

The defect at the base is "gate unset AND attention is math-only". The head is fixed when
the gate is set and a sub-quadratic backend becomes usable.

The control readings decide whether the question is even answerable on this host: if
forcing the gate to "1" changes nothing relative to "0", then the flag is not load-bearing
on gfx1151 with this torch build and no state of this PR can be shown to fix anything here.
That is reported as a failed gate, which lands the run on INCONCLUSIVE rather than letting
an unchanged reading read as a pass.

Pairs with probes/rocm_sdpa_gate_probe.py.
"""

from __future__ import annotations

GATE = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
TITLE = "ROCm fused SDPA gate, base versus head"
MODE = "differential"
NEEDS = ["gpu", "rocm", "gfx1200_rdna4", "wsl", "multi_gpu", "nvidia", "mig", "xpu", "mlx",
         "windows"]

_SUBQUADRATIC = ("flash", "mem_efficient", "cudnn")


def _shipped(state: dict) -> dict:
    return state.get("as_shipped") or {}


def _fused(reading: dict) -> list:
    return [b for b in (reading.get("available_backends") or []) if b in _SUBQUADRATIC]


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    states = {n: v for n, v in obs.items() if not n.startswith("_")}

    for name, v in states.items():
        s = _shipped(v)
        ran = bool(s) and not s.get("child_failed") and s.get("torch_error") is None
        detail = s.get("torch_error") or s.get("stderr_tail") or f"torch {s.get('torch_version')}"
        out.append((f"{name}: torch loaded in the probe child", ran, str(detail)[:300]))
        out.append((f"{name}: ROCm GPU visible", bool(s.get("cuda_available")),
                    f"arch={s.get('arch')} device={s.get('device_name')} hip={s.get('torch_hip')}"))

    # Non-vacuity: is the flag load-bearing on THIS chip and THIS torch build?
    for name, v in states.items():
        c0, c1 = v.get("control_gate_0") or {}, v.get("control_gate_1") or {}
        f0, f1 = _fused(c0), _fused(c1)
        moved = f0 != f1
        out.append((f"{name}: forcing the gate to 1 changes the usable backends",
                    moved,
                    f"gate=0 -> {f0 or 'math only'}; gate=1 -> {f1 or 'math only'}"))
        break  # one control pair is enough; they are state-independent by construction
    return out


def table(obs: dict) -> str:
    rows = ["| state | gate after `import unsloth` | unsloth imported | usable backends | "
            "math peak GiB | fused peak GiB | studio math_only |",
            "|---|---|---|---|---|---|---|"]
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        s = _shipped(v)
        usable = s.get("sdpa_usable") or {}
        math_peak = (usable.get("math") or {}).get("peak_gib")
        fused = _fused(s)
        fused_peak = (usable.get(fused[0]) or {}).get("peak_gib") if fused else None
        rows.append(
            f"| {name} | `{s.get('gate_after_import')}` | {s.get('unsloth_imported')} | "
            f"{', '.join(s.get('available_backends') or []) or 'none'} | "
            f"{math_peak if math_peak is not None else '-'} | "
            f"{fused_peak if fused_peak is not None else '-'} | "
            f"{s.get('studio_sdpa_math_only')} |")
    rows.append("")
    for name, v in obs.items():
        if name.startswith("_"):
            continue
        c0, c1 = v.get("control_gate_0") or {}, v.get("control_gate_1") or {}
        rows.append(f"Control on {name}: gate forced to 0 -> "
                    f"{', '.join(c0.get('available_backends') or []) or 'none'}; "
                    f"forced to 1 -> {', '.join(c1.get('available_backends') or []) or 'none'}.")
        break
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    s = _shipped(base)
    gate = s.get("gate_after_import")
    fused = _fused(s)
    math_only = s.get("studio_sdpa_math_only")
    if gate == "1":
        return False, (f"the base already sets {GATE}={gate!r} on `import unsloth`, so there is "
                       "no gate for this PR to open here")
    if fused and not math_only:
        return False, (f"the base already hands out {', '.join(fused)} without the gate, so the "
                       "quadratic MATH fallback #8819 reports is not reproduced on this host")
    return True, (f"{GATE} is {gate!r} after `import unsloth` and the only usable SDPA backend is "
                  f"{', '.join(s.get('available_backends') or []) or 'none'} "
                  f"(studio sdpa_math_only={math_only})")


def head_is_fixed(head: dict) -> tuple[bool, str]:
    s = _shipped(head)
    gate = s.get("gate_after_import")
    fused = _fused(s)
    if gate != "1":
        return False, f"{GATE} is still {gate!r} after `import unsloth` at the head"
    if not fused:
        return False, (f"the head sets {GATE}=1 but torch still offers no sub-quadratic kernel "
                       f"(usable: {', '.join(s.get('available_backends') or []) or 'none'}), so "
                       "the gate is open and the fallback remains")
    return True, (f"{GATE}=1 after `import unsloth` and {', '.join(fused)} is usable "
                  f"(studio sdpa_math_only={s.get('studio_sdpa_math_only')})")
