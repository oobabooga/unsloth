#!/usr/bin/env python3
"""Criteria: dequant kernels round to 16 bits exactly like torch on gfx1151 (base vs head).

One module, two names: hip_rne_zoo.py (MXFP4, unsloth-zoo) and hip_rne_fp8.py (FP8 block dequant,
unsloth). The kind is the file name.

Differential mode.
Defect at the base (default arm):
  zoo  the fused MXFP4 kernel is not bit-equal to mxfp4_dequantize_torch for bf16 or fp16 on the
       every-byte x every-scale data or the self-check data, and mxfp4_kernel_available() is False
       for at least one dtype
  fp8  the byte decoder or the fp8e4nv kernel is not bit-equal to the torch loop for fp16 output
Fixed at the head when all of:
  * default arm: 0 mismatches (NaN payloads excepted) in every parity cell; fp8: the fp8e4nv kernel
    really ran (supported on this Triton), and bf16 output stays bit-equal; zoo: available for
    bf16 and fp16, both layouts
  * kill switch arm (exact rounding forced off): mismatches come back, so the head's exact path
    is what the default arm measured (not some other change)
  * head pytest: no failing test, and the real-device bit-equality tests passed (not skipped)
Timing (kernel vs torch path; head vs base) is reported, not gated, except zoo: the head product
route must be <= 0.33x the head kill-switch product route (the kernel is used, not the torch path).
"""

from __future__ import annotations

from pathlib import Path

KIND = "fp8" if "fp8" in Path(__file__).stem else "zoo"
TITLE = {"zoo": "MXFP4 dequant kernel 16-bit rounding on gfx1151, base vs head (unsloth-zoo)",
         "fp8": "FP8 block dequant kernels fp16 rounding on gfx1151, base vs head (unsloth)"}[KIND]
MODE = "differential"
NEEDS = ["rocm", "gpu", "nvidia"]
ARMS = ("default", "switch_off")
REQUIRED = {"zoo": ("test_rocm_self_check_passes", "test_kernel_is_bit_identical_for_every_nibble_and_scale",
                    "test_fused_kernel_is_bit_identical"),
            "fp8": ("test_dequant_kernels_match_torch_loop_fp16",)}[KIND]


def _arm(o, arm):
    return ((o or {}).get("arms") or {}).get(arm) or {}


def _done(a):
    return a.get("stage") == "done" and a.get("ok") is True


def _cells(a, fp16_only = False):
    out = {}
    for k, v in (a.get("parity") or {}).items():
        if fp16_only and "float16" not in k.split("/"):
            continue
        out[k] = v
    return out


def _bad_cells(cells):
    return {k: (v.get("error") or v.get("mismatch")) for k, v in cells.items()
            if "skipped" not in v and (v.get("error") or v.get("mismatch"))}


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        for arm in ARMS:
            a = _arm(o, arm)
            f = str(a.get("module_file") or "")
            out.append((f"{name}/{arm} imported the state's module", bool(f) and f.startswith(ck),
                        f or f"stage={a.get('stage')} rc={a.get('rc')} {a.get('error') or ''}"[:300]))
            out.append((f"{name}/{arm} finished", _done(a), (a.get("error") or f"stage={a.get('stage')} rc={a.get('rc')}")[:300]))
    a = _arm(obs.get("base"), "default")
    out.append(("ROCm torch on gfx1151", bool(a.get("hip")) and str(a.get("arch") or "").startswith("gfx1151"),
                f"hip={a.get('hip')} arch={a.get('arch')} gpu={a.get('gpu')} torch={a.get('torch')} triton={a.get('triton')}"))
    p = (obs.get("head") or {}).get("pytest") or {}
    out.append(("head pytest ran", p.get("rc") in (0, 1), f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:]}"))
    out.append(("head pytest collected", (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    return out


def base_shows_defect(base):
    a = _arm(base, "default")
    if not _done(a):
        return False, "base default arm did not finish"
    if KIND == "zoo":
        bad = _bad_cells(_cells(a))
        unavailable = [d for d, v in (a.get("available") or {}).items() if not all(v.values())]
        return bool(bad) and bool(unavailable), f"mismatching cells {bad}; kernel unavailable for {unavailable}"
    bad = _bad_cells(_cells(a, fp16_only = True))
    return bool(bad), f"fp16 mismatching cells {bad}"


def _ms(a, *path):
    d = a
    for p in path:
        d = (d or {}).get(p) or {}
    return d.get("median_ms") if isinstance(d, dict) else None


def head_is_fixed(head, base = None):
    problems = []
    a, k = _arm(head, "default"), _arm(head, "switch_off")
    if not (_done(a) and _done(k)):
        return False, "a head arm did not finish"
    bad = _bad_cells(_cells(a))
    if bad:
        problems.append(f"default arm mismatches {bad}")
    if KIND == "fp8":
        if not a.get("fp8e4nv_supported"):
            problems.append("fp8e4nv kernel not supported here, so half the fix is unmeasured")
        if not _bad_cells(_cells(k, fp16_only = True)):
            problems.append("kill switch arm shows no fp16 mismatch: the exact path was not what changed the result")
    else:
        unavailable = [d for d, v in (a.get("available") or {}).items() if not all(v.values())]
        if unavailable or len(a.get("available") or {}) < 2:
            problems.append(f"kernel unavailable for {unavailable or 'unknown'}")
        if not _bad_cells(_cells(k)):
            problems.append("kill switch arm shows no mismatch: the exact path was not what changed the result")
        for dt in ("bfloat16", "float16"):
            p, kp = _ms(a, "timing_gate_up_T", dt, "product"), _ms(k, "timing_gate_up_T", dt, "product")
            if not (p and kp and p <= 0.33 * kp):
                problems.append(f"{dt}: product route {p} ms not <= 0.33x kill switch {kp} ms")
    pt = (head or {}).get("pytest") or {}
    failed = set(pt.get("failed", [])) | set(pt.get("errors", []))
    if failed:
        problems.append(f"head pytest failures {sorted(failed)[:8]}")
    passed = pt.get("passed", [])
    for req in REQUIRED:
        if not any(f"::{req}[" in t or t.endswith(f"::{req}") for t in passed):
            problems.append(f"required test {req} did not pass (skipped or absent)")
    if problems:
        return False, "; ".join(problems)
    return True, "kernels bit-equal to torch in every cell, kill switch brings the mismatches back, required tests passed"


def table(obs):
    def f(v):
        return "n/a" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))

    rows = ["| state | arm | cell | elements | mismatches | classes | NaN payload diffs | first examples |", "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for arm in ARMS:
            a = _arm(obs.get(name), arm)
            for key, v in (a.get("parity") or {}).items():
                rows.append(f"| {name} | {arm} | {key} | {v.get('n')} | {v.get('mismatch', v.get('error') or v.get('skipped'))} | "
                            f"{v.get('classes', '')} | {v.get('nan_bits_differ', '')} | {(v.get('examples') or [])[:3]} |")
    rows += ["", "| state | arm | availability / flags |", "|---|---|---|"]
    for name in ("base", "head"):
        for arm in ARMS:
            a = _arm(obs.get(name), arm)
            flags = {x: a.get(x) for x in ("available", "exact_round", "is_hip_flag", "exact_f16", "exact_flag", "fp8e4nv_supported") if x in a}
            rows.append(f"| {name} | {arm} | {flags} |")
    rows += ["", "| state | arm | dtype | timing (median ms per call) |", "|---|---|---|---|"]
    tkey = "timing_gate_up_T" if KIND == "zoo" else "timing_4096x14336"
    for name in ("base", "head"):
        for arm in ARMS:
            a = _arm(obs.get(name), arm)
            for dt, arms in (a.get(tkey) or {}).items():
                cells = ", ".join(k + "=" + f(v.get("median_ms")) for k, v in arms.items())
                rows.append(f"| {name} | {arm} | {dt} | {cells} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest") or {}
        bad = ", ".join("`" + t.split("::")[-1] + "`" for t in sorted(set(p.get("failed", [])) | set(p.get("errors", []))))
        rows.append(f"| {name} | {p.get('n_passed')} | {p.get('n_failed')} | {p.get('n_skipped')} | {bad} |")
    a = _arm(obs.get("base"), "default")
    rows += ["", f"Versions: torch={a.get('torch')} hip={a.get('hip')} triton={a.get('triton')} gpu={a.get('gpu')} arch={a.get('arch')}"]
    return "\n".join(rows)
