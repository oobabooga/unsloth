#!/usr/bin/env python3
"""Criteria: ROCm MXFP4 dequant (gpt-oss experts) takes the fused Triton kernel instead of the torch path.

Differential mode.
Defect at the base (default arm): on gfx1151 the fused kernel's bit-exact self-check fails for
bf16 (mxfp4_kernel_available() False), mxfp4_dequantize routes every expert stack to
mxfp4_dequantize_torch (0 kernel calls), and that product route is >= 3x slower per call than
the forced kernel (or, if the kernel cannot run, than torch.compile of the plain dequant) on the
gate_up stack, both layouts.

Fixed at the head when all of:
  * default arm: bf16 self-check passes for both layouts, every stack routes to the kernel
    (0 torch-path calls), output bit-equal to the head's torch reference AND expert 0 bit-equal
    to an fp64 CPU truth rounded once (so the kernel is not merely agreeing with a wrong reference)
  * default arm: the device torch reference has 0 mismatches vs the fp64 truth on the self-check data
  * kill switch arm (UNSLOTH_MXFP4_HIP_EXACT_SCALE=0): routes back to the torch path, as at base
  * per call: product route <= 0.33x the kill-switch product route on every stack / layout
  * head pytest: no failure that the base does not also have; the real-ROCm tests
    test_rocm_uses_the_fused_kernel[*] passed (not skipped)

Gates (non-vacuity): each arm imported its own state's unsloth_zoo; ROCm torch on gfx1151; Triton
present; every arm finished; A/A: the forced torch path at base and the kill-switch arm at head
(same code) agree within 15% on every stack; head pytest ran and collected tests.

The self-check diagnosis (mismatches by e8m0 scale byte for kernel / reference / forced reference
formulations vs the fp64 truth) is reported in the table whatever the verdict: if the head fails
because the KERNEL disagrees with the truth, the cause is in the kernel, not the reference.
"""

from __future__ import annotations

TITLE = "MXFP4 dequant on gfx1151: fused Triton kernel vs torch path, base vs head"
MODE = "differential"
# nvidia: routing there must not change; unreachable on this pool (covered by the local suite).
NEEDS = ["rocm", "gpu", "nvidia"]

ARMS = ("default", "switch_off")
KEYS = ("gate_up/T", "gate_up/plain", "down/T", "down/plain")
REQUIRED_HEAD_TESTS = ("test_rocm_uses_the_fused_kernel",)


def _arm(o, arm):
    return ((o or {}).get("arms") or {}).get(arm) or {}


def _done(a):
    return a.get("stage") == "done" and a.get("ok") is True


def _km(a, key, which):
    return (((a.get("kernel_ms") or {}).get(key) or {}).get(which) or {}).get("median_ms")


def _bad(p):
    return set((p or {}).get("failed", [])) | set((p or {}).get("errors", []))


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        for arm in ARMS:
            a = _arm(o, arm)
            f = str(a.get("zoo_file") or "")
            out.append((f"{name}/{arm} imported the state's unsloth_zoo", bool(f) and f.startswith(ck),
                        f or f"stage={a.get('stage')} rc={a.get('rc')} {a.get('error') or ''}"[:300]))
            out.append((f"{name}/{arm} finished", _done(a), a.get("error") or f"stage={a.get('stage')} rc={a.get('rc')}"))
    a = _arm(obs.get("base"), "default")
    out.append(("ROCm torch on gfx1151", bool(a.get("hip")) and str(a.get("arch") or "").startswith("gfx1151"),
                f"hip={a.get('hip')} arch={a.get('arch')} gpu={a.get('gpu')} torch={a.get('torch')}"))
    out.append(("Triton importable", a.get("has_triton") is True, f"triton={a.get('triton')}"))
    k = _arm(obs.get("head"), "switch_off")
    spreads = {}
    for key in KEYS:
        b, h = _km(a, key, "torch"), _km(k, key, "torch")
        if b and h:
            spreads[key] = round(abs(h / b - 1), 3)
    out.append(("A/A: torch path base vs head kill switch within 15%",
                len(spreads) == len(KEYS) and max(spreads.values()) <= 0.15, f"spreads={spreads}"))
    p = (obs.get("head") or {}).get("pytest") or {}
    out.append(("head pytest ran", p.get("rc") in (0, 1), f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:]}"))
    out.append(("head pytest collected", (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    return out


def table(obs):
    def f(v):
        return "n/a" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))

    def mm(d):
        return "n/a" if not d else (f"{d.get('total')} {d.get('by_scale')}" if d.get("total") else "0")

    rows = ["| state | arm | exact-scale flag | dtype | available (T) | verified (plain) | kernel failed | warnings | kernel vs fp64 | reference vs fp64 | ldexp ref vs fp64 | ldexp fp32 ref vs fp64 | bits ref vs fp64 | kernel error |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for arm in ARMS:
            a = _arm(obs.get(name), arm)
            for dt, d in (a.get("selfcheck") or {}).items():
                rows.append(f"| {name} | {arm} | {a.get('hip_exact_scale')} | {dt} | {d.get('available_transposed')} | "
                            f"{d.get('verified_plain')} | {d.get('kernel_failed')} | {'; '.join(d.get('warnings') or []) or 'none'} | "
                            f"{mm(d.get('kernel_vs_truth'))} | {mm(d.get('reference_vs_truth'))} | {mm(d.get('reference_ldexp_vs_truth'))} | "
                            f"{mm(d.get('reference_ldexp_fp32_vs_truth'))} | {mm(d.get('reference_exact_bits_vs_truth'))} | {d.get('kernel_error') or ''} |")
    rows += ["", "| state | arm | stack | route (kernel / torch calls) | = reference | expert 0 = fp64 | product ms | torch ms | kernel ms | compiled plain ms | product / kernel |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for arm in ARMS:
            a = _arm(obs.get(name), arm)
            for key in KEYS:
                c = (a.get("route") or {}).get(key) or {}
                pr = (a.get("parity") or {}).get(key) or {}
                p, t = _km(a, key, "product"), _km(a, key, "kernel")
                rows.append(f"| {name} | {arm} | {key} | {c.get('kernel')} / {c.get('torch')} | {pr.get('equal_reference')} | "
                            f"{pr.get('expert0_equal_fp64_truth')} | {f(p)} | {f(_km(a, key, 'torch'))} | {f(t)} | "
                            f"{f(_km(a, key, 'compiled_plain'))} | {f(p / t) if p and t else 'n/a'} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest")
        if not p:
            continue
        bad = ", ".join(f"`{t.split('::')[-1]}`" for t in sorted(_bad(p))) or "none"
        rows.append(f"| {name} | {p.get('n_passed', 0)} | {p.get('n_failed', 0)} | {p.get('n_skipped', 0)} | {bad} |")
        if p.get("skipped"):
            reasons = {}
            for t, m in p["skipped"].items():
                reasons.setdefault(m, []).append(t)
            rows.append(f"| {name} skips | " + "; ".join(f"{len(v)} x {k!r}" for k, v in reasons.items()) + " | | | |")
    a = _arm(obs.get("base"), "default")
    rows += ["", f"Versions: torch={a.get('torch')} hip={a.get('hip')} triton={a.get('triton')} gpu={a.get('gpu')} arch={a.get('arch')}"]
    return "\n".join(rows)


# differential.py calls base_shows_defect before head_is_fixed; the head's pytest is compared to it.
_SEEN: dict = {}


def base_shows_defect(base):
    _SEEN["base"] = base
    a = _arm(base, "default")
    if not _done(a):
        return False, "base default arm did not finish"
    bf = ((a.get("selfcheck") or {}).get("bfloat16") or {})
    unavailable = bf.get("available_transposed") is False
    route = a.get("route") or {}
    torch_route = all(((route.get(k) or {}).get("kernel", 0) == 0 and (route.get(k) or {}).get("torch", 0) > 0) for k in KEYS)
    slower = {}
    for k in ("gate_up/T", "gate_up/plain"):
        p = _km(a, k, "product")
        ref = _km(a, k, "kernel") or _km(a, k, "compiled_plain")
        slower[k] = bool(p and ref and p / ref >= 3)
    ok = unavailable and torch_route and all(slower.values())
    return ok, f"bf16 kernel unavailable={unavailable}; torch route on every stack={torch_route}; product >= 3x slower than one pass {slower}"


def head_is_fixed(head, base = None):
    base = base if base is not None else _SEEN.get("base")
    problems = []
    a, k = _arm(head, "default"), _arm(head, "switch_off")
    if not _done(a) or not _done(k):
        return False, "a head arm did not finish"
    bf = (a.get("selfcheck") or {}).get("bfloat16") or {}
    if not (bf.get("available_transposed") and bf.get("verified_plain")):
        problems.append(f"bf16 self-check failed: {bf.get('warnings')} kernel_vs_truth={bf.get('kernel_vs_truth')}")
    for dt, d in (a.get("selfcheck") or {}).items():
        if (d.get("reference_vs_truth") or {}).get("total") != 0:
            problems.append(f"{dt} device reference vs fp64: {d.get('reference_vs_truth')}")
    for key in KEYS:
        c = (a.get("route") or {}).get(key) or {}
        if c.get("kernel", 0) == 0 or c.get("torch", 0) != 0:
            problems.append(f"default/{key} route {c}")
        pr = (a.get("parity") or {}).get(key) or {}
        if not (pr.get("equal_reference") and pr.get("expert0_equal_fp64_truth")):
            problems.append(f"default/{key} parity {pr}")
        ck = (k.get("route") or {}).get(key) or {}
        if ck.get("kernel", 0) != 0 or ck.get("torch", 0) == 0:
            problems.append(f"switch_off/{key} route {ck}")
        p, s = _km(a, key, "product"), _km(k, key, "product")
        if not (p and s and p / s <= 0.33):
            problems.append(f"{key} product/kill-switch={p}/{s}")
    hp = head.get("pytest") or {}
    new_bad = sorted(_bad(hp) - _bad((base or {}).get("pytest")))
    if new_bad:
        problems.append("failing at the head only: " + ", ".join(f"`{t.split('::')[-1]}`" for t in new_bad))
    for req in REQUIRED_HEAD_TESTS:
        got = [t for t in hp.get("passed", []) if t.split("::")[-1].startswith(req)]
        if not got:
            problems.append(f"{req} did not pass at the head (skipped: {[t for t in (hp.get('skipped') or {}) if req in t]})")
    if problems:
        return False, "; ".join(problems)
    return True, "fused kernel routed and verified, bit-equal to reference and fp64 truth, kill switch restores the torch path, >= 3x faster"
