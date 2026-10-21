#!/usr/bin/env python3
"""Criteria: ROCm batch-1 NF4 decode takes Unsloth's Triton GEMV instead of bitsandbytes' ctypes GEMV.

Differential mode.
Defect at the base (default arm): eager batch-1 decode on gfx1151 routes every decode shape
through bitsandbytes' ctypes GEMV (no Triton GEMV call), while the Triton GEMV is measurably
faster per call on the Llama 8B MLP shapes (ctypes route / forced Triton >= 1.5x on up, down,
qwen_up), i.e. the slow kernel is chosen.

Fixed at the head when all of:
  * default arm: every shape routes to the Triton GEMV (0 ctypes calls)
  * default arm: output finite, within bf16 tolerance (1.2e-2 relative) of an fp64
    dequantize + matmul reference, no worse than bitsandbytes' GEMV + 4e-3, reruns identical
  * kill switch arm (UNSLOTH_TRITON_GEMV_EAGER=0): every shape routes back to bitsandbytes
  * per-call: product route / ctypes <= 0.7 on up, down, qwen_up
  * decode loop per token (host overhead included): Llama 3 8B shapes product <= 0.85x ctypes,
    Llama 3.2 1B shapes product <= 1.05x ctypes (no host-bound regression)
  * head pytest: 0 failures

Gates (non-vacuity): each arm imported its own state's unsloth; ROCm torch on gfx1151 with
Triton >= 3.6 (the version the head enables); every arm finished; decode-loop A/A noise at the
base (product route == ctypes route there) within 10%; head pytest ran and collected tests.
"""

from __future__ import annotations

TITLE = "NF4 batch-1 decode on gfx1151: Triton GEMV vs bitsandbytes ctypes GEMV, base vs head"
MODE = "differential"
# nvidia: routing there must not change; unreachable on this pool (covered by the local suite).
NEEDS = ["rocm", "gpu", "nvidia"]

ARMS = ("default", "switch_off")
DECODE_SHAPES = ("q", "kv", "up", "down", "qwen_up", "tiny")
BIG = ("up", "down", "qwen_up")
TOL = 1.2e-2


def _arm(o, arm):
    return ((o or {}).get("arms") or {}).get(arm) or {}


def _done(a):
    return a.get("stage") == "done" and a.get("ok") is True


def _km(a, shape, which):
    return (((a.get("kernel_ms") or {}).get(shape) or {}).get(which) or {}).get("median_ms")


def _dm(a, model, which):
    return (((a.get("decode") or {}).get(model) or {}).get(which) or {}).get("median_ms_per_token")


def _tri(c):
    return (c or {}).get("triton_planned", 0) + (c or {}).get("triton_gemv", 0)


def _ver(s):
    try:
        return tuple(int(x) for x in str(s).split(".")[:2])
    except ValueError:
        return (0, 0)


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ck = str(o.get("checkout") or "<none>")
        for arm in ARMS:
            a = _arm(o, arm)
            f = str(a.get("unsloth_file") or "")
            out.append((f"{name}/{arm} imported the state's unsloth", bool(f) and f.startswith(ck),
                        f or f"stage={a.get('stage')} rc={a.get('rc')} {a.get('error') or ''}"[:300]))
            out.append((f"{name}/{arm} finished", _done(a), a.get("error") or f"stage={a.get('stage')} rc={a.get('rc')}"))
    a = _arm(obs.get("base"), "default")
    out.append(("ROCm torch on gfx1151", bool(a.get("hip")) and str(a.get("arch") or "").startswith("gfx1151"),
                f"hip={a.get('hip')} arch={a.get('arch')} gpu={a.get('gpu')} torch={a.get('torch')}"))
    out.append(("Triton >= 3.6 (the head's HIP threshold)", _ver(a.get("triton")) >= (3, 6), f"triton={a.get('triton')}"))
    out.append(("NF4 Triton kernels loaded at base", a.get("use_nf4_kernels") is True, f"_USE_NF4_KERNELS={a.get('use_nf4_kernels')}"))
    spreads = []
    for model in ("llama3_8b_x8", "llama32_1b_x16"):
        p, c = _dm(a, model, "product"), _dm(a, model, "ctypes")
        if p and c:
            spreads.append(abs(p / c - 1))
    out.append(("decode-loop A/A at base (product route == ctypes there) within 10%",
                len(spreads) == 2 and max(spreads) <= 0.10, f"spreads={[round(s, 3) for s in spreads]}"))
    p = (obs.get("head") or {}).get("pytest") or {}
    out.append(("head pytest ran", p.get("rc") in (0, 1), f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:]}"))
    out.append(("head pytest collected", (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    return out


def table(obs):
    f = lambda v: "n/a" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
    rows = ["| state | arm | _TRITON_GEMV_EAGER | shape | route (ctypes / triton calls) | product rel err | ctypes rel err | product ms | ctypes ms | triton ms | bnb.matmul_4bit ms |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for arm in ARMS:
            a = _arm(obs.get(name), arm)
            for s in DECODE_SHAPES:
                c = (a.get("route") or {}).get(s) or {}
                pr = (a.get("parity") or {}).get(s) or {}
                rows.append(f"| {name} | {arm} | {a.get('triton_gemv_eager')} | {s} | {c.get('ctypes')} / {_tri(c)} "
                            f"| {f(pr.get('product_rel'))} | {f(pr.get('ctypes_rel'))} | {f(_km(a, s, 'product'))} "
                            f"| {f(_km(a, s, 'ctypes'))} | {f(_km(a, s, 'triton'))} | {f(_km(a, s, 'bnb_matmul_4bit'))} |")
    rows += ["", "| state | arm | decode loop | product ms/token | ctypes ms/token | triton ms/token | product / ctypes |", "|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for arm in ARMS:
            a = _arm(obs.get(name), arm)
            for m in ("llama3_8b_x8", "llama32_1b_x16"):
                p, c, t = _dm(a, m, "product"), _dm(a, m, "ctypes"), _dm(a, m, "triton")
                rows.append(f"| {name} | {arm} | {m} | {f(p)} | {f(c)} | {f(t)} | {f(p / c) if p and c else 'n/a'} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        p = (obs.get(name) or {}).get("pytest")
        if not p:
            continue
        bad = ", ".join(f"`{t.split('::')[-1]}`" for t in p.get("failed", []) + p.get("errors", [])) or "none"
        rows.append(f"| {name} | {p.get('n_passed', 0)} | {p.get('n_failed', 0)} | {p.get('n_skipped', 0)} | {bad} |")
        if p.get("skipped"):
            reasons = {}
            for t, m in p["skipped"].items():
                reasons.setdefault(m, []).append(t)
            rows.append(f"| {name} skips | " + "; ".join(f"{len(v)} x {k!r}" for k, v in reasons.items()) + " | | | |")
    a = _arm(obs.get("base"), "default")
    rows += ["", f"Versions: torch={a.get('torch')} hip={a.get('hip')} triton={a.get('triton')} "
                 f"bitsandbytes={a.get('bitsandbytes')} gpu={a.get('gpu')} arch={a.get('arch')}"]
    return "\n".join(rows)


def base_shows_defect(base):
    a = _arm(base, "default")
    if not _done(a):
        return False, "base default arm did not finish"
    route = a.get("route") or {}
    bnb_route = all(((route.get(s) or {}).get("ctypes", 0) > 0 and _tri(route.get(s)) == 0) for s in DECODE_SHAPES)
    slower = []
    for s in BIG:
        p, t = _km(a, s, "product"), _km(a, s, "triton")
        slower.append(bool(p and t and p / t >= 1.5))
    ok = bnb_route and all(slower)
    return ok, f"bitsandbytes route on every shape={bnb_route}; ctypes route >= 1.5x slower than Triton on {dict(zip(BIG, slower))}"


def head_is_fixed(head):
    problems = []
    a, k = _arm(head, "default"), _arm(head, "switch_off")
    if not _done(a) or not _done(k):
        return False, "a head arm did not finish"
    for s in DECODE_SHAPES:
        c = (a.get("route") or {}).get(s) or {}
        if c.get("ctypes", 0) != 0 or _tri(c) == 0:
            problems.append(f"default/{s} route {c}")
        pr = (a.get("parity") or {}).get(s) or {}
        rel, crel = pr.get("product_rel"), pr.get("ctypes_rel")
        if not (pr.get("finite") and pr.get("repeat_equal") and rel is not None and rel < TOL
                and crel is not None and rel <= crel + 4e-3):
            problems.append(f"default/{s} parity {pr}")
        ck = (k.get("route") or {}).get(s) or {}
        if ck.get("ctypes", 0) == 0 or _tri(ck) != 0:
            problems.append(f"switch_off/{s} route {ck}")
    for s in BIG:
        p, c = _km(a, s, "product"), _km(a, s, "ctypes")
        if not (p and c and p / c <= 0.7):
            problems.append(f"default/{s} per-call product/ctypes={p}/{c}")
    for m, lim in (("llama3_8b_x8", 0.85), ("llama32_1b_x16", 1.05)):
        p, c = _dm(a, m, "product"), _dm(a, m, "ctypes")
        if not (p and c and p / c <= lim):
            problems.append(f"decode {m} product/ctypes={p}/{c} (limit {lim})")
    hp = head.get("pytest") or {}
    bad = sorted(set(hp.get("failed", [])) | set(hp.get("errors", [])))
    if bad:
        problems.append("failing at the head: " + ", ".join(f"`{t.split('::')[-1]}`" for t in bad))
    if problems:
        return False, "; ".join(problems)
    return True, "Triton GEMV routed, parity held, kill switch restores bitsandbytes, faster per call and per token"
