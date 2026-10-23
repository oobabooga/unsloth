#!/usr/bin/env python3
"""Criteria: on gfx1151, block-FP8 (e4m3fn) weights the torch loop used to dequantize take the bit-exact byte decoder.

Differential mode.
Defect at the base (default arm): ragged / rectangular-block e4m3fn weights with bf16 output go to the chunked
torch loop, >= 3x slower per call than the byte decoder _e4m3fn_bytes_dequant on the ragged shape (measured 7.5x).

Fixed at the head when all of:
  * default arm: ragged/bfloat16 and rect_block/bfloat16 route to the byte decoder only, bit-equal to the torch
    loop, reruns identical
  * default arm: every other shape / dtype (even tiles, fp16 output) keeps the base's route
  * kill switch arm (UNSLOTH_FP8_HIP_BYTES_DEQUANT=0): routes equal to the base's on every shape
  * per call: head default / head kill switch product time <= 0.33 on ragged, <= 1.15 on q, up, down (same route)
  * head pytest: no failure the base does not also have; the real-ROCm test and the ROCm routing table passed

Gates (non-vacuity): each arm imported its own state's unsloth and finished; ROCm torch on gfx1151;
fp8e4nv compiles there (Triton >= 3.5) so the fp8e4nv arm is real; the byte decoder exists at the
base; A/A: the same route timed in two processes (base default vs head kill switch, and base default
vs base kill switch) within 15% on `up`; head pytest ran and collected tests.
"""

from __future__ import annotations

TITLE = "Block-FP8 weight dequant on gfx1151: byte decoder vs emulated fp8e4nv kernel / torch loop, base vs head"
MODE = "differential"
# nvidia: routing there must not change; unreachable on this pool (covered by the local suite).
NEEDS = ["rocm", "gpu", "nvidia"]

ARMS = ("default", "switch_off")
TIMED = ("q", "up", "down", "ragged")
LOOP_CASES = ("ragged/bfloat16", "rect_block/bfloat16")
REQUIRED_PASS = ("test_real_rocm_device_takes_bytes_kernel_bit_exact", "test_rocm_routes")


def _arm(o, arm):
    return ((o or {}).get("arms") or {}).get(arm) or {}


def _done(a):
    return a.get("stage") == "done" and a.get("ok") is True


def _km(a, shape, which):
    return (((a.get("kernel_ms") or {}).get(shape) or {}).get(which) or {}).get("median_ms")


def _ratio(x, y):
    return x / y if x and y else None


def _bad(p):
    return set((p or {}).get("failed", [])) | set((p or {}).get("errors", []))


# differential.py calls head_is_fixed(head) alone; the head checks that compare against the base
# (kill-switch routes, pytest failures the base does not share) read it from here. gates() runs first.
_OBS: dict = {}


def gates(obs):
    _OBS.clear()
    _OBS.update(obs or {})
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
    out.append(("fp8e4nv compiles here (the emulated fp8 kernel arm is real)", a.get("fp8e4nv_unsupported") is False,
                f"fp8e4nv_unsupported={a.get('fp8e4nv_unsupported')} triton={a.get('triton')}"))
    out.append(("byte decoder present at base", a.get("has_bytes_kernel") is True, f"has_bytes_kernel={a.get('has_bytes_kernel')}"))
    h = _arm(obs.get("head"), "default")
    out.append(("head reads the kill switch", h.get("hip_bytes_flag") is True
                and _arm(obs.get("head"), "switch_off").get("hip_bytes_flag") is False,
                f"default={h.get('hip_bytes_flag')} switch_off={_arm(obs.get('head'), 'switch_off').get('hip_bytes_flag')}"))
    for label, x in (("head/switch_off", _arm(obs.get("head"), "switch_off")), ("base/switch_off", _arm(obs.get("base"), "switch_off"))):
        r = _ratio(_km(x, "up", "product"), _km(a, "up", "product"))
        out.append((f"A/A: base/default vs {label} product on `up` within 15% (same route)",
                    r is not None and abs(r - 1) <= 0.15, f"ratio={r}"))
    p = (obs.get("head") or {}).get("pytest") or {}
    out.append(("head pytest ran", p.get("rc") in (0, 1), f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:]}"))
    out.append(("head pytest collected", (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    return out


def table(obs):
    f = lambda v: "n/a" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
    rows = ["| state | arm | flag | shape | route bytes / fp8e4nv / torch | = torch loop | = fp8e4nv kernel | bytes = torch loop |",
            "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for arm in ARMS:
            a = _arm(obs.get(name), arm)
            for k, c in (a.get("route") or {}).items():
                pr = (a.get("parity") or {}).get(k) or {}
                rows.append(f"| {name} | {arm} | {a.get('hip_bytes_flag')} | {k} | {c.get('bytes')} / {c.get('triton_fp8')} / {c.get('torch')} "
                            f"| {pr.get('equal_torch_loop')} | {pr.get('equal_triton_fp8', 'n/a')} | {pr.get('bytes_equal_torch_loop')} |")
    rows += ["", "| state | arm | shape | product ms | bytes ms | fp8e4nv ms | torch loop ms | compiled one pass ms | product / one pass |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for arm in ARMS:
            a = _arm(obs.get(name), arm)
            for s in TIMED:
                p, c = _km(a, s, "product"), _km(a, s, "compiled_one_pass")
                rows.append(f"| {name} | {arm} | {s} | {f(p)} | {f(_km(a, s, 'bytes'))} | {f(_km(a, s, 'triton_fp8'))} "
                            f"| {f(_km(a, s, 'torch_loop'))} | {f(c)} | {f(_ratio(p, c))} |")
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


def base_shows_defect(base):
    a = _arm(base, "default")
    if not _done(a):
        return False, "base default arm did not finish"
    route = a.get("route") or {}
    loop = all(((route.get(k) or {}).get("torch", 0) > 0 and (route.get(k) or {}).get("bytes", 0) == 0)
               for k in LOOP_CASES)
    r = _ratio(_km(a, "ragged", "product"), _km(a, "ragged", "bytes"))
    slow = r is not None and r >= 3.0
    return loop and slow, (f"ragged / rect bf16 take the torch loop={loop}; product / byte decoder on ragged = {r} (>= 3 needed)")


def head_is_fixed(head, base = None):
    base = base if base is not None else _OBS.get("base")
    if not base:
        return False, "no base observation to compare the head against"
    problems = []
    a, k = _arm(head, "default"), _arm(head, "switch_off")
    if not _done(a) or not _done(k):
        return False, "a head arm did not finish"
    base_route = _arm(base, "default").get("route") or {}
    for key, c in (a.get("route") or {}).items():
        if key in LOOP_CASES:
            if not (c.get("bytes", 0) > 0 and c.get("triton_fp8", 0) == 0 and c.get("torch", 0) == 0):
                problems.append(f"default/{key} route {c}")
            pr = (a.get("parity") or {}).get(key) or {}
            if not (pr.get("equal_torch_loop") and pr.get("repeat_equal")):
                problems.append(f"default/{key} parity {pr}")
        elif c != base_route.get(key):
            problems.append(f"default/{key} route {c} != base {base_route.get(key)} (should be unchanged)")
        ck = (k.get("route") or {}).get(key) or {}
        if ck != base_route.get(key):
            problems.append(f"switch_off/{key} route {ck} != base {base_route.get(key)}")
    if not a.get("route"):
        problems.append("no routes recorded")
    for s_ in TIMED:
        r = _ratio(_km(a, s_, "product"), _km(k, s_, "product"))
        lim = 0.33 if s_ == "ragged" else 1.15
        if r is None or r > lim:
            problems.append(f"{s_}: head default / kill switch product = {r} (limit {lim})")
    hp = head.get("pytest") or {}
    new_bad = sorted(_bad(hp) - _bad(base.get("pytest")))
    if new_bad:
        problems.append("failing at the head only: " + ", ".join(f"`{t.split('::')[-1]}`" for t in new_bad))
    passed = hp.get("passed", [])
    for t in REQUIRED_PASS:
        if not any(t in p for p in passed) or any(t in s for s in (hp.get("skipped") or {})):
            problems.append(f"`{t}` did not pass at the head (or was skipped)")
    if problems:
        return False, "; ".join(problems)
    return True, "torch-loop cases take the byte decoder bit-exactly, other routes unchanged, kill switch restores the base, faster per call, no new test failures"
