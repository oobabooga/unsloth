#!/usr/bin/env python3
"""Criteria: ROCm block-FP8 (e4m3fn) weight dequant takes the bit-exact byte decoder on gfx1151.

Differential mode.
Defect at the base (default arm): on gfx1151 (no OCP fp8 hardware) weight_dequant never calls the
byte decoder _e4m3fn_bytes_dequant: even 128x128 tiles go to the software-emulated fp8e4nv Triton
kernel, ragged / rectangular blocks to the chunked torch loop, and the product route on the 14336 x
4096 weight is >= 3x slower than one torch.compile'd pass over it (the Triton audit measured 10.4x).

Fixed at the head when all of:
  * default arm: every shape / dtype routes to the byte decoder only (no fp8e4nv kernel, no torch loop)
  * default arm: output bit-equal (NaN positions included) to the torch loop and to the fp8e4nv
    Triton kernel it replaces, reruns identical
  * kill switch arm (UNSLOTH_FP8_HIP_BYTES_DEQUANT=0): routes equal to the base's on every shape
  * per call: head default / head kill-switch product time <= 0.5 on `up`, <= 1.0 on q, down, ragged
  * head pytest: no failure the base does not also have; the real-ROCm bit-exact test and the
    ROCm routing table tests passed (not skipped)

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
    no_bytes = bool(route) and all((c or {}).get("bytes", 0) == 0 for c in route.values())
    r = _ratio(_km(a, "up", "product"), _km(a, "up", "compiled_one_pass"))
    slow = r is not None and r >= 3.0
    return no_bytes and slow, (f"byte decoder never called={no_bytes}; product / one compiled pass on `up` = {r} (>= 3 needed); "
                               f"product / forced bytes = {_ratio(_km(a, 'up', 'product'), _km(a, 'up', 'bytes'))}")


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
        if not (c.get("bytes", 0) > 0 and c.get("triton_fp8", 0) == 0 and c.get("torch", 0) == 0):
            problems.append(f"default/{key} route {c}")
        pr = (a.get("parity") or {}).get(key) or {}
        if not (pr.get("equal_torch_loop") and pr.get("repeat_equal") and pr.get("equal_triton_fp8", True)):
            problems.append(f"default/{key} parity {pr}")
        ck = (k.get("route") or {}).get(key) or {}
        if ck != base_route.get(key):
            problems.append(f"switch_off/{key} route {ck} != base {base_route.get(key)}")
        if ck.get("bytes", 0) != 0:
            problems.append(f"switch_off/{key} still calls the byte decoder")
    if not a.get("route"):
        problems.append("no routes recorded")
    for s in TIMED:
        r = _ratio(_km(a, s, "product"), _km(k, s, "product"))
        lim = 0.5 if s == "up" else 1.0
        if r is None or r > lim:
            problems.append(f"{s}: head default / kill switch product = {r} (limit {lim})")
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
    return True, "byte decoder routed and bit-exact, kill switch restores the base routes, faster per call, no new test failures"
