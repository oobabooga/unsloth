#!/usr/bin/env python3
"""Criteria (#19768): on a real Windows gfx1151 APU whose RAM cap binds (host RAM held by the
fixture so Windows available RAM < HIP free), the base sizes the GPU at available RAM - 1024 and
the head adds exactly the unused BIOS carve-out its real registry + WDDM reader returns, never
past HIP free. The pool's carve-out size is whatever the box's BIOS has; it is recorded."""

from __future__ import annotations

TITLE = "Windows APU usable GPU memory: RAM cap + unused BIOS carve-out (real readers)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows", "integrated_gpu", "windows_rocm_wddm"]
TOL = 512


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_") and isinstance(v, dict)}


def gates(obs):
    out = []
    fx = obs.get("_fixture") or {}
    out.append(("fixture held host RAM", fx.get("status") == "READY", str(fx)[:200]))
    for n, o in _states(obs).items():
        out.append((f"{n}: Windows + ROCm torch on gfx1151",
                    o.get("platform") == "win32" and bool(o.get("hip"))
                    and str(o.get("arch") or "").startswith("gfx1151"),
                    f"torch={o.get('torch')} hip={o.get('hip')} arch={o.get('arch')} {o.get('torch_error') or ''}"))
        out.append((f"{n}: unified-memory branch reached (device 0 classified unified)",
                    0 in (o.get("unified_ids") or []) and not o.get("classifier_import_error"),
                    f"unified_ids={o.get('unified_ids')} {o.get('classifier_import_error') or ''}"))
        hip, avail = o.get("hip_free_mib"), o.get("host_avail_mib")
        out.append((f"{n}: RAM cap binds (HIP free > host available)",
                    hip is not None and avail is not None and hip > avail + 2048,
                    f"HIP free {hip} MiB, host avail {avail} MiB"))
        out.append((f"{n}: WDDM usage + registry carve-out readable",
                    o.get("wddm_used_mib") is not None and (o.get("registry_carve_out_mib") or 0) > 0,
                    f"carve-out {o.get('registry_carve_out_mib')} MiB, used {o.get('wddm_used_mib')} MiB"))
    return out


def table(obs):
    rows = ["| state | HIP free | host avail | carve-out | WDDM used | reader (cold s) | usable | probe s |",
            "|---|---|---|---|---|---|---|---|"]
    for n, o in _states(obs).items():
        rows.append(f"| {n} | {o.get('hip_free_mib')} | {o.get('host_avail_mib')}..{o.get('host_avail_after_mib')} "
                    f"| {o.get('registry_carve_out_mib')} | {o.get('wddm_used_mib')} "
                    f"| {o.get('reader_mib', '-')} ({o.get('reader_cold_s', '-')}) | {o.get('usable_mib')} "
                    f"| {o.get('probe_s')} |")
    return "\n".join(rows)


def _avail_band(o):
    a, b = o.get("host_avail_mib"), o.get("host_avail_after_mib")
    return (min(a, b if b is not None else a), max(a, b if b is not None else a)) if a is not None else (None, None)


def base_shows_defect(base):
    lo, hi = _avail_band(base)
    usable, c, u = base.get("usable_mib"), base.get("registry_carve_out_mib"), base.get("wddm_used_mib")
    if None in (usable, lo, c, u):
        return None, "missing reading"
    unused = max(0, c - u)
    capped = lo - 1024 - TOL <= usable <= hi - 1024 + TOL
    return capped and unused > 0, (f"usable {usable} MiB = host avail ({lo}..{hi}) - 1024; "
                                   f"the {unused} MiB unused carve-out is left out")


def head_is_fixed(head):
    lo, hi = _avail_band(head)
    usable, hip, reader = head.get("usable_mib"), head.get("hip_free_mib"), head.get("reader_mib")
    if None in (usable, lo, hip, reader):
        return None, "missing reading"
    want_lo, want_hi = min(hip, lo + reader) - 1024 - TOL, min(hip, hi + reader) - 1024 + TOL
    ok = want_lo <= usable <= want_hi and usable <= hip
    return ok, (f"usable {usable} MiB; expected min(HIP {hip}, avail {lo}..{hi} + reader {reader}) - 1024 "
                f"= {want_lo + TOL}..{want_hi - TOL}")
