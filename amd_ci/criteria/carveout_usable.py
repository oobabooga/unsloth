#!/usr/bin/env python3
"""Criteria (#19768): a Windows gfx1151 APU with a BIOS carve-out must be sized above Windows RAM.

Base defect: usable ~= host available - 1024 although HIP and the carve-out offer more.
Head fixed: usable rises by most of the unused carve-out, stays within HIP free, and that
much really allocates on the GPU."""

from __future__ import annotations

TITLE = "Windows APU usable GPU memory with a BIOS carve-out"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows", "integrated_gpu", "windows_rocm_wddm", "exclusive_gpu"]


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_") and isinstance(v, dict)}


def gates(obs):
    out = []
    for n, o in _states(obs).items():
        out.append((f"{n}: Windows + ROCm torch on gfx1151",
                    o.get("platform") == "win32" and bool(o.get("hip")) and
                    str(o.get("arch") or "").startswith("gfx1151"),
                    f"{o.get('platform')} torch={o.get('torch')} hip={o.get('hip')} arch={o.get('arch')} "
                    f"{o.get('torch_error') or ''}"))
        out.append((f"{n}: probe produced a row", o.get("usable_mib") is not None,
                    o.get("backend_error") or f"rows={o.get('probe_rows')}"))
    b = obs.get("base") or {}
    carve = b.get("registry_carve_out_mib")
    out.append(("box has a carve-out >= 4 GiB (else nothing to credit)",
                bool(carve) and carve >= 4096, f"registry carve-out {carve} MiB"))
    out.append(("WDDM usage readable", b.get("wddm_used_mib") is not None,
                f"{b.get('wddm_used_mib')} MiB in {b.get('wddm_used_s')} s"))
    return out


def table(obs):
    rows = ["| state | HIP free | host avail | carve-out | WDDM used | reader | usable | probe s | alloc |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n, o in _states(obs).items():
        a = o.get("alloc") or {}
        alloc = f"{a.get('allocated_mib')}/{a.get('target_mib')} ok={a.get('ok')} minavail={a.get('min_host_avail_mib')}" if a else "-"
        rows.append(f"| {n} | {o.get('hip_free_mib')} | {o.get('host_avail_mib')} | {o.get('registry_carve_out_mib')} "
                    f"| {o.get('wddm_used_mib')} | {o.get('reader_mib', '-')} (cold {o.get('reader_cold_s', '-')}s) "
                    f"| {o.get('usable_mib')} | {o.get('probe_s')} | {alloc} |")
    return "\n".join(rows)


def _unused_carve(o):
    c, u = o.get("registry_carve_out_mib"), o.get("wddm_used_mib")
    return None if c is None or u is None else max(0, c - u)


def base_shows_defect(base):
    usable, avail, hip = base.get("usable_mib"), base.get("host_avail_mib"), base.get("hip_free_mib")
    unused = _unused_carve(base)
    if None in (usable, avail, hip, unused):
        return None, "missing reading"
    room = min(hip, avail + unused) - 1024
    return usable <= avail and room - usable > 2048, (
        f"usable {usable} MiB vs host avail {avail}; HIP {hip} and unused carve-out {unused} allow {room}")


def head_is_fixed(head):
    usable, avail, hip = head.get("usable_mib"), head.get("host_avail_mib"), head.get("hip_free_mib")
    unused, a = _unused_carve(head), head.get("alloc") or {}
    if None in (usable, avail, hip, unused):
        return None, "missing reading"
    gain = usable - (avail - 1024)
    want = min(hip - avail, unused)
    ok = usable <= hip and gain >= 0.8 * want and bool(a.get("ok"))
    return ok, (f"usable {usable} MiB (+{gain} over the RAM cap, want ~{want}); HIP {hip}; "
                f"alloc {a.get('allocated_mib')}/{a.get('target_mib')} ok={a.get('ok')} {a.get('error') or ''}")
