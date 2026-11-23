#!/usr/bin/env python3
"""Criteria (gfx1151, differential): the FLUX.2 1024 px whole-map strip decode with the upsample fold on.
Defect (base): the whole-map fold's peak above the pre-decode allocation exceeds the tree's own estimate
full_map_decode_mib (the admission check under-prices it). Fixed (head): the fold-on peak is within the estimate and
within 15% (+64 MiB) of the fold-off peak, the fold still ran (fold calls > 0) in strips (conv_transpose2d calls >
fold calls), and the output stays within the fold's own tolerance of the base (PSNR >= 40 dB vs the base fold render).
Speed (fold on vs off, both states) is reported in the table, not gated. Judges probes/vae_fold_peak_probe.py."""

from __future__ import annotations

import statistics

TITLE = "gfx1151: FLUX.2 VAE whole-map decode, upsample fold peak vs full_map_decode_mib (base vs head)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "linux", "windows", "windows_rocm_wddm", "nvidia", "discrete_gpu"]


def _arms(o, res, arm):
    return [v for k, v in (o.get("arms") or {}).items() if k.startswith(f"{res}_{arm}_") and v.get("ok")]


def _med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(vals) if vals else None


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        arms = o.get("arms") or {}
        bad = [k for k, v in arms.items() if not v.get("ok")]
        out.append((f"{name}: on ROCm gfx1151", bool(o.get("cuda")) and bool(o.get("hip"))
                    and str(o.get("arch", "")).split(":")[0] == "gfx1151", f"{o.get('arch')} hip {o.get('hip')}"))
        out.append((f"{name}: every arm decoded", bool(arms) and not bad and not o.get("error"),
                    f"{len(arms)} arms; failed {bad[:4]} {str(o.get('error') or '')[:200]}"))
        mods = {str(v.get("module", "")) for v in arms.values() if v.get("module")}
        out.append((f"{name}: arms imported this state's tree", bool(mods) and all(
            m.startswith(str(o.get("checkout", "")).rstrip("/") + "/") or str(o.get("checkout", "")) in m for m in mods),
                    ", ".join(sorted(mods))[:300]))
        on = _arms(o, 1024, "auto")
        off = _arms(o, 1024, "off")
        out.append((f"{name}: fold engaged on auto, off on the kill switch",
                    bool(on) and all(r.get("fold_calls", 0) > 0 for r in on) and bool(off)
                    and all(r.get("fold_calls", 1) == 0 for r in off),
                    f"auto fold calls {[r.get('fold_calls') for r in on]}, off {[r.get('fold_calls') for r in off]}"))
    return out


def base_shows_defect(base):
    on = _arms(base, 1024, "auto")
    if not on:
        return False
    return max(r["peak_delta_mib"] for r in on) > on[0]["estimate_mib"]


def head_is_fixed(head):
    on, off = _arms(head, 1024, "auto"), _arms(head, 1024, "off")
    if not on or not off:
        return False
    peak_on = max(r["peak_delta_mib"] for r in on)
    peak_off = max(r["peak_delta_mib"] for r in off)
    strips = all(r.get("conv_transpose_calls", 0) > r.get("fold_calls", 0) > 0 for r in on)
    c = (head.get("compare") or {}).get("1024_auto_vs_base_auto") or {}
    close = c.get("psnr_db") is not None and c["psnr_db"] >= 40.0
    return peak_on <= on[0]["estimate_mib"] and peak_on <= 1.15 * peak_off + 64 and strips and close


def table(obs):
    rows = ["| state | px | arm | decode s (median of runs) | peak above pre-decode MiB | estimate MiB | fold calls | "
            "conv_transpose2d calls | md5 |", "|---|---|---|---|---|---|---|---|---|"]
    speed = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for res in (1024, 2048):
            for arm in ("off", "auto", "whole"):
                rs = _arms(o, res, arm)
                if not rs:
                    continue
                rows.append(f"| {name} | {res} | {arm} | {_med(rs, 'decode_s')} ({', '.join(str(r['decode_s']) for r in rs)}) | "
                            f"{', '.join(str(r['peak_delta_mib']) for r in rs)} | {rs[0].get('estimate_mib'):.0f} | "
                            f"{rs[0].get('fold_calls')} | {rs[0].get('conv_transpose_calls')} | {rs[0].get('md5', '')[:10]} |")
            a, b = _med(_arms(o, res, "off"), "decode_s"), _med(_arms(o, res, "auto"), "decode_s")
            if a and b:
                speed.append(f"- {name} {res} px: fold off / on = {a:.4f} / {b:.4f} s = {a / b:.3f}x")
    cmp = []
    for name in ("base", "head"):
        for k, v in sorted(((obs.get(name) or {}).get("compare") or {}).items()):
            cmp.append(f"- {name} {k}: {v}")
    return "\n".join(rows + [""] + speed + [""] + cmp)
