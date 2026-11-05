#!/usr/bin/env python3
"""Tables for the nov5 gfx1151 round from downloaded AMD CI artifacts (obs_base.json / obs_head.json per job).

  python diffusion_bench/amd/nov5_table.py RUN_DIR [RUN_DIR ...] --out outputs/amd_gfx1151_nov5/tables.md --json T.json

Per family: one row per arm (s/step median over the arm and its *_repeat copies, with min-max spread), s/image
(new prompt), steady, cold (first render incl. load-time compile), load, peak (Studio in-process torch peak; device-
wide VRAM+GTT delta for every backend when sysfs gave a reading), LPIPS vs the same stack's reference cell, and the
engaged attention / speed facts. Ratios: Studio arm / ComfyUI c_rock (same torch) and / c_def (README torch).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

FAM_NAME = {"klein": "FLUX.2-klein-4B", "f3": "FLUX.3 klein", "zimg": "Z-Image-Turbo", "q21": "Qwen-Image-2.1",
            "wan": "Wan2.2-TI2V-5B (1280x704x21)", "tiny": "tiny FLUX pipe (plumbing)", "h3": "MiniMax-H3"}
ARM_DESC = {"s_def": "Studio main (torch 2.11+rocm7.13)", "s_flash": "Studio + flash-attn (auto)",
            "s_flashx": "Studio + flash-attn (explicit)", "s_pt72n": "Studio on torch 2.14.1+rocm7.2",
            "c_rock": "ComfyUI v0.44.0 (Studio's torch)", "c_def": "ComfyUI v0.44.0 (README torch 2.14.1+rocm7.2)",
            "c_nockhip": "ComfyUI, Kitchen HIP off", "s_pt72n_flash": "Studio on torch 2.14.1 + flash-attn",
            "s_int8": "Studio int8 weight-only", "s_int8act": "Studio int8 W8A8"}


def load(dirs):
    cells = {}
    meta = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("obs_*.json")):
            o = json.loads(p.read_text(encoding = "utf-8"))
            st = o.get("state")
            meta.setdefault(st, {"host": o.get("host"), "rev": o.get("checkout_rev")})
            for tag, c in (o.get("cells") or {}).items():
                key = tag if st == "base" else f"{tag}@head"
                cells[key] = {**c, "state": st, "src": str(p)}
    return cells, meta


def arm_of(tag):
    t = tag.split("@")[0]
    fam, _, rest = t.partition("_")
    for suf in ("_2_repeat", "_3_repeat"):
        if rest.endswith(suf):
            rest = rest[: -len(suf)]
    return fam, rest


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (statistics.median(xs), min(xs), max(xs), len(xs)) if xs else None


def fmt(m, nd = 2):
    if not m:
        return "-"
    v, lo, hi, n = m
    return f"{v:.{nd}f}" + (f" ({lo:.{nd}f}-{hi:.{nd}f}, n={n})" if n > 1 else "")


def build(cells):
    fams = {}
    for tag, c in cells.items():
        fam, arm = arm_of(tag)
        fams.setdefault(fam, {}).setdefault(arm, []).append((tag, c))
    rows = []
    for fam, arms in fams.items():
        for arm, items in arms.items():
            ok = [c for _, c in items if c.get("verdict") == "ok"]
            r = {"family": fam, "arm": arm, "cells": [t for t, _ in items], "ok": f"{len(ok)}/{len(items)}",
                 "errors": [(t, (c.get("error") or "")[:200]) for t, c in items if c.get("verdict") != "ok"],
                 "step_s": med([c.get("step_s") for c in ok]), "new_s": med([c.get("new_s") for c in ok]),
                 "steady_s": med([c.get("steady_s") for c in ok]), "cold_s": med([c.get("cold_s") for c in ok]),
                 "load_s": med([c.get("load_s") for c in ok]),
                 "peak_alloc_gib": med([c.get("peak_alloc_gib") for c in ok]),
                 "peak_dev_gib": med([c.get("peak_smi_delta_gib") or None for c in ok]),
                 "peak_rss_gib": med([c.get("peak_rss_gib") for c in ok]),
                 "lpips": med([c.get("lpips") for c in ok]),
                 "attention": sorted({str((c.get("status") or {}).get("attention_backend")) for c in ok}),
                 "speed_optims": sorted({o for c in ok for o in ((c.get("status") or {}).get("speed_optims") or [])}),
                 "flags": sorted({str(f) for c in ok for f in (c.get("flags") or [])})}
            rows.append(r)
    return rows


def ratio(a, b):
    return f"{a[0] / b[0]:.2f}x" if a and b and b[0] else "-"


def markdown(rows):
    out = []
    by = {}
    for r in rows:
        by.setdefault(r["family"], {})[r["arm"]] = r
    for fam, arms in by.items():
        out.append(f"\n### {FAM_NAME.get(fam, fam)}\n")
        out.append("| arm | ok | s/step | s/image (new prompt) | steady s | cold s | load s | peak GiB (torch / device) "
                   "| LPIPS vs own ref | s/step vs c_rock | s/step vs c_def | attention | notes |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        cr, cd = arms.get("c_rock", {}).get("step_s"), arms.get("c_def", {}).get("step_s")
        order = ["s_def", "s_flash", "s_flashx", "s_pt72n", "s_pt72n_flash", "s_int8", "s_int8act", "c_rock", "c_def", "c_nockhip"]
        for arm in sorted(arms, key = lambda a: order.index(a) if a in order else 99):
            r = arms[arm]
            peak = f"{fmt(r['peak_alloc_gib'], 1).split(' ')[0]} / {fmt(r['peak_dev_gib'], 1).split(' ')[0]}"
            notes = "; ".join([f"{t}: {e}" for t, e in r["errors"]][:2] + r["flags"][:3])
            out.append(f"| {ARM_DESC.get(arm, arm)} | {r['ok']} | {fmt(r['step_s'], 3)} | {fmt(r['new_s'])} | "
                       f"{fmt(r['steady_s'])} | {fmt(r['cold_s'], 1)} | {fmt(r['load_s'], 1)} | {peak} | "
                       f"{fmt(r['lpips'], 4)} | {ratio(r['step_s'], cr)} | {ratio(r['step_s'], cd)} | "
                       f"{','.join(r['attention'])} | {notes} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs = "+")
    ap.add_argument("--out", type = Path)
    ap.add_argument("--json", type = Path)
    a = ap.parse_args()
    cells, meta = load(a.dirs)
    rows = build(cells)
    md = markdown(rows)
    hosts = {st: {k: (m.get("host") or {}).get(k) for k in ("node", "torch", "hip", "device_name", "arch")} | {
        "rev": m.get("rev")} for st, m in meta.items()}
    md = "hosts / states: " + json.dumps(hosts) + "\n" + md
    print(md)
    if a.out:
        a.out.parent.mkdir(parents = True, exist_ok = True)
        a.out.write_text(md, encoding = "utf-8")
    if a.json:
        a.json.write_text(json.dumps({"rows": rows, "hosts": hosts}, indent = 1, default = str), encoding = "utf-8")


if __name__ == "__main__":
    main()
