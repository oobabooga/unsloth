#!/usr/bin/env python3
"""Tables for the prio8 image job: one row per cell (record.json + scores.json + prof_hook calls), then the
profiled denoiser step / text encoder / VAE decode per cell (category shares, top kernels, idle gaps).

  prio8_report.py --runs RUNS --prof PROF --spec SPEC --out REPORT.md [--json OUT.json]

s/step (device): the sum of the denoiser calls' device time in the steady render (the long render when the cell has
a long/short pair, else the last new-prompt render) / steps; the run_cell derived figure ((long - short) / step
difference) is printed beside it. Peaks: whole cell (load included) device VRAM + GTT delta from amdgpu sysfs, and the
process tree's RSS (the baseline's server included).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def calls(d: Path) -> list:
    f = d / "calls.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding = "utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def steady_render(rec: dict) -> str:
    """render label of the steady render: r0 warm-up, r1..rN prompts, then long / short pairs."""
    n = len(rec.get("renders") or [])
    return f"r{n + 1}" if rec.get("long") else f"r{n}"


def per_render(cs: list, label_suffix: str) -> dict:
    by = {}
    for c in cs:
        if str(c.get("label", "")).endswith(label_suffix) and c.get("gpu_ms") is not None:
            by.setdefault(c.get("render"), []).append(c["gpu_ms"])
    return by


def fnum(v, nd = 2):
    return "-" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required = True, type = Path)
    ap.add_argument("--prof", required = True, type = Path)
    ap.add_argument("--spec", type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--json", type = Path)
    a = ap.parse_args()
    lpips = {}
    for sj in a.runs.glob("*/scores.json"):
        st = sj.parent.name
        for r in json.loads(sj.read_text(encoding = "utf-8")).get("rows", []):
            lpips[(st, r["tag"])] = r.get("lpips")
    rows, profs = [], []
    for rp in sorted(a.runs.glob("*/*/record.json")):
        st, tag = rp.parent.parent.name, rp.parent.name
        rec = json.loads(rp.read_text(encoding = "utf-8"))
        name = tag if st == "base" else f"{tag}@head"
        pd = a.prof / name
        cs = calls(pd)
        steps = int((rec.get("cell") or {}).get("steps") or 0)
        den = per_render(cs, "denoiser")
        sr = steady_render(rec)
        dev_step = round(sum(den[sr]) / steps / 1000.0, 4) if steps and den.get(sr) else None
        te = per_render(cs, "_te")
        vae = per_render(cs, "_vae")
        te_ms = [sum(v) for k, v in te.items() if k and k.startswith("r") and k != "r0"]
        vae_ms = [sum(v) for k, v in vae.items() if k and k.startswith("r") and k != "r0"]
        walls = [r.get("wall_s") for r in rec.get("renders") or []]
        steady = rec.get("steady_s_median") or (walls[-1] if walls else None)
        st_ = rec.get("status") or {}
        row = {"cell": name, "backend": rec.get("backend"), "verdict": rec.get("verdict"),
               "error": (rec.get("error") or "")[:160], "steps": steps, "dev_s_step": dev_step,
               "derived_s_step": rec.get("step_s_derived"), "calls_per_step": round(len(den.get(sr, [])) / steps, 2)
               if steps and den.get(sr) else None,
               "new_s": statistics.median(walls[1:]) if len(walls) > 1 else (walls[0] if walls else None),
               "first_new_s": walls[0] if walls else None, "steady_s": steady, "load_s": rec.get("load_s"),
               "cold_s": rec.get("cold_s"), "te_ms": round(statistics.median(te_ms), 1) if te_ms else None,
               "vae_ms": round(statistics.median(vae_ms), 1) if vae_ms else None,
               "peak_dev_cell_gib": rec.get("cell_dev_peak_delta_gib"), "peak_dev_timed_gib": rec.get("peak_smi_delta_gib"),
               "peak_tree_rss_gib": rec.get("cell_tree_rss_peak_gib"), "memavail_drop_gib": rec.get("cell_memavail_drop_gib"),
               "lpips": lpips.get((st, tag)),
               "engaged": {k: st_.get(k) for k in ("transformer_quant", "text_encoder_quant", "offload_policy",
                                                     "speed_optims", "attention_backend", "gguf_filename", "engine",
                                                     "prof_attach") if st_.get(k) not in (None, [], "")}}
        if row["load_s"] is not None and row["cold_s"] is not None:
            row["first_output_s"] = round(row["load_s"] + row["cold_s"], 1)
        rows.append(row)
        for pf in sorted(pd.glob("prof_*.json")):
            try:
                profs.append((name, json.loads(pf.read_text(encoding = "utf-8"))))
            except ValueError:
                pass
    out = ["## prio8 image cells", "",
           "| cell | ok | s/step (device) | s/step (long-short) | calls/step | s/image new (median) | steady s/image | "
           "load s | first render s | load + first s | TE ms | VAE ms | peak dev GiB (cell / timed) | tree RSS GiB | "
           "MemAvail drop GiB | LPIPS vs own bf16 | engaged |", "|" + "---|" * 17]
    for r in rows:
        ok = r["verdict"] if r["verdict"] == "ok" else f"{r['verdict']}: {r['error']}"
        out.append(f"| {r['cell']} | {ok} | {fnum(r['dev_s_step'], 3)} | {fnum(r['derived_s_step'], 3)} | "
                   f"{r['calls_per_step'] if r['calls_per_step'] is not None else '-'} | {fnum(r['new_s'])} | "
                   f"{fnum(r['steady_s'])} | {fnum(r['load_s'], 1)} | {fnum(r['cold_s'], 1)} | "
                   f"{fnum(r.get('first_output_s'), 1)} | {fnum(r['te_ms'], 0)} | {fnum(r['vae_ms'], 0)} | "
                   f"{fnum(r['peak_dev_cell_gib'], 1)} / {fnum(r['peak_dev_timed_gib'], 1)} | "
                   f"{fnum(r['peak_tree_rss_gib'], 1)} | {fnum(r['memavail_drop_gib'], 1)} | {fnum(r['lpips'], 4)} | "
                   f"{json.dumps(r['engaged'], default = str)[:300]} |")
    for name, p in profs:
        out.append(f"\n### {name} {p.get('label')} (render prof, call {p.get('call')}): wall {p['wall_ms']} ms, kernels "
                   f"{p['n_kernels']}, kernel sum {p['kernel_sum_us'] / 1000:.1f} ms, GPU busy {p['gpu_busy_us'] / 1000:.1f}"
                   f" of span {p['gpu_span_us'] / 1000:.1f} ms, idle gaps >0.2 ms: {p.get('idle_gaps_over_0p2ms')}\n")
        tot = sum(p["categories_us"].values()) or 1
        out.append("| category | ms | share |")
        out.append("|---|---|---|")
        for k, v in p["categories_us"].items():
            out.append(f"| {k} | {v / 1000:.1f} | {100 * v / tot:.1f}% |")
        out.append(f"\nhost sync-like: {json.dumps(p.get('host_sync_like'))}\n")
        out.append("| top kernel | n | ms | cat |")
        out.append("|---|---|---|---|")
        for k in p["kernels"][:15]:
            out.append(f"| `{k['name'][:100]}` | {k['count']} | {k['us'] / 1000:.2f} | {k['cat']} |")
    a.out.write_text("\n".join(out) + "\n", encoding = "utf-8")
    if a.json:
        a.json.write_text(json.dumps({"rows": rows, "profiles": [(n, {k: v for k, v in p.items() if k != "ops"})
                                                                    for n, p in profs]}, indent = 1, default = str),
                          encoding = "utf-8")
    print("\n".join(out[:60]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
