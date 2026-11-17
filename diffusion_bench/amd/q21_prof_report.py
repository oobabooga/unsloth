#!/usr/bin/env python3
"""Markdown summary of q21_prof_hook output: per arm, steady decode-step device time and the profiled calls' kernel
categories. Usage: q21_prof_report.py OUT_DIR/prof"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def calls(d: Path) -> list:
    f = d / "calls.jsonl"
    if not f.exists():
        return []
    return [json.loads(x) for x in f.read_text(encoding = "utf-8").splitlines() if x.strip()]


def main() -> int:
    root = Path(sys.argv[1])
    print("## Qwen-Image-2.1 bf16 1024^2 denoiser call profile\n")
    print("| arm | calls | decode gpu ms (median, render 4 steps 2-24) | long render steps 2-24 | host ms median | s/step derived | record |")
    print("|---|---|---|---|---|---|---|")
    profs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        cs = calls(d)
        by = {c["call"]: c for c in cs}

        def med(lo, hi, key = "gpu_ms"):
            v = [by[i][key] for i in range(lo, hi) if i in by and key in by[i]]
            return (round(statistics.median(v), 1), len(v)) if v else (None, 0)

        rec = {}
        rf = root / f"{d.name}.record.json"
        if rf.exists():
            try:
                rec = json.loads(rf.read_text(encoding = "utf-8"))
            except Exception:  # noqa: BLE001
                rec = {}
        r4 = med(102, 125)
        lg = med(127, 150)
        hm = med(127, 150, "host_ms")
        print(f"| {d.name} | {len(cs)} | {r4[0]} (n={r4[1]}) | {lg[0]} (n={lg[1]}) | {hm[0]} | "
              f"{rec.get('step_s_derived')} | {rec.get('verdict')} |")
        for pf in sorted(d.glob("prof_*.json")):
            profs.append((d.name, json.loads(pf.read_text(encoding = "utf-8"))))
    for name, p in profs:
        print(f"\n### {name} call {p['call']}: wall {p['wall_ms']} ms, kernels {p['n_kernels']}, "
              f"kernel sum {p['kernel_sum_us'] / 1000:.1f} ms, GPU busy {p['gpu_busy_us'] / 1000:.1f} ms of span "
              f"{p['gpu_span_us'] / 1000:.1f} ms (idle {p['gpu_idle_us'] / 1000:.1f} ms)\n")
        print("| category | ms |")
        print("|---|---|")
        for k, v in p["categories_us"].items():
            print(f"| {k} | {v / 1000:.1f} |")
        print(f"\nhost sync-like: {json.dumps(p.get('host_sync_like'))}\n")
        print("| top kernel | n | ms | cat |")
        print("|---|---|---|---|")
        for k in p["kernels"][:25]:
            print(f"| `{k['name'][:110]}` | {k['count']} | {k['us'] / 1000:.2f} | {k['cat']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
