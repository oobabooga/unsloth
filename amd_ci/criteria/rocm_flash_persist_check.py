#!/usr/bin/env python3
"""Judge the rocm_flash_persist_probe runs: head process 1 times and writes, head process 2 reads the file, makes
the same choice and times nothing, forced modes time nothing and leave the file untouched; base process 2 times
again (the defect). Prints the verdict and writes JSON to --out."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(d: Path, name: str):
    p = d / f"persist_{name}.json"
    return json.loads(p.read_text(encoding = "utf-8")) if p.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    runs = {n: load(args.dir, n) for n in ("base_1", "base_2", "head_1", "head_2", "head_off", "head_on")}
    checks = {}

    def check(name, cond):
        checks[name] = bool(cond)

    b1, b2, h1, h2, off, on = (runs[n] for n in ("base_1", "base_2", "head_1", "head_2", "head_off", "head_on"))
    missing = [n for n, r in runs.items() if r is None]
    if not missing:
        verdict = lambda r: [g["admitted"] for g in r["geometries"]]
        check("base_1 timed both geometries", b1["timer_calls_total"] == 2)
        check("base_2 times again (defect: no persistence)", b2["timer_calls_total"] == 2)
        check("head_1 timed both geometries", h1["timer_calls_total"] == 2)
        check("head_1 wrote one cache file with both verdicts", len(h1["cache_files"]) == 1
              and len(next(iter(h1["cache_files"].values()))["data"]["verdicts"]) == 2)
        check("head_2 timed nothing", h2["timer_calls_total"] == 0)
        check("head_2 records came from the cache", all((g["record"] or {}).get("cached") for g in h2["geometries"]))
        check("head_2 same choice as head_1", verdict(h1) == verdict(h2))
        check("head_2 left the file untouched", h2["cache_files"] == h1["cache_files"])
        for r, mode in ((off, "0"), (on, "1")):
            check(f"ROCM_FLASH={mode} timed nothing", r["timer_calls_total"] == 0)
            check(f"ROCM_FLASH={mode} left the file untouched", r["cache_files"] == h1["cache_files"])
            check(f"ROCM_FLASH={mode} has no admission records", all(g["record"] is None for g in r["geometries"]))
    ok = not missing and all(checks.values())
    summary = {
        "verdict": "CONFIRMED" if ok else ("INCONCLUSIVE" if missing else "FAIL"),
        "missing": missing,
        "checks": checks,
        "per_run": {
            n: None if r is None else {
                "timer_calls": r["timer_calls_total"],
                "geometries": [
                    {k: g[k] for k in ("geometry", "admitted", "seconds", "timer_calls")}
                    | {k: (g["record"] or {}).get(k) for k in ("flash_ms", "native_ms", "rel_l1", "cached")}
                    for g in r["geometries"]
                ],
            }
            for n, r in runs.items()
        },
        "toolchain_key": (h1 or {}).get("toolchain_key"),
    }
    args.out.write_text(json.dumps(summary, indent = 1), encoding = "utf-8")
    print(json.dumps(summary, indent = 1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
