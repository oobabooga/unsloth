#!/usr/bin/env python3
"""Judge the rocm_flash_persist_probe runs. Head process 1 times the 4096-token load bucket per geometry plus every
lazily timed bucket and writes them; head process 2 reads the file, makes the same choices and times nothing; forced
modes time nothing and leave the file untouched. Base: runs without crashing, has no block admission, and its default
``auto`` pick is native for Z-Image and Wan. Prints the verdict and writes JSON to --out."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NAMES = ("base_1", "base_2", "head_1", "head_2", "head_off", "head_on")


def load(d: Path, name: str):
    p = d / f"persist_{name}.json"
    return json.loads(p.read_text(encoding = "utf-8")) if p.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    runs = {n: load(args.dir, n) for n in NAMES}
    checks = {}

    def check(name, cond):
        checks[name] = bool(cond)

    b1, b2, h1, h2, off, on = (runs[n] for n in NAMES)
    missing = [n for n, r in runs.items() if r is None]
    if not missing:
        for name, b in (("base_1", b1), ("base_2", b2)):
            check(f"{name} has no block admission (merge base)", not b["has_block_admission"])
            check(f"{name} default auto pick is native for z-image and wan",
                  b["auto_pick"].get("z-image") is None and b["auto_pick"].get("wan2.2-ti2v-5b") is None)
        slots = lambda r: [(tuple(b["geometry"]), b["bucket"]) for b in r["buckets"]]
        adm = lambda r: [g["admitted"] for g in r["geometries"]] + [b["admitted"] for b in r["buckets"]]
        check("head_1 auto pick is the timed candidate for z-image",
              h1["auto_pick"].get("z-image") == "_rocm_flash_candidate")
        check("head_1 timed both load geometries", [g["timer_calls"] for g in h1["geometries"]] == [1, 1])
        check("head_1 timed every non-4096 bucket once, 4096 from memory",
              all(b["timer_calls"] == (0 if b["bucket"] == 4096 else 1) for b in h1["buckets"]))
        check("head_1 no bucket raised", all(b["error"] is None for b in h1["buckets"]))
        f1 = next(iter(h1["cache_files"].values()), {}).get("data", {})
        check("head_1 wrote one cache file with every valid verdict",
              len(h1["cache_files"]) == 1 and len(f1.get("verdicts", {})) == len(set(slots(h1))))
        check("head_1 cache file key is v3", str(f1.get("key", "")).startswith("v3|"))
        check("head_2 timed nothing", h2["timer_calls_total"] == 0)
        check("head_2 every record came from the cache",
              all(all(r.get("cached") for r in g["records"].values()) for g in h2["geometries"])
              and all((b["record"] or {}).get("cached") for b in h2["buckets"]))
        check("head_2 same choices as head_1", adm(h1) == adm(h2) and slots(h1) == slots(h2))
        check("head_2 left the file untouched", h2["cache_files"] == h1["cache_files"])
        for r, mode in ((off, "0"), (on, "1")):
            check(f"ROCM_FLASH={mode} timed nothing", r["timer_calls_total"] == 0)
            check(f"ROCM_FLASH={mode} left the file untouched", r["cache_files"] == h1["cache_files"])
            check(f"ROCM_FLASH={mode} no length gate installed", not any(g["gate_installed"] for g in r["geometries"]))
    ok = not missing and all(checks.values())
    keep = ("flash_ms", "native_ms", "rel_l1", "max_rel", "block_rel_l1", "cached", "admitted")
    summary = {
        "verdict": "CONFIRMED" if ok else ("INCONCLUSIVE" if missing else "FAIL"),
        "missing": missing,
        "checks": checks,
        "per_run": {
            n: None if r is None else {
                "auto_pick": r.get("auto_pick"),
                "has_block_admission": r.get("has_block_admission"),
                "timer_calls": r.get("timer_calls_total"),
                "geometries": [
                    {k: g[k] for k in ("geometry", "admitted", "seconds", "timer_calls", "peak_alloc_gib", "gate_installed")}
                    for g in r["geometries"]
                ],
                "buckets": [
                    {k: b[k] for k in ("geometry", "bucket", "admitted", "seconds", "timer_calls", "peak_alloc_gib", "error")}
                    | {k: (b["record"] or {}).get(k) for k in keep}
                    for b in r["buckets"]
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
