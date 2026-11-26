#!/usr/bin/env python3
"""Leg gate: FAIL unless run_bench measured at least one row on the expected llama.cpp backend.
usage: calib_gate.py OUT.jsonl BACKEND(ROCm|Vulkan) SUMMARY.md"""
import json, sys

path, backend, summ = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
try:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
except OSError as e:
    print(f"GATE FAILED: no results file {path}: {e}")
    sys.exit(1)
skips, errs, ok = [], [], []
devs = set()
for r in rows:
    for d in ((r.get("hw") or {}).get("llama_devices") or []):
        devs.add(d)
    if r.get("skip"):
        skips.append(r)
    elif r.get("error"):
        errs.append(r)
    else:
        res = r.get("result") or {}
        if (res.get("tg") or {}).get("avg_ts") or (res.get("pp") or {}).get("avg_ts") or res.get("tg_med") or res.get("pp_med"):
            ok.append(r)
on_backend = any(d.strip().startswith(backend) for d in devs)
lines = [f"## {path}", f"rows {len(rows)}: measured {len(ok)}, error {len(errs)}, skip {len(skips)}",
         f"llama devices: {sorted(devs) or 'NONE'}"]
reasons = {}
for r in skips:
    k = str(r["skip"])[:90]
    reasons[k] = reasons.get(k, 0) + 1
for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:15]:
    lines.append(f"- skip x{v}: {k}")
for r in errs[:10]:
    lines.append(f"- error {r.get('id')} r{r.get('round')}: {str(r.get('error'))[:200]}")
txt = "\n".join(lines) + "\n"
print(txt)
with open(summ, "a", encoding="utf-8") as f:
    f.write(txt)
if not on_backend:
    print(f"GATE FAILED: the bundle loaded no {backend} device (devices {sorted(devs)})")
    sys.exit(1)
if not ok:
    print("GATE FAILED: no row measured")
    sys.exit(1)
print("gate ok")
