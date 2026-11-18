#!/usr/bin/env python3
"""Studio's persisted INT8 attention verdicts (diffusion_int8_attention.json under each studio home), one row per card
entry with the device ordinal its gpu_uuid maps to on this machine. Usage: persist_dump.py --out X.json HOME [HOME ...]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("homes", nargs = "+")
    a = ap.parse_args()
    import torch

    uuids = {}
    for i in range(torch.cuda.device_count()):
        u = str(getattr(torch.cuda.get_device_properties(i), "uuid", "") or "")
        uuids[u] = i
    res = {"ordinals_by_uuid": uuids, "files": []}
    for h in a.homes:
        for f in sorted(Path(h).rglob("diffusion_int8_attention.json")) if Path(h).exists() else []:
            try:
                data = json.loads(f.read_text(encoding = "utf-8"))
            except Exception as exc:  # noqa: BLE001
                res["files"].append({"path": str(f), "error": repr(exc)})
                continue
            entries = []
            for key, e in data.items():
                fp = (e or {}).get("fingerprint") or {}
                dec = (e or {}).get("decisions") or {}
                entries.append({
                    "slot": key, "gpu_name": fp.get("gpu_name"), "gpu_uuid": fp.get("gpu_uuid"),
                    "ordinal": uuids.get(str(fp.get("gpu_uuid"))), "capability": fp.get("capability"),
                    "hip": fp.get("hip"), "n_decisions": len(dec),
                    "decisions": {k: {kk: (v or {}).get(kk) for kk in ("ok", "impl", "min_tokens", "cos", "rel_l1")}
                                  if isinstance(v, dict) else v for k, v in dec.items()},
                })
            res["files"].append({"path": str(f), "entries": entries})
    Path(a.out).write_text(json.dumps(res, indent = 1, default = str))
    n = sum(len(x.get("entries", [])) for x in res["files"])
    print("PERSIST_DUMP", n, "entries", json.dumps({e["ordinal"]: e["n_decisions"] for x in res["files"]
                                                    for e in x.get("entries", [])}), flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
