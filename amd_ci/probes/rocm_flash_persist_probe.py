#!/usr/bin/env python3
"""One process of the ROCm flash admission on the real card, at the klein / Wan2.2-5B (3072, 24, 128) and Z-Image
(3840, 30, 128) geometries: counts how often the block timer ran, the verdict, the wall time and what the cache
dir holds afterwards. Observes only; writes JSON to --out. Run it twice in a row per state: on the head the second
process must read the file and time nothing; on the base it times again."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-dir", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--label", required = True)
    args = ap.parse_args()
    sys.path.insert(0, args.backend_dir)
    logging.basicConfig(level = logging.INFO, format = "%(levelname)s %(name)s: %(message)s")

    import torch
    from torch import nn

    import core.inference.diffusion_attention as att

    calls = []
    real_timer = att._time_rocm_flash_block

    def counted(*a, **k):
        calls.append([int(v) for v in a[2:5]])
        return real_timer(*a, **k)

    att._time_rocm_flash_block = counted
    out = {
        "label": args.label,
        "mode": os.environ.get("UNSLOTH_DIFFUSION_ROCM_FLASH", "auto"),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(0),
        "has_persistence": hasattr(att, "_rocm_flash_toolchain_key"),
        "geometries": [],
    }
    try:
        import flash_attn

        out["flash_attn"] = getattr(flash_attn, "__version__", None)
        out["flash_attn_file"] = flash_attn.__file__
    except Exception as exc:  # noqa: BLE001
        out["flash_attn_error"] = repr(exc)
    if out["has_persistence"]:
        out["toolchain_key"] = att._rocm_flash_toolchain_key("cuda:0")
    target = type("T", (), {"device": "cuda", "torch_device": "cuda:0", "dtype": torch.bfloat16})()
    for hidden, heads in ((3072, 24), (3840, 30)):
        attn = nn.Module()
        attn.heads = heads
        attn.to_q = nn.Linear(hidden, hidden, bias = False)
        dit = nn.Module()
        dit.attn = attn
        att._attention_dits = lambda p, d = dit: [d]
        n0 = len(calls)
        t0 = time.perf_counter()
        admitted = att._rocm_flash_block_admitted(object(), target)
        dt = time.perf_counter() - t0
        rec = att.ROCM_FLASH_ADMISSIONS.get(("cuda:0", str(torch.bfloat16), hidden, heads, hidden // heads))
        out["geometries"].append({
            "geometry": [hidden, heads, hidden // heads],
            "admitted": bool(admitted),
            "seconds": round(dt, 4),
            "timer_calls": len(calls) - n0,
            "record": rec,
        })
    out["timer_calls_total"] = len(calls)
    root = os.environ.get("UNSLOTH_DIFFUSION_ROCM_FLASH_CACHE_DIR", "")
    files = sorted(Path(root).glob("*.json")) if root and Path(root).is_dir() else []
    out["cache_files"] = {
        f.name: {"mtime_ns": f.stat().st_mtime_ns, "data": json.loads(f.read_text(encoding = "utf-8"))}
        for f in files
    }
    args.out.write_text(json.dumps(out, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
