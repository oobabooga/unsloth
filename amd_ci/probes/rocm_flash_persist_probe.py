#!/usr/bin/env python3
"""One process of the ROCm flash admission on the real card. Observes only; writes JSON to --out.

1. Load-time admission (``_rocm_flash_block_admitted``) for the Wan2.2-5B (3072, 24, 128) and Z-Image (3840, 30, 128)
   DiT geometries, through a stand-in DiT.
2. (auto mode only) the per-length buckets the generation path times lazily, called the way
   ``_rocm_flash_length_reason`` calls them (``_rocm_flash_geometry_admitted(..., tokens=bucket)``): wall seconds, peak
   allocated memory, the record (flash_ms, native_ms, rel_l1, max_rel, admitted, cached).
3. The cache dir contents afterwards.

The timer (``_time_rocm_flash_block``) is counted when the module has it. The merge base has no block admission at all:
then nothing is hooked, and the default ``auto`` pick (``select_attention_backend``) is recorded per family instead,
so the base arm reports the default behaviour rather than crashing. Run it twice in a row per state: on the head the
second process must read the file and time nothing."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

WAN = (3072, 24)
ZIMG = (3840, 30)
BUCKETS = {WAN: (1024, 4096, 8192, 16384, 32768, 65536), ZIMG: (1024, 4096, 8192)}


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

    out: dict = {
        "label": args.label,
        "mode": os.environ.get("UNSLOTH_DIFFUSION_ROCM_FLASH", "auto"),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(0),
        "has_block_admission": hasattr(att, "_rocm_flash_block_admitted"),
        "has_timer": hasattr(att, "_time_rocm_flash_block"),
        "has_persistence": hasattr(att, "_rocm_flash_toolchain_key"),
        "has_buckets": hasattr(att, "_rocm_flash_bucket"),
        "probe_version": getattr(att, "ROCM_FLASH_PROBE_VERSION", None),
        "geometries": [],
        "buckets": [],
    }
    calls: list = []
    if out["has_timer"]:
        real_timer = att._time_rocm_flash_block

        def counted(*a, **k):
            calls.append([int(v) for v in a[2:5]] + [int(k.get("tokens", a[5] if len(a) > 5 else -1))])
            return real_timer(*a, **k)

        att._time_rocm_flash_block = counted
    try:
        import flash_attn

        out["flash_attn"] = getattr(flash_attn, "__version__", None)
        out["flash_attn_file"] = flash_attn.__file__
    except Exception as exc:  # noqa: BLE001
        out["flash_attn_error"] = repr(exc)
    if out["has_persistence"]:
        out["toolchain_key"] = att._rocm_flash_toolchain_key("cuda:0")
    target = type("T", (), {"device": "cuda", "torch_device": "cuda:0", "dtype": torch.bfloat16})()

    # what an `auto` load picks before any timing (base: None for both; head: the timed candidate)
    picks = {}
    for fam in ("z-image", "wan2.2-ti2v-5b", "flux.2-klein"):
        try:
            picks[fam] = att.select_attention_backend(target, "auto", speed_active = True, family = fam)
        except TypeError:
            picks[fam] = att.select_attention_backend(target, "auto", speed_active = True)
        except Exception as exc:  # noqa: BLE001
            picks[fam] = f"error: {exc!r}"
    out["auto_pick"] = picks

    def admission(key_prefix):
        recs = getattr(att, "ROCM_FLASH_ADMISSIONS", {})
        return {str(k[-1]): v for k, v in recs.items() if tuple(k[: len(key_prefix)]) == key_prefix}

    if out["has_block_admission"]:
        for hidden, heads in (WAN, ZIMG):
            attn = nn.Module()
            attn.heads = heads
            attn.to_q = nn.Linear(hidden, hidden, bias = False)
            dit = nn.Module()
            dit.attn = attn
            att._attention_dits = lambda p, d = dit: [d]
            n0 = len(calls)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            admitted = att._rocm_flash_block_admitted(object(), target)
            dt = time.perf_counter() - t0
            out["geometries"].append({
                "geometry": [hidden, heads, hidden // heads],
                "admitted": bool(admitted),
                "seconds": round(dt, 4),
                "timer_calls": len(calls) - n0,
                "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "gate_installed": bool(getattr(dit, getattr(att, "ROCM_FLASH_GATE_ATTR", "_none_"), None)),
                "records": admission(("cuda:0", str(torch.bfloat16), hidden, heads, hidden // heads)),
            })
    # Lazy per-bucket timing, as the length gate calls it (auto mode only: the gate is installed only by an auto load)
    if out["has_buckets"] and att._rocm_flash_mode() == "auto":
        for (hidden, heads), buckets in BUCKETS.items():
            for tokens in buckets:
                n0 = len(calls)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base_alloc = torch.cuda.memory_allocated()
                t0 = time.perf_counter()
                err = None
                try:
                    ok = att._rocm_flash_geometry_admitted(
                        "cuda:0", torch.bfloat16, (hidden, heads, hidden // heads), tokens = tokens
                    )
                except Exception as exc:  # noqa: BLE001
                    ok, err = None, repr(exc)
                dt = time.perf_counter() - t0
                rec = att.ROCM_FLASH_ADMISSIONS.get(
                    ("cuda:0", str(torch.bfloat16), hidden, heads, hidden // heads, tokens)
                )
                out["buckets"].append({
                    "geometry": [hidden, heads, hidden // heads],
                    "bucket": tokens,
                    "admitted": ok,
                    "error": err,
                    "seconds": round(dt, 3),
                    "timer_calls": len(calls) - n0,
                    "peak_alloc_gib": round((torch.cuda.max_memory_allocated() - base_alloc) / 2**30, 3),
                    "record": rec,
                })
                out["timer_calls_so_far"] = len(calls)
                args.out.write_text(json.dumps(out, indent = 1, default = str), encoding = "utf-8")
    out["timer_calls_total"] = len(calls) if out["has_timer"] else None
    out["timer_call_args"] = calls
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
