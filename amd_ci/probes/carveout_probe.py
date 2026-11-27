#!/usr/bin/env python3
"""Probe (#19768): what the unified-memory torch probe reports on a Windows gfx1151 APU,
next to the raw readings it is built from. Observes only. On states whose checkout has the
carve-out reader it also allocates (and touches) the reported usable MiB minus a margin."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MIB = 1024 * 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state, "platform": sys.platform}

    def save():
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "psutil", "httpx", "structlog"],
                       capture_output = True, text = True, encoding = "utf-8", errors = "replace")
    obs["pip_rc"] = r.returncode
    try:
        import torch

        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        obs["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            obs["device"] = torch.cuda.get_device_name(0)
            obs["arch"] = getattr(p, "gcnArchName", None)
            f, t = torch.cuda.mem_get_info(0)
            obs["hip_free_mib"], obs["hip_total_mib"] = f // MIB, t // MIB
    except Exception as e:
        obs["torch_error"] = f"{type(e).__name__}: {e}"
    save()

    backend = args.checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    try:
        import psutil

        vm = psutil.virtual_memory()
        obs["host_total_mib"], obs["host_avail_mib"] = vm.total // MIB, vm.available // MIB
        from core.inference.llama_cpp import LlamaCppBackend as B

        carve = B._igpu_dedicated_memory_bytes(rocm_fallback = False)
        obs["registry_carve_out_mib"] = None if carve is None else int(carve) // MIB
        t0 = time.monotonic()
        used = B._windows_apu_gpu_used_bytes()
        obs["wddm_used_s"] = round(time.monotonic() - t0, 3)
        obs["wddm_used_mib"] = None if used is None else int(used) // MIB
        obs["has_reader"] = hasattr(B, "_windows_apu_carve_out_free_mib")
        if obs["has_reader"]:
            B._WINDOWS_APU_CARVE_OUT_FREE = None
            t0 = time.monotonic()
            obs["reader_mib"] = B._windows_apu_carve_out_free_mib()
            obs["reader_cold_s"] = round(time.monotonic() - t0, 3)
            t0 = time.monotonic()
            B._windows_apu_carve_out_free_mib()
            obs["reader_warm_s"] = round(time.monotonic() - t0, 4)
            B._WINDOWS_APU_CARVE_OUT_FREE = None
        obs["unified_ids"] = sorted(B._rocm_unified_memory_gpu_ids())
        times = []
        rows = None
        for _ in range(3):
            t0 = time.monotonic()
            rows = B._get_gpu_memory_torch(None, False)
            times.append(round(time.monotonic() - t0, 3))
        obs["probe_rows"] = rows
        obs["probe_s"] = times
        obs["usable_mib"] = rows[0][1] if rows else None
    except Exception as e:
        import traceback

        obs["backend_error"] = f"{type(e).__name__}: {e}"
        obs["backend_tb"] = traceback.format_exc()[-2000:]
    save()

    # Allocate what the head now publishes, minus 1 GiB: proves the credited carve-out is real.
    if obs.get("has_reader") and obs.get("usable_mib") and obs.get("cuda_available"):
        import psutil
        import torch

        target = max(0, int(obs["usable_mib"]) - 1024)
        chunks, got, err = [], 0, None
        min_avail = obs.get("host_avail_mib")
        try:
            while got + 1024 <= target:
                x = torch.empty(1024 * MIB, dtype = torch.uint8, device = "cuda")
                x.fill_(1)
                chunks.append(x)
                got += 1024
                min_avail = min(min_avail, psutil.virtual_memory().available // MIB)
            torch.cuda.synchronize()
            ok = bool(chunks) and all(int(c[-1].item()) == 1 for c in chunks[-2:])
        except Exception as e:
            err, ok = f"{type(e).__name__}: {str(e)[:300]}", False
        obs["alloc"] = {"target_mib": target, "allocated_mib": got, "ok": ok and got + 1024 > target,
                        "error": err, "min_host_avail_mib": min_avail}
        try:
            used = B._windows_apu_gpu_used_bytes()
            obs["alloc"]["wddm_used_at_peak_mib"] = None if used is None else int(used) // MIB
        except Exception:
            pass
        del chunks
        torch.cuda.empty_cache()
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
