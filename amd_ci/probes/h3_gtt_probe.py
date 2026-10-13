#!/usr/bin/env python3
"""Probe: the real amdgpu GTT pool as sysfs reports it, and what this checkout's H3 native fit does with it.

Observes only; criteria/h3_gtt_fit.py judges. Sizes are the MiniMax-H3 Q4 split scaled to the live free GTT, so the
over-pool case reproduces the reported 31 GiB-pinned-into-23 GiB shape on whatever pool this box has."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

GIB = 1024**3
Q4 = (10.64, 16.97, 5.41)  # dit, te, vae GiB (MiniMax-H3 Q4)
AUTO = ("--offload-to-cpu", "--diffusion-fa", "--max-vram", "-1")


def _int(path: Path):
    try:
        return int(path.read_text(encoding = "utf-8").strip())
    except (OSError, ValueError):
        return None


def raw_cards() -> list:
    out = []
    for card in sorted(Path("/sys/class/drm").glob("card*")):
        dev = card / "device"
        if not (dev / "mem_info_gtt_total").exists():
            continue
        try:
            vendor = (dev / "vendor").read_text(encoding = "utf-8").strip()
        except OSError:
            vendor = None
        out.append(
            {
                "card": card.name,
                "pci_slot": os.path.basename(os.path.realpath(dev)),
                "vendor": vendor,
                "gtt_total": _int(dev / "mem_info_gtt_total"),
                "gtt_used": _int(dev / "mem_info_gtt_used"),
                "vram_total": _int(dev / "mem_info_vram_total"),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state, "raw_cards": raw_cards()}
    try:
        obs["proc_meminfo_memtotal_kib"] = int(
            next(l for l in open("/proc/meminfo", encoding = "utf-8") if l.startswith("MemTotal")).split()[1]
        )
    except Exception as e:  # noqa: BLE001
        obs["meminfo_error"] = repr(e)
    backend = args.checkout / "studio" / "backend"
    sys.path.insert(0, str(backend))
    try:
        from core.inference import video_minimax_h3 as h3  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        obs["import_error"] = f"{type(e).__name__}: {e}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    obs["has_gtt_fit"] = hasattr(h3, "amdgpu_gtt_reading") and hasattr(h3, "h3_native_memory_fit")
    if obs["has_gtt_fit"]:
        free, known = h3.amdgpu_gtt_reading()
        obs["reading_any"] = {"free": free, "use_known": known}
        obs["reading_per_slot"] = {
            c["pci_slot"]: list(h3.amdgpu_gtt_reading(c["pci_slot"])) for c in obs["raw_cards"]
        }
        try:
            obs["host_budgets"] = list(h3.h3_native_host_budgets())
        except Exception as e:  # noqa: BLE001
            obs["host_budgets_error"] = repr(e)
        try:
            from core.inference.sd_cpp_backend import rocm_runtime_missing_sonames  # noqa: PLC0415

            obs["rocm_missing_sonames"] = rocm_runtime_missing_sonames()
        except Exception as e:  # noqa: BLE001
            obs["rocm_missing_error"] = repr(e)
        if free:
            fits = {}
            for label, scale in (("over_pool", 1.4), ("under_pool", 0.5)):
                k = scale * free / (sum(Q4) * GIB)
                sizes = {n: int(v * GIB * k) for n, v in zip(("dit_bytes", "te_bytes", "vae_bytes"), Q4)}
                fit = h3.h3_native_memory_fit(
                    AUTO,
                    accelerator = "vulkan",
                    memory_mode = "auto",
                    pinned_free_bytes = free,
                    host_ram_bytes = 8 * free,
                    host_swap_bytes = 0,
                    env = {},
                    **sizes,
                )
                fits[label] = {
                    "pinned_bytes": sum(sizes.values()),
                    "flags": list(fit.flags),
                    "note": fit.note,
                    "refusal": fit.refusal,
                }
            obs["fits"] = fits
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
