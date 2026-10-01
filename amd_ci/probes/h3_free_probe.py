#!/usr/bin/env python3
"""Probe: what free VRAM does this checkout's MiniMax-H3 resident check read, and what does the driver say?

Observes only (pairs with criteria/h3_free_not_overestimated.py). Records, for one checkout:
  - torch.cuda.mem_get_info() free / total, taken as a bystander (nothing allocated here);
  - `_h3_card_free_bytes("cuda", None)` and `("cuda", 0)` from core/inference/video.py, read
    BEFORE and AFTER this process attaches a HIP context, so both sides of the comparison can
    be matched against a driver figure that includes the same context;
  - the raw `utils.hardware.get_visible_gpu_utilization()` devices the function reads;
  - `h3_native_render_flags(...)` for a 24 GiB file set at 960x544x124 under memory auto.
A checkout without the function (the merge base) is recorded as `available: false`, not an error.

`--baseline-obs` names a finished observations.json (the same probe, no holder); its reading for
this state is copied in verbatim so a criteria module can judge the drop without observing.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

GIB = 1024 ** 3


def _import_backend(checkout: Path):
    backend = checkout / "studio" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"no backend at {backend}")
    sys.path.insert(0, str(backend))


def _h3_reads(video) -> dict:
    out = {}
    for key, ordinal in (("none", None), ("zero", 0)):
        try:
            v = video._h3_card_free_bytes("cuda", ordinal)
            out[key] = v
        except Exception as e:  # noqa: BLE001
            out[key] = None
            out[f"{key}_error"] = f"{type(e).__name__}: {e}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--baseline-obs", type = Path, default = None)
    args = ap.parse_args()

    obs: dict = {"state": args.state}
    _import_backend(args.checkout)

    video = None
    h3 = None
    try:
        import core.inference.video as video  # noqa: PLC0415
        obs["video_file"] = video.__file__
    except Exception as e:  # noqa: BLE001
        obs["video_import_error"] = f"{type(e).__name__}: {e}"
        obs["video_import_tb"] = traceback.format_exc()[-2000:]
    try:
        import core.inference.video_minimax_h3 as h3  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        obs["h3_import_error"] = f"{type(e).__name__}: {e}"

    available = video is not None and hasattr(video, "_h3_card_free_bytes")
    obs["h3_free_fn_available"] = available

    # Out-of-process read first, before this process holds any HIP context.
    if available:
        obs["h3_free_pre_ctx"] = _h3_reads(video)

    try:
        import torch  # noqa: PLC0415
        free_b, total_b = torch.cuda.mem_get_info()
        props = torch.cuda.get_device_properties(0)
        obs["driver_free_bytes"] = int(free_b)
        obs["driver_total_bytes"] = int(total_b)
        obs["props_total_bytes"] = int(props.total_memory)
        obs["is_integrated"] = getattr(props, "is_integrated", None)
        obs["arch"] = getattr(props, "gcnArchName", None)
        obs["device_name"] = torch.cuda.get_device_name(0)
        obs["torch_hip"] = getattr(torch.version, "hip", None)
        obs["observer_allocated_bytes"] = int(torch.cuda.memory_allocated())
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"

    if available:
        obs["h3_free_post_ctx"] = _h3_reads(video)
        try:
            # Re-read the driver right after, so the pair brackets the same moment.
            import torch  # noqa: PLC0415
            obs["driver_free_bytes_after"] = int(torch.cuda.mem_get_info()[0])
        except Exception as e:  # noqa: BLE001
            obs["driver_after_error"] = f"{type(e).__name__}: {e}"

    try:
        from utils.hardware import get_visible_gpu_utilization  # noqa: PLC0415
        try:
            from utils.hardware import gpu_query  # noqa: PLC0415
            ctx = gpu_query.fresh_reads()
        except Exception:  # noqa: BLE001
            import contextlib  # noqa: PLC0415
            ctx = contextlib.nullcontext()
        with ctx:
            obs["visible_gpu_utilization"] = get_visible_gpu_utilization()
    except Exception as e:  # noqa: BLE001
        obs["utilization_error"] = f"{type(e).__name__}: {e}"

    have_flags = h3 is not None and hasattr(h3, "h3_native_render_flags") \
        and hasattr(h3, "h3_native_resident_bytes")
    obs["h3_flags_fn_available"] = have_flags
    if have_flags and available:
        try:
            need = h3.h3_native_resident_bytes(int(24 * 2 ** 30), 960, 544, 124)
            free_read = (obs.get("h3_free_post_ctx") or {}).get("none")
            flags, resident = h3.h3_native_render_flags(
                ["--offload-to-cpu", "--diffusion-fa"], memory_mode = "auto",
                free_bytes = free_read, need_bytes = need, env = {})
            obs["need_bytes"] = int(need)
            obs["flags"] = flags
            obs["resident"] = bool(resident)
            # The decision the driver's own figure would have produced, for contrast.
            dflags, dres = h3.h3_native_render_flags(
                ["--offload-to-cpu", "--diffusion-fa"], memory_mode = "auto",
                free_bytes = obs.get("driver_free_bytes_after", obs.get("driver_free_bytes")),
                need_bytes = need, env = {})
            obs["driver_resident"] = bool(dres)
        except Exception as e:  # noqa: BLE001
            obs["flags_error"] = f"{type(e).__name__}: {e}"

    if args.baseline_obs is not None:
        try:
            prior = json.loads(args.baseline_obs.read_text(encoding = "utf-8"))
            obs["baseline"] = prior.get(args.state)
        except Exception as e:  # noqa: BLE001
            obs["baseline_error"] = f"{type(e).__name__}: {e}"

    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
