#!/usr/bin/env python3
"""Probe: what does a coarse FP8 expert scale do to the Triton dequant path here?

The question this exists to answer is whether Triton's maximum tensor numel is
the same on ROCm as on CUDA. It is enforced by validate_block_shape() in
Triton's PYTHON frontend, before a backend is chosen, so it should be, but
unsloth-zoo#1321 turns on that number and "should be" is not a measurement.

So: read the constant out of this machine's Triton, then actually drive the
path. A per-expert per-tensor scale (p == q == 1) derives BLOCK_SIZE = M, so an
M x N tile of 2048 x 2048 is 4x over the cap and the kernel cannot compile.

Observes only. Whether "raised" at the base and "declined" at the head is the
expected pair is criteria/fp8_tile_limit_declines.py's business.

Pairs with criteria/fp8_tile_limit_declines.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Deliberately NOT setting UNSLOTH_ZOO_DISABLE_GPU_INIT. It is the zoo's opt-out
# for its "unsloth must be importable" guard and it reads like a harmless CI
# convenience, but measured here it also leaves unsloth's kernels decorated with
# a passthrough `triton.jit`: weight_dequant_kernel stays a plain function, so
# `kernel[grid](...)` raises "'function' object is not subscriptable" at EVERY
# tile size. That looks like the defect while proving nothing about the tile
# limit, and it would have made both states fail identically. unsloth is
# importable here (the gate below insists on it), so the guard is satisfied
# without the flag.

# A tile at the cap exactly, which must stay on the Triton path at both states:
# without it a probe reporting "declined" everywhere would look like a fix.
LEGAL = 1024
# The first size comfortably over it. 2048 x 2048 is 4 x 1048576.
OVERSIZED = 2048


def import_zoo_module(checkout: Path):
    """Import THIS checkout's moe_utils_fp8, not the installed zoo."""
    if not (checkout / "unsloth_zoo").is_dir():
        raise SystemExit(f"no unsloth_zoo package at {checkout}")
    sys.path.insert(0, str(checkout))
    for stale in [m for m in sys.modules if m == "unsloth_zoo" or m.startswith("unsloth_zoo.")]:
        del sys.modules[stale]
    from unsloth_zoo.temporary_patches import moe_utils_fp8  # noqa: PLC0415
    return moe_utils_fp8


def read_triton_limit(obs: dict) -> None:
    """Triton's own cap, as this machine's Triton reports it."""
    try:
        import triton  # noqa: PLC0415
        obs["triton_version"] = getattr(triton, "__version__", None)
    except Exception as e:  # noqa: BLE001
        obs["triton_error"] = f"{type(e).__name__}: {e}"
        return
    for module_name in ("triton.language", "triton._utils"):
        try:
            import importlib  # noqa: PLC0415
            value = getattr(
                importlib.import_module(module_name), "TRITON_MAX_TENSOR_NUMEL", None
            )
        except Exception as e:  # noqa: BLE001
            obs.setdefault("triton_limit_errors", {})[module_name] = f"{type(e).__name__}: {e}"
            continue
        # Recorded as a type name too: with no GPU visible unsloth installs a
        # Triton stub whose attributes are placeholders, and a placeholder read
        # as a number is exactly the confusion this probe must not create.
        obs.setdefault("triton_limits", {})[module_name] = {
            "value": value if type(value) is int else repr(value),
            "type": type(value).__name__,
        }


def attempt(module, M: int, N: int) -> dict:
    """Drive the Triton dequant path for one tile size and report what happened."""
    import torch  # noqa: PLC0415

    try:
        weight = torch.zeros(2, M, N, device = "cuda").to(torch.float8_e4m3fn)
        scale = torch.full((2, 1, 1), 0.05, device = "cuda", dtype = torch.float32)
    except Exception as e:  # noqa: BLE001
        return {"outcome": "setup_failed", "error": f"{type(e).__name__}: {e}"}

    try:
        out = module._dequantize_full_expert_weights_unsloth(weight, scale, torch.bfloat16)
    except Exception as e:  # noqa: BLE001
        # The defect: Triton refuses to compile the kernel. Kept as text rather
        # than judged here, so the criteria can insist it is the numel error and
        # not some unrelated ROCm failure that happens to look like one.
        return {"outcome": "raised", "error": f"{type(e).__name__}: {e}"[:2000]}
    if out is None:
        return {"outcome": "declined"}
    return {
        "outcome": "returned",
        "shape": list(out.shape),
        "dtype": str(out.dtype),
        # The path is only meaningful if the numbers are right; zeros times a
        # scale is zeros, so this is a shape-and-finiteness check, not accuracy.
        "all_finite": bool(torch.isfinite(out).all().item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "oversized_mn": OVERSIZED, "legal_mn": LEGAL}

    try:
        import torch
        obs["torch_version"] = torch.__version__
        obs["torch_hip"] = getattr(torch.version, "hip", None)
        obs["torch_cuda"] = getattr(torch.version, "cuda", None)
        obs["gpu_available"] = bool(torch.cuda.is_available())
        if obs["gpu_available"]:
            obs["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    read_triton_limit(obs)

    # Whether the kernel this path depends on is importable at all. Without it
    # _dequantize_full_expert_weights_unsloth returns None for EVERY input, so
    # the base would "decline" the oversized tile and the differential would be
    # vacuous. The criteria gates on this rather than letting it pass as a fix.
    try:
        from unsloth.kernels.fp8 import weight_dequant_block  # noqa: F401, PLC0415
        obs["unsloth_fp8_kernel"] = True
    except Exception as e:  # noqa: BLE001
        obs["unsloth_fp8_kernel"] = False
        obs["unsloth_fp8_kernel_error"] = f"{type(e).__name__}: {e}"

    try:
        module = import_zoo_module(args.checkout)
        obs["module_file"] = module.__file__
        obs["has_resolver"] = hasattr(module, "_triton_max_tensor_numel")
        obs["module_limit"] = (
            module._triton_max_tensor_numel() if obs["has_resolver"]
            else getattr(module, "_TRITON_MAX_TENSOR_NUMEL", None)
        )
    except Exception as e:  # noqa: BLE001
        obs["module_error"] = f"{type(e).__name__}: {e}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    obs["oversized"] = attempt(module, OVERSIZED, OVERSIZED)
    obs["legal"] = attempt(module, LEGAL, LEGAL)

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
