#!/usr/bin/env python3
"""Probe: can Windows-native reach the line unsloth-zoo#887 changes?

OBSERVES ONLY. It answers a reachability question, not a correctness one.

The changed line lives in `forward_triton_grouped_gemm`, which runs only when
`select_moe_backend()` picks `unsloth_triton`. That needs a GPU-enabled torch AND a
working Triton. If either is missing on Windows-native, the line never executes there
and the PR cannot change Windows behaviour - but that has to be measured, because
"surely Triton does not work there" is an assumption, and assumptions are how a
platform claim goes wrong.

Every check is independently guarded: one missing piece must not stop the rest from
reporting, or a single ImportError would look like a total absence of evidence.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def _try(fn, default=None):
    try:
        return fn()
    except Exception as e:
        return f"{type(e).__name__}: {str(e).splitlines()[0][:200]}" if default is None else default


def collect() -> dict:
    obs: dict = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }

    try:
        import torch
    except Exception as e:
        obs["torch"] = f"NOT IMPORTABLE: {type(e).__name__}: {e}"
        obs["reachable"] = False
        obs["why"] = "torch is not importable on this Windows host"
        return obs

    obs["torch"] = torch.__version__
    obs["torch_hip"] = getattr(torch.version, "hip", None)
    obs["torch_cuda"] = getattr(torch.version, "cuda", None)
    obs["cuda_available"] = _try(lambda: bool(torch.cuda.is_available()), False)
    obs["device_count"] = _try(lambda: int(torch.cuda.device_count()), 0)
    obs["device_name"] = _try(lambda: torch.cuda.get_device_name(0)) \
        if obs["cuda_available"] else None

    # Triton is the hard requirement: the grouped-GEMM MoE kernels are Triton kernels.
    def _triton():
        import triton
        return getattr(triton, "__version__", "unknown")
    obs["triton"] = _try(_triton)

    def _kernels():
        import unsloth  # noqa: F401  (must precede transformers; sets UNSLOTH_IS_PRESENT)
        from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (  # noqa: F401
            KernelConfigForward)
        return "importable"
    obs["moe_kernels"] = _try(_kernels)

    def _backend():
        import unsloth  # noqa: F401
        from unsloth_zoo.temporary_patches import moe_utils as mu
        return {
            "selected": str(mu.select_moe_backend()),
            "grouped_mm_supported": _try(
                lambda: bool(mu._check_torch_grouped_mm_supported())),
        }
    obs["backend"] = _try(_backend)

    sel = (obs.get("backend") or {})
    sel = sel.get("selected") if isinstance(sel, dict) else None

    # Two DIFFERENT questions, kept apart on purpose. Collapsing them would report a
    # host as unaffected merely because it prefers another backend, when any user
    # setting UNSLOTH_MOE_BACKEND=unsloth_triton - and every torch 2.8 host, where
    # Triton is the default on cards the grouped_mm gate rejects - still runs the line.
    obs["triton_backend_usable"] = bool(
        obs["cuda_available"]
        and isinstance(obs.get("triton"), str) and "Error" not in str(obs.get("triton"))
        and obs.get("moe_kernels") == "importable"
    )
    obs["triton_is_default"] = (sel == "unsloth_triton")
    obs["selected_backend"] = sel

    obs["reachable"] = obs["triton_backend_usable"]
    obs["why"] = (
        f"the Triton grouped-GEMM MoE backend is usable here (default backend is "
        f"{sel!r}), so the changed line runs whenever it is selected"
        if obs["reachable"] else
        f"cuda_available={obs['cuda_available']}, triton={obs.get('triton')}, "
        f"moe_kernels={obs.get('moe_kernels')}, selected_backend={sel}"
    )
    return obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    obs = {}
    try:
        obs = collect()
    except Exception as e:
        import traceback
        obs = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-1500:]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obs, indent=2), encoding="utf-8")
    # Reporting a negative reachability result is a successful probe run.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
