#!/usr/bin/env python3
"""Probe: on THIS host, is the host-pinned embedding discount withheld?

PR 9931 subtracts the host-pinned embeddings (token_embd, per_layer_token_embd)
from a DISCRETE device's VRAM budget. On an integrated GPU the "VRAM" and the
host are one pool, so the discount must be zero or the budget under-counts.
This observes, on the live gfx1151, every reading that decision is built from:

  props               gcnArchName / is_integrated as the ROCm torch wheel reports them
  rocm_classify       what _rocm_classify_unified_memory(props) says (arch, unified)
  classification_known  _torch_unified_memory_classification_known(None), the new
                      "was every device classifiable" seam the PR adds
  apu_wants_unified   _amd_apu_wants_unified_memory(None), the amd-smi / HIP detector
  shared_memory       the production expression's in-process terms, composed the
                      way load_model composes them
  discount_candidate  _host_pinned_vram_discount with shared_memory=False and a
                      PATCHED 1 GiB weight reading -- proves the discount path is live
  discount_applied    the same call with the derived shared_memory -- must be 0 here

The candidate is synthetic on purpose: a real GGUF would make this a test of
whichever file is on the runner's disk. Observes only; never judges.

Pairs with criteria/no_discount_on_unified_pool.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GIB = 1024 ** 3


def _load(checkout: Path):
    backend = checkout / "studio" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"no backend at {backend}")
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules
                  if m.startswith(("core.", "utils.", "core", "utils", "loggers"))]:
        del sys.modules[stale]
    return backend


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state}

    props = None
    try:
        import torch
        obs["torch_version"] = getattr(torch, "__version__", None)
        obs["torch_hip"] = getattr(getattr(torch, "version", None), "hip", None)
        obs["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            obs["arch"] = getattr(props, "gcnArchName", None)
            obs["has_is_integrated_attr"] = hasattr(props, "is_integrated")
            obs["is_integrated"] = int(getattr(props, "is_integrated", 0) or 0)
            obs["total_gib"] = props.total_memory / GIB
            obs["n_gpus"] = torch.cuda.device_count()
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        _load(args.checkout)
    except SystemExit as e:
        obs["error"] = str(e)
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    # The per-device classifier (training/worker.py). Present on base and head.
    try:
        from core.training.worker import _rocm_classify_unified_memory  # noqa: PLC0415
        if props is not None:
            arch, unified = _rocm_classify_unified_memory(props)
            obs["rocm_classify"] = {"arch": arch, "unified": bool(unified)}
    except Exception as e:  # noqa: BLE001
        obs["rocm_classify_error"] = f"{type(e).__name__}: {e}"

    try:
        from core.inference.llama_cpp import LlamaCppBackend as B  # noqa: PLC0415
        from utils.hardware import is_apple_silicon  # noqa: PLC0415

        known_fn = getattr(B, "_torch_unified_memory_classification_known", None)
        disc_fn = getattr(B, "_host_pinned_vram_discount", None)
        obs["has_classification_seam"] = bool(known_fn)
        obs["has_discount"] = bool(disc_fn)

        wants = getattr(B, "_amd_apu_wants_unified_memory", None)
        icuda = getattr(B, "_integrated_cuda_unified_memory", None)
        obs["apu_wants_unified"] = bool(wants(None)) if wants else None
        obs["integrated_cuda"] = bool(icuda(None)) if icuda else None
        obs["classification_known"] = bool(known_fn(None)) if known_fn else None
        obs["apple"] = bool(is_apple_silicon())

        # The in-process terms of load_model's _shared_memory expression. The
        # Vulkan and `bool(_gpu_mem)` terms are not reachable without launching a
        # binary; on a ROCm torch host the memory reading is non-empty, so the
        # classification term applies as written.
        if known_fn:
            shared = bool(
                obs["apple"] or obs["apu_wants_unified"] or obs["integrated_cuda"]
                or (not obs["classification_known"])
            )
            obs["shared_memory"] = shared

        if disc_fn:
            # Patch the weight reading so the discount path is exercised without a
            # GGUF on disk. 1 GiB is unmistakable.
            B._host_pinned_weight_bytes = staticmethod(lambda _p: GIB)
            obs["discount_candidate"] = int(disc_fn("synthetic.gguf", None, shared_memory = False))
            obs["discount_applied"] = int(disc_fn("synthetic.gguf", None, shared_memory = obs.get("shared_memory", True)))
    except ImportError as e:
        obs["import_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        obs["backend_error"] = f"{type(e).__name__}: {e}"

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
