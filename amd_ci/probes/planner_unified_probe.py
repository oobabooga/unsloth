#!/usr/bin/env python3
"""Probe: on THIS host, does the offload planner see a unified-memory APU?

Observes only. Three readings per state, all from the checkout under test:

  detector  -- what LlamaCppBackend's own ROCm unified-memory detection returns
               on the real hardware. This is the reading that cannot be faked in
               a unit test: it walks amd-smi / HIP on the live GPU.
  unified   -- what the planner decides for a synthetic layout when that
               detected flag is passed through, i.e. the production wiring.
  discrete  -- the SAME layout and budget with unified_memory forced False.

The third is what makes the second mean anything. A layout that fits anyway
would "abstain" on any host, so the discrete reading has to show a real spill
for the unified reading's silence to be attributable to the flag.

The layout is synthetic on purpose: a GGUF would make this a test of file IO and
of whichever model happened to be on the runner's disk.

Pairs with criteria/planner_declines_on_unified.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GIB = 1024 ** 3


def _load(checkout: Path):
    """Import THIS checkout's inference modules, the way the app does."""
    backend = checkout / "studio" / "backend"
    if not backend.is_dir():
        raise SystemExit(f"no backend at {backend}")
    sys.path.insert(0, str(backend))
    for stale in [m for m in sys.modules
                  if m.startswith(("core.", "utils.", "core", "utils"))]:
        del sys.modules[stale]
    return backend


def _synthetic_layout(mod):
    """A 64-block model too big for the budget below, built in memory."""
    block = mod.BlockLayout
    blocks = tuple(
        block(index = i, spillable_bytes = 192 * 1024 ** 2, resident_bytes = 48 * 1024 ** 2)
        for i in range(64)
    )
    return mod.ModelLayout(
        arch = "llama",
        n_layers = 64,
        n_attention_layers = 64,
        blocks = blocks,
        lm_head_bytes = 512 * 1024 ** 2,
        token_embd_bytes = 512 * 1024 ** 2,
        other_resident_bytes = 32 * 1024 ** 2,
        kv_bytes_per_token_f16 = 65536,
        recurrent_bytes = 0,
        n_ctx_train = 131072,
        complete = True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state}

    try:
        import torch
        obs["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            obs["arch"] = getattr(props, "gcnArchName", None)
            obs["is_integrated"] = bool(getattr(props, "is_integrated", False))
            obs["total_gib"] = props.total_memory / GIB
            free_b, total_b = torch.cuda.mem_get_info()
            obs["driver_free_gib"] = free_b / GIB
            obs["n_gpus"] = torch.cuda.device_count()
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        _load(args.checkout)
    except SystemExit as e:
        obs["error"] = str(e)
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    # The detector. Absent at any state that predates the planner, which is a
    # real observation about that state and not an error.
    try:
        from core.inference.llama_cpp import LlamaCppBackend  # noqa: PLC0415
        ids = getattr(LlamaCppBackend, "_rocm_unified_memory_gpu_ids", None)
        wants = getattr(LlamaCppBackend, "_amd_apu_wants_unified_memory", None)
        obs["has_detector"] = bool(ids and wants)
        if ids:
            obs["unified_gpu_ids"] = sorted(ids())
        if wants:
            obs["detector_says_unified"] = bool(wants(None))
    except Exception as e:  # noqa: BLE001
        obs["detector_error"] = f"{type(e).__name__}: {e}"

    # The planner, under the flag the detector just produced and under its
    # opposite. Absent before this change; recorded as such.
    try:
        import core.inference.offload_planner as pl  # noqa: PLC0415
        import core.inference.offload_layout as lay  # noqa: PLC0415
        from core.inference.offload_cost_model import HostProfile  # noqa: PLC0415

        class _Mod:
            BlockLayout = lay.BlockLayout
            ModelLayout = lay.ModelLayout

        layout = _synthetic_layout(_Mod)
        obs["has_planner"] = True
        budget = [8 * GIB]
        host_ram = 64 * GIB
        for name, unified in (("unified", True), ("discrete", False)):
            plan = pl.plan_placement(
                layout,
                budget,
                int(host_ram),
                8192,
                opts = pl.PlanOptions(host = HostProfile(unified_memory = unified)),
            )
            obs[name] = {
                "changed": bool(plan.changed),
                "n_spilled_blocks": len(plan.spilled_blocks),
                "spilled_lm_head": bool(plan.spilled_lm_head),
                "insufficient": bool(plan.insufficient),
                "reason": plan.reason,
            }
    except ImportError:
        obs["has_planner"] = False
    except Exception as e:  # noqa: BLE001
        obs["planner_error"] = f"{type(e).__name__}: {e}"

    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
