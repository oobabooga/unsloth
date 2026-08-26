#!/usr/bin/env python3
"""Probe: with the planner now DEFAULT-ON, is a unified-memory APU still spared?

Observes only. The existing planner_unified_probe calls ``plan_placement``
directly with a synthetic ``HostProfile``, which exercises the planner's
arithmetic but never the launch seam. The abstain that actually protects this
hardware lives in ``_planned_tensor_spill``, and the gate that decides whether
the seam runs at all lives in ``smart_offload_enabled``. Neither is reached by
that probe, so it cannot answer the question this one asks.

Readings, all from the checkout under test:

  gate      -- smart_offload_enabled({}) with NOTHING in the environment. On a
               default-on build this is True. This is what makes the rest
               meaningful: if the gate is off, the seam declines for a reason
               that has nothing to do with the hardware.
  detector  -- the live ROCm unified-memory detection on the real GPU.
  default   -- the seam, called with env={} and no arguments, i.e. exactly what
               a user gets by typing nothing. THE reading this probe exists for.
  discrete  -- the SAME inputs with the APU detector forced False.

``discrete`` is the non-vacuity control: a load that fits anyway abstains on any
host, so it has to spill for ``default``'s silence to be about the hardware.

The layout is synthetic on purpose: a GGUF would make this a test of file IO and
of whichever model happened to be on the runner's disk.

Pairs with criteria/planner_spares_unified_by_default.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GIB = 1024 ** 3
MIB = 1024 ** 2


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


def _inputs(free_mib: int) -> dict:
    """The seam's input dict: a 30 GiB load against a budget that cannot hold it."""
    return {
        "model_size": 30 * GIB,
        "kv_cache_bytes": 2 * GIB,
        "gpus": [(0, free_mib)],
        "gpu_usable_mib": {},
        "compute_buffer_flat": 0,
        "ctx_compute_per_device": 0,
        "extra_gpu_bytes": 0,
        "gpu_indices": None,
        "soft_overhead": 0,
        "model_path": "/models/synthetic.gguf",
        "n_ctx": 32768,
        "n_parallel": 1,
        "shared_gpu_ids": set(),
    }


def _stub_class(backend_cls, layout_mod, unified_answer):
    """A minimal host for the seam, borrowing the REAL method under test.

    ``unified_answer`` is None to use the live detector and False to force the
    discrete control. Everything else is fixed so the two readings differ by
    exactly that one answer.
    """

    class _Stub:
        _PIPELINE_PER_DEVICE_OVERHEAD_MIB = 1024
        _HOST_RAM_HEADROOM_MIB = 2048
        _excluded_bytes = 0
        n_moe_layers = 0

        def _amd_apu_wants_unified_memory(self, gpu_indices = None):
            if unified_answer is None:
                # A staticmethod on the backend, so it takes no self.
                return bool(backend_cls._amd_apu_wants_unified_memory(gpu_indices))
            return unified_answer

        # The CUDA-SoC twin of the above (Jetson / DGX Spark). Live on the
        # default reading like its AMD counterpart; on ROCm it answers False,
        # so an abstain here is attributable to the APU path.
        def _integrated_cuda_unified_memory(self, gpu_indices = None):
            if unified_answer is None:
                # A staticmethod on the backend, so it takes no self.
                fn = getattr(backend_cls, "_integrated_cuda_unified_memory", None)
                return bool(fn(gpu_indices)) if fn else False
            return unified_answer

        def _available_system_memory_mib(self):
            return 64 * 1024

        def _can_estimate_kv(self):
            return True

        def _tensor_spill_layout(self, model_path):
            n = 64
            return layout_mod.ModelLayout(
                arch = "llama",
                n_layers = n,
                n_attention_layers = 16,
                blocks = tuple(
                    layout_mod.BlockLayout(
                        index = i,
                        spillable_bytes = (10 * GIB) // n,
                        resident_bytes = (2 * GIB) // n,
                    )
                    for i in range(n)
                ),
                lm_head_bytes = 1 * GIB,
                token_embd_bytes = 512 * MIB,
                kv_bytes_per_token_f16 = 65536,
                n_ctx_train = 262144,
                complete = True,
            )

    _Stub._planned_tensor_spill = backend_cls._planned_tensor_spill
    return _Stub


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
            obs["n_gpus"] = torch.cuda.device_count()
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        _load(args.checkout)
    except SystemExit as e:
        obs["error"] = str(e)
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    # The gate, with NOTHING in the environment. False here at a state that
    # predates the default-on flip, which is a real observation about that
    # state rather than an error.
    try:
        from core.inference.offload_planner import smart_offload_enabled  # noqa: PLC0415
        obs["has_gate"] = True
        obs["gate_default"] = bool(smart_offload_enabled({}))
    except ImportError:
        obs["has_gate"] = False
    except Exception as e:  # noqa: BLE001
        obs["gate_error"] = f"{type(e).__name__}: {e}"

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

    # The seam itself, through the default path. env={} is the whole point:
    # nothing set, which on a default-on build means the planner runs.
    try:
        import core.inference.offload_layout as lay  # noqa: PLC0415
        from core.inference.llama_cpp import LlamaCppBackend  # noqa: PLC0415

        obs["has_seam"] = True
        for name, answer in (("default", None), ("discrete", False)):
            stub = _stub_class(LlamaCppBackend, lay, answer)()
            plan = stub._planned_tensor_spill(_inputs(14 * 1024), env = {})
            if plan is None:
                obs[name] = {"planned": False}
                continue
            obs[name] = {
                "planned": True,
                "changed": bool(plan.changed),
                "n_spilled_blocks": len(plan.spilled_blocks),
                "spilled_lm_head": bool(plan.spilled_lm_head),
                "reason": plan.reason,
            }
    except ImportError:
        obs["has_seam"] = False
    except Exception as e:  # noqa: BLE001
        obs["seam_error"] = f"{type(e).__name__}: {e}"

    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
