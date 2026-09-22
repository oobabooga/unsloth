#!/usr/bin/env python3
"""Probe: what dtype does the checkpoint recompute actually see, and are the bytes intact?

unslothai/unsloth-zoo PR 1316. `UnslothCheckpointFunction` offloads a large activation into
a staging buffer that was allocated in the dtype checkpointing was INITIALISED with. That is
not always the dtype the model runs in: a FORCE_FLOAT32 family (qwen3_5, gemma3) on a GPU
without bfloat16 initialises bfloat16 and runs float16. Backward handed the recompute the
BUFFER, so the recompute ran in the buffer's dtype.

This probe observes only. It runs the real `UnslothCheckpointFunction` over an activation
big enough to be offloaded, records the dtype and the exact bytes seen by every forward call
and every recompute call, and writes them to JSON. It never decides whether that is right:
criteria/gc_offload_dtype.py does that.

The LLVM abort the PR reports is gfx10-specific (Triton cannot compile bf16 there). gfx1151
HAS bfloat16, so the abort is NOT reachable on this host and is not what is measured. What is
measured is the cause the abort shares with the `BFloat16 != Half` reports: the recompute
running on a tensor of the wrong dtype.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Read at init time by _double_buffer_disabled(). "0" forces double buffering ON, which an
# integrated GPU (Strix Halo is one) otherwise disables, leaving GPU_BUFFERS_B - a buffer
# this PR changes - completely unexercised.
os.environ.setdefault("UNSLOTH_DISABLE_DOUBLE_BUFFER", "0")

# Big enough to clear the 2MB offload cutoff in every dtype tested here:
# 2 * 1024 * 2048 elements = 4,194,304 -> 8 MB in 16-bit, 16 MB in float32.
SHAPE = (2, 1024, 2048)
N_LAYERS = 4
N_PASSES = 2          # double buffering only turns on after the first pass

_DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}


def _case_list(torch):
    return [
        # name, init dtype (what checkpointing was set up with), activation dtype
        ("bf16_buffers_fp16_activation", torch.bfloat16, torch.float16),
        ("fp16_buffers_fp32_activation", torch.float16, torch.float32),
        ("bf16_buffers_bf16_activation", torch.bfloat16, torch.bfloat16),
    ]


def _run_case(torch, gc, init_dtype, act_dtype):
    """One case: init checkpointing, then N_PASSES forward+backward over N_LAYERS."""
    gc.initialize_unsloth_gradient_checkpointing(init_dtype)
    passes = []
    for pass_no in range(1, N_PASSES + 1):
        seen = []   # (phase_marker, dtype, cloned tensor) appended by the layer itself

        def layer(hidden):
            seen.append((str(hidden.dtype).replace("torch.", ""), hidden.detach().clone()))
            return hidden * 2

        hidden = torch.randn(*SHAPE, device = "cuda", dtype = act_dtype, requires_grad = True)
        out = hidden
        for _ in range(N_LAYERS):
            out = gc.UnslothCheckpointFunction.apply(layer, False, out)
        n_forward = len(seen)
        cpu_index_after_forward = getattr(gc, "CPU_INDEX", None)
        out.float().sum().backward()
        torch.cuda.synchronize()

        forward_calls = seen[:n_forward]
        recompute_calls = seen[n_forward:]

        # Backward recomputes in reverse layer order, so recompute i corresponds to
        # forward call (n_forward - 1 - i).
        bitexact, pairs = True, []
        for i, (dt, tensor) in enumerate(recompute_calls):
            j = n_forward - 1 - i
            if j < 0 or j >= n_forward:
                bitexact = False
                pairs.append({"recompute_index": i, "forward_index": j, "matched": False,
                              "note": "no corresponding forward call"})
                continue
            fdt, ftensor = forward_calls[j]
            same_dtype = (dt == fdt)
            same_bytes = bool(same_dtype and torch.equal(tensor, ftensor))
            if not same_bytes:
                bitexact = False
            pairs.append({"recompute_index": i, "forward_index": j,
                          "forward_dtype": fdt, "recompute_dtype": dt,
                          "same_dtype": same_dtype, "same_bytes": same_bytes})

        grad = hidden.grad
        passes.append({
            "pass": pass_no,
            "use_double_buffer": bool(getattr(gc, "USE_DOUBLE_BUFFER", False)),
            "double_buffer_allocated": getattr(gc, "GPU_BUFFERS_B", None) is not None,
            "minimum_size": getattr(gc, "MINIMUM_SIZE", None),
            "cpu_index_after_forward": cpu_index_after_forward,
            "offloaded": bool(cpu_index_after_forward),
            "n_forward_calls": n_forward,
            "n_recompute_calls": len(recompute_calls),
            "forward_dtypes": sorted({d for d, _ in forward_calls}),
            "recompute_dtypes": sorted({d for d, _ in recompute_calls}),
            "recompute_bitexact": bitexact,
            "pairs": pairs,
            "grad_dtype": str(grad.dtype).replace("torch.", "") if grad is not None else None,
            # f(x) = 2x chained N_LAYERS times, so d(sum)/dx is exactly 2**N_LAYERS.
            "grad_correct": bool(grad is not None and torch.equal(
                grad, torch.full_like(grad, float(2 ** N_LAYERS)))),
        })
        del hidden, out, seen
        torch.cuda.empty_cache()
    return passes


def _threshold_observation(torch, gc):
    """Informational: MINIMUM_SIZE moved from elements to bytes, so the cutoff is now the
    same amount of MEMORY whatever width an activation arrives in. Record whether a 3 MB
    float32 activation over bfloat16 buffers is offloaded; it is not a defect either way."""
    out = {}
    try:
        gc.initialize_unsloth_gradient_checkpointing(torch.bfloat16)
        out["minimum_size_value"] = getattr(gc, "MINIMUM_SIZE", None)
        shape = (1, 384, 2048)     # 786,432 float32 elements = 3 MB
        out["shape"] = list(shape)
        out["nbytes"] = 786432 * 4
        out["numel"] = 786432

        def layer(hidden):
            return hidden * 2

        hidden = torch.randn(*shape, device = "cuda", dtype = torch.float32, requires_grad = True)
        o = gc.UnslothCheckpointFunction.apply(layer, False, hidden)
        o = gc.UnslothCheckpointFunction.apply(layer, False, o)
        out["cpu_index_after_forward"] = getattr(gc, "CPU_INDEX", None)
        out["offloaded"] = bool(getattr(gc, "CPU_INDEX", 0))
        o.float().sum().backward()
        torch.cuda.synchronize()
        del hidden, o
        torch.cuda.empty_cache()
    except Exception:
        out["error"] = traceback.format_exc()[-3000:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": args.checkout,
                 "shape": list(SHAPE), "n_layers": N_LAYERS, "n_passes": N_PASSES}

    # The state's own source must win over anything installed in the venv, otherwise both
    # legs measure the same site-packages copy and the differential is meaningless. The
    # criteria gate on gc_file_in_checkout for exactly that reason.
    sys.path.insert(0, args.checkout)
    try:
        import torch
        from unsloth_zoo import gradient_checkpointing as gc
    except Exception:
        obs["import_error"] = traceback.format_exc()[-4000:]
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    obs["gc_file"] = getattr(gc, "__file__", None)
    obs["gc_file_in_checkout"] = bool(
        obs["gc_file"] and os.path.realpath(obs["gc_file"]).startswith(
            os.path.realpath(args.checkout) + os.sep))
    obs["torch_version"] = torch.__version__
    obs["torch_hip"] = getattr(torch.version, "hip", None)
    obs["device_type"] = getattr(gc, "DEVICE_TYPE", None)
    obs["bf16_supported"] = None
    try:
        obs["cuda_available"] = bool(torch.cuda.is_available())
        if obs["cuda_available"]:
            obs["device_name"] = torch.cuda.get_device_name(0)
            obs["device_count"] = torch.cuda.device_count()
            obs["gcn_arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
            obs["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception:
        obs["device_error"] = traceback.format_exc()[-2000:]

    if not obs.get("cuda_available"):
        obs["error"] = "no GPU visible to torch, so the offload path cannot run"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    cases: dict = {}
    for name, init_dtype, act_dtype in _case_list(torch):
        entry = {"init_dtype": str(init_dtype).replace("torch.", ""),
                 "activation_dtype": str(act_dtype).replace("torch.", "")}
        try:
            entry["passes"] = _run_case(torch, gc, init_dtype, act_dtype)
        except Exception:
            # A base state that dies here is an observation, not a probe failure.
            entry["error"] = traceback.format_exc()[-3000:]
        cases[name] = entry
    obs["cases"] = cases
    obs["threshold"] = _threshold_observation(torch, gc)

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
