# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""#14550 on real ROCm (gfx1151): the per-expert MoE dW kernel with the heuristic tile that
UNSLOTH_MOE_DISABLE_AUTOTUNE=1 picks (64x64x256, 8 warps, 4 stages; permute_x like gate_up, and unpermuted
like down_proj). Expect on main: OutOfResources (shared memory, ~81920 > 65536) for at least one case;
on branch moe-dw-legacy-fallback-on-oor: PASS for every case, equal to the legacy kernel bit for bit.
Prints one RESULT line per case and a final VERDICT line. Run with the checkout under test on PYTHONPATH."""
import os, sys, json
import torch
import unsloth  # noqa: F401
from unsloth.kernels.moe.grouped_gemm import interface as gg

print("torch", torch.__version__, "hip", torch.version.hip, "device", torch.cuda.get_device_name(0),
      "arch", getattr(torch.cuda.get_device_properties(0), "gcnArchName", "?"), "interface", gg.__file__, flush=True)
dtype = torch.bfloat16
results = []
for permute_x in (True, False):
    E, N, K, T, topk = 16, 1536, 2048, 64, 8
    g = torch.Generator(device="cuda").manual_seed(3)
    X = torch.randn(T if permute_x else T * topk, K, device="cuda", dtype=dtype, generator=g)
    dY = torch.randn(T * topk, N, device="cuda", dtype=dtype, generator=g)
    m_sizes = torch.full((E,), T * topk // E, device="cuda", dtype=torch.int32)
    m_sizes[0] += m_sizes[1]; m_sizes[1] = 0
    gather = (torch.randperm(T * topk, device="cuda", generator=g) if permute_x
              else torch.arange(T * topk, device="cuda")).to(torch.int32)
    kw = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256, num_warps=8, num_stages=4, permute_x=permute_x)
    os.environ["UNSLOTH_MOE_DW_KERNEL"] = "legacy"
    try:
        legacy = gg._grouped_gemm_dW_impl(X, dY, m_sizes, gather, topk, **kw); torch.cuda.synchronize()
    except Exception as e:
        legacy = None; print("legacy raised", type(e).__name__, str(e)[:200], flush=True)
    os.environ.pop("UNSLOTH_MOE_DW_KERNEL")
    try:
        out = gg._grouped_gemm_dW_impl(X, dY, m_sizes, gather, topk, **kw); torch.cuda.synchronize()
        status = "PASS" if legacy is not None and torch.equal(out, legacy) else "MISMATCH"
        err = None
    except Exception as e:
        status, err = type(e).__name__, str(e)[:200]
    r = {"permute_x": permute_x, "status": status, "error": err}
    results.append(r); print("RESULT", json.dumps(r), flush=True)
print("VERDICT", "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL", flush=True)
