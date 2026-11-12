# SPDX-License-Identifier: AGPL-3.0-only
"""Diagnose test_existing_gradient_survives_in_microbatch_cancellation on ROCm: which dW path loses the
existing gradient. Prints one DIAG line per variant (JSON)."""
import json, os
import torch
import unsloth  # noqa: F401
from unsloth.kernels.moe.grouped_gemm import interface as gg
from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW, KernelConfigBackward_dX, KernelConfigForward,
)
print("HDR", json.dumps({"torch": torch.__version__, "hip": torch.version.hip, "dev": torch.cuda.get_device_name(0),
      "file": gg.__file__}), flush=True)


def run(big, bm, acc, kernel, rows, dtype=torch.bfloat16):
    os.environ["UNSLOTH_MOE_DW_ACCUMULATE"] = "1" if acc else "0"
    if kernel == "legacy":
        os.environ["UNSLOTH_MOE_DW_KERNEL"] = "legacy"
    else:
        os.environ.pop("UNSLOTH_MOE_DW_KERNEL", None)
    E, N, K, T = 1, 64, 64, 128
    X = torch.zeros(T, K, device="cuda", dtype=dtype)
    G = torch.zeros(T, N, device="cuda", dtype=dtype)
    X[rows[0]], X[rows[1]] = big, big
    G[rows[0]], G[rows[1]] = big, -big
    m = torch.tensor([T], device="cuda", dtype=torch.int32)
    gather = torch.arange(T, device="cuda", dtype=torch.int32)
    torch.manual_seed(1)
    W = torch.nn.Parameter(torch.randn(E, N, K, device="cuda").to(dtype) * 0.05)
    W.grad = torch.ones_like(W)
    cf = (KernelConfigForward(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64),
          KernelConfigBackward_dX(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64),
          KernelConfigBackward_dW(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64))
    out = {"big": big, "bm": bm, "acc": acc, "kernel": kernel, "rows": rows}
    try:
        y = gg.grouped_gemm(X=X, W=W, m_sizes=m, topk=1, gather_indices=gather, permute_x=False, permute_y=False,
                            kernel_config_fwd=cf[0], kernel_config_bwd_dX=cf[1], kernel_config_bwd_dW=cf[2],
                            is_first_gemm=True)
        y.backward(G)
        torch.cuda.synchronize()
        g = W.grad.float()
        out.update(grad_unique=sorted(set(g.flatten().tolist()))[:6], ok=bool(torch.equal(W.grad, torch.ones_like(W))))
        ref = (G.double().T @ X.double()).float()
        out["ref_unique"] = sorted(set(ref.flatten().tolist()))[:4]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    print("DIAG", json.dumps(out), flush=True)


for kernel in ("per_expert", "legacy"):
    for acc in (True, False):
        for bm in (64, 32, 128):
            for big in (1e4, 1e2):
                run(big, bm, acc, kernel, (0, 64))
        run(1e4, 64, acc, kernel, (0, 1))  # both rows in one M tile
