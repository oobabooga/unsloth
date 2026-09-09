#!/usr/bin/env python3
"""Probe: measure the down-LoRA row-order error in a given unsloth-zoo checkout.

OBSERVES ONLY. It reports relative errors against an independent fp32 per-token
reference and never decides whether they are acceptable; criteria/moe_roworder.py
does that.

The PR's own test file exists only at the head, so the generic pytest probe would
collect nothing at the base and the differential would be VOID. This probe carries
its own reproducer instead, so the SAME measurement runs at every state.

moe_utils.py is loaded from the checkout via importlib rather than by importing the
installed unsloth_zoo, so the SAME file the state actually contains is what runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def measure(checkout: str) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    obs: dict = {}
    obs["torch"] = torch.__version__
    obs["hip"] = getattr(torch.version, "hip", None)
    if not torch.cuda.is_available():
        obs["error"] = "no GPU visible to torch"
        return obs
    obs["device"] = torch.cuda.get_device_name(0)
    try:
        obs["arch"] = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        obs["arch"] = None

    path = Path(checkout) / "unsloth_zoo" / "temporary_patches" / "moe_utils.py"
    if not path.is_file():
        obs["error"] = f"no moe_utils.py at {path}"
        return obs

    # Import unsloth BEFORE the checkout's moe_utils. moe_utils pulls in the
    # unsloth_zoo package, whose __init__ raises "Please install Unsloth via
    # `pip install unsloth`!" unless UNSLOTH_IS_PRESENT is set - and only
    # importing unsloth sets it. Importing it afterwards is too late.
    try:
        import unsloth  # noqa: F401
        from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (
            KernelConfigBackward_dW, KernelConfigBackward_dX, KernelConfigForward)
    except Exception as e:
        obs["error"] = f"unsloth MoE kernels unavailable: {type(e).__name__}: {e}"
        return obs

    sys.path.insert(0, checkout)
    spec = importlib.util.spec_from_file_location("probe_moe_utils", path)
    mu = importlib.util.module_from_spec(spec)
    sys.modules["probe_moe_utils"] = mu
    try:
        spec.loader.exec_module(mu)
    except Exception as e:
        obs["error"] = f"could not load moe_utils: {type(e).__name__}: {e}"
        return obs

    dev, dt = "cuda", torch.bfloat16
    E, H, I, R, T, K = 8, 256, 512, 16, 512, 2
    torch.manual_seed(0)

    ex = nn.Module()
    ex.num_experts = E
    ex.act_fn = F.silu
    ex.gate_up_proj = nn.Parameter((torch.randn(E, 2 * I, H, device=dev) * 0.2).to(dt))
    ex.down_proj = nn.Parameter((torch.randn(E, H, I, device=dev) * 0.2).to(dt))
    c = lambda: (KernelConfigForward(), KernelConfigBackward_dX(), KernelConfigBackward_dW())
    ex._unsloth_moe_configs = (I, c(), c())

    A = (torch.randn(E, I, R, device=dev) * 0.3).to(dt).requires_grad_(True)
    B = (torch.randn(E, R, H, device=dev) * 0.3).to(dt).requires_grad_(True)
    s = 0.5
    ex._unsloth_lora_down_proj = (A, B, s)

    X = (torch.randn(T, H, device=dev) * 0.5).to(dt)
    idx = torch.randint(0, E, (T, K), device=dev)
    w = torch.softmax(torch.randn(T, K, device=dev), dim=-1)

    # Non-vacuity: the routing must actually permute rows, else the defect cannot bite.
    _, gi = mu._get_routing_indices(idx, E)
    obs["routing_permutes"] = not bool(torch.equal(gi, torch.arange(T * K, device=dev)))
    obs["gather_is_permutation"] = bool(
        torch.equal(gi.sort().values, torch.arange(T * K, device=dev)))

    try:
        out = mu.forward_triton_grouped_gemm(ex, X, idx, w)
    except Exception as e:
        obs["error"] = f"forward raised: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
        return obs

    Ar = A.detach().float().requires_grad_(True)
    Br = B.detach().float().requires_grad_(True)
    ref = torch.zeros(T, H, device=dev, dtype=torch.float32)
    for e in range(E):
        t, k = (idx == e).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        g_, u_ = (X[t].float() @ ex.gate_up_proj[e].float().T).chunk(2, dim=-1)
        h = F.silu(g_) * u_
        d = h @ ex.down_proj[e].float().T + (h @ Ar[e] @ Br[e]) * s
        ref.index_add_(0, t, w[t, k].unsqueeze(1).float() * d)

    torch.manual_seed(99)
    cw = torch.randn_like(ref)
    (out.float() * cw).sum().backward()
    (ref * cw).sum().backward()

    def rel(got, want):
        return float((got.float() - want).abs().max()) / max(float(want.abs().max()), 1e-9)

    obs["rel_forward"] = rel(out, ref)
    obs["rel_lora_A"] = rel(A.grad, Ar.grad)
    obs["rel_lora_B"] = rel(B.grad, Br.grad)
    obs["worst"] = max(obs["rel_forward"], obs["rel_lora_A"], obs["rel_lora_B"])
    obs["delta_nonzero"] = bool(float(A.grad.float().abs().max()) > 0)
    # Which combination the checkout uses, read off the source rather than inferred.
    # Scope to the FUNCTION BODY: later functions in this file use index_add_ too, so
    # searching from the def to end-of-file reports every checkout as carrying the fix.
    import ast
    src = path.read_text(encoding="utf-8")
    body = ""
    try:
        tree = ast.parse(src)
        fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "forward_triton_grouped_gemm"), None)
        if fn is not None:
            body = ast.get_source_segment(src, fn) or ""
    except SyntaxError:
        body = ""
    obs["fn_found"] = bool(body)
    obs["uses_index_add"] = "index_add" in body
    obs["uses_plain_add"] = "second_gemm_output = second_gemm_output + lora_delta" in body
    return obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    obs: dict = {"state": args.state, "checkout": args.checkout}
    try:
        obs.update(measure(args.checkout))
    except Exception as e:
        import traceback
        obs["error"] = f"{type(e).__name__}: {e}"
        obs["traceback"] = traceback.format_exc()[-1500:]
    # JSON goes to --out, never stdout: import banners corrupt stdout (lint E004).
    args.out.write_text(json.dumps(obs, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
