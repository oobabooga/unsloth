#!/usr/bin/env python3
"""Observe (no verdict): on this GPU, which _fl_log1p / _fl_expm1 bodies unsloth_zoo's fused_linear compiled
(libdevice vs Kahan fallback), whether the Triton log-prob kernel launched (CALLS["triton"]) without marking the
device broken, and the confident-row (softmax collapse) error of fused_linear_logprobs vs an fp64 oracle.

PYTHONPATH must put the zoo checkout under test first. Writes JSON to --out.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    args = ap.parse_args()
    obs: dict = {}
    try:
        import torch
        import unsloth_zoo
        from unsloth_zoo.fused_losses import fused_linear as fl

        obs["zoo_file"] = unsloth_zoo.__file__
        obs["torch"] = torch.__version__
        obs["hip"] = torch.version.hip
        obs["cuda_available"] = torch.cuda.is_available()
        obs["arch"] = torch.cuda.get_device_properties(0).gcnArchName if torch.cuda.is_available() else None
        obs["has_triton"] = bool(getattr(fl, "_HAS_TRITON", False))
        try:
            import triton

            obs["triton"] = triton.__version__
        except Exception as e:  # noqa: BLE001
            obs["triton"] = f"{type(e).__name__}: {e}"
        for name in ("_fl_log1p", "_fl_expm1", "_fl_tanh"):
            fn = getattr(fl, name, None)
            if fn is None:
                obs[name] = "absent"
                continue
            src = inspect.getsource(getattr(fn, "fn", fn))
            obs[name] = "libdevice" if "_libdevice" in src else "fallback"
        dev = "cuda"
        torch.manual_seed(0)
        cases = []
        for V in (4096, 40001):
            for margin in (5.0, 20.0, 30.0):
                for dtype in (torch.float32, torch.bfloat16):
                    N, H = 64, 128
                    W = torch.randn(V, H, dtype = torch.float64)
                    t = torch.randint(0, V, (N,))
                    h = torch.randn(N, H, dtype = torch.float64) * 0.05
                    # Push the target logit `margin` above the row max of the others.
                    z = h @ W.T
                    zt = z[torch.arange(N), t]
                    zo = z.clone()
                    zo[torch.arange(N), t] = -float("inf")
                    need = (zo.max(-1).values + margin - zt) / W[t].pow(2).sum(-1)
                    h = h + need[:, None] * W[t]
                    hd = h.to(dtype).double()
                    Wd = W.to(dtype).double()
                    z = hd @ Wd.T
                    zt = z[torch.arange(N), t]
                    zo = z.clone()
                    zo[torch.arange(N), t] = -float("inf")
                    lse_o = torch.logsumexp(zo, -1)
                    ref_lp = -torch.nn.functional.softplus(lse_o - zt)
                    p = torch.softmax(z, -1)
                    oh = torch.zeros_like(p)
                    oh[torch.arange(N), t] = 1.0
                    ref_g = (oh - p) @ Wd
                    hh = h.to(dtype).to(dev).requires_grad_(True)
                    ww = W.to(dtype).to(dev)
                    c0 = dict(fl.CALLS)
                    lp = fl.fused_linear_logprobs(hh, ww, t.to(dev))
                    lp.sum().backward()
                    c1 = dict(fl.CALLS)
                    got_lp = lp.detach().double().cpu()
                    got_g = hh.grad.double().cpu()
                    rel_g = ((got_g - ref_g).norm(dim = -1) / ref_g.norm(dim = -1).clamp_min(1e-300)).max().item()
                    rel_lp = ((got_lp - ref_lp).abs() / ref_lp.abs().clamp_min(1e-300)).max().item()
                    cases.append({"V": V, "margin": margin, "dtype": str(dtype).split(".")[-1],
                                  "max_rel_hidden_grad": rel_g, "max_rel_logp": rel_lp,
                                  "max_logp": got_lp.max().item(),
                                  "triton_calls": c1.get("triton", 0) - c0.get("triton", 0),
                                  "torch_calls": c1.get("torch", 0) - c0.get("torch", 0)})
        obs["cases"] = cases
        obs["triton_broken"] = sorted(str(d) for d in getattr(fl, "_TRITON_BROKEN", ()))
        obs["calls_end"] = dict(fl.CALLS)
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
        obs["traceback"] = traceback.format_exc()[-3000:]
    with open(args.out, "w", encoding = "utf-8") as f:
        json.dump(obs, f, indent = 2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
