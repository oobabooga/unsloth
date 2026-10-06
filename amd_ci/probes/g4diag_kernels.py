#!/usr/bin/env python3
"""Diagnostic (observes only): bitsandbytes NF4 kernels and torch._grouped_mm on this GPU vs references."""
import json
import sys
import traceback

out = {}
try:
    import torch
    out["torch"] = torch.__version__
    out["arch"] = torch.cuda.get_device_properties(0).gcnArchName
    import bitsandbytes as bnb
    import bitsandbytes.functional as F
    out["bnb"] = bnb.__version__
    try:
        import bitsandbytes.cextension as ce
        out["bnb_lib"] = str(getattr(getattr(ce, "lib", None), "_lib", None))
    except Exception as e:  # noqa: BLE001
        out["bnb_lib"] = repr(e)
    torch.manual_seed(0)
    res = {}
    for (n, k) in ((2816, 2816), (704, 2816), (2816, 704)):
        W = (torch.randn(n, k, device = "cuda", dtype = torch.bfloat16) * 0.02)
        r = {}
        try:
            q, st = F.quantize_4bit(W, quant_type = "nf4", compress_statistics = True)
            Wd = F.dequantize_4bit(q, st)
            torch.cuda.synchronize()
            r["dequant_rel_err"] = ((Wd.float() - W.float()).norm() / W.float().norm()).item()
            for b in (1, 8):
                x = torch.randn(b, k, device = "cuda", dtype = torch.bfloat16)
                y = bnb.matmul_4bit(x, q.t(), quant_state = st)
                ref = x.float() @ Wd.float().t()
                torch.cuda.synchronize()
                r[f"matmul_b{b}_rel_err"] = ((y.float() - ref).norm() / ref.norm()).item()
                r[f"matmul_b{b}_finite"] = bool(torch.isfinite(y).all().item())
        except Exception as e:  # noqa: BLE001
            r["error"] = repr(e)[:500]
            r["tb"] = traceback.format_exc()[-1500:]
        res[f"{n}x{k}"] = r
    out["bnb_nf4"] = res
    g = {}
    try:
        E, T, K, N = 8, 64, 512, 256
        x = torch.randn(T, K, device = "cuda", dtype = torch.bfloat16)
        w = torch.randn(E, K, N, device = "cuda", dtype = torch.bfloat16) * 0.05
        offs = torch.tensor([8 * (i + 1) for i in range(E)], device = "cuda", dtype = torch.int32)
        y = torch._grouped_mm(x, w, offs = offs)
        ref = torch.cat([x[8 * i:8 * (i + 1)].float() @ w[i].float() for i in range(E)])
        torch.cuda.synchronize()
        g["rel_err"] = ((y.float() - ref).norm() / ref.norm()).item()
        g["finite"] = bool(torch.isfinite(y).all().item())
    except Exception as e:  # noqa: BLE001
        g["error"] = repr(e)[:500]
    out["grouped_mm"] = g
except Exception as e:  # noqa: BLE001
    out["fatal"] = repr(e)
    out["tb"] = traceback.format_exc()[-2000:]
with open(sys.argv[1], "w", encoding = "utf-8") as f:
    json.dump(out, f, indent = 2)
