#!/usr/bin/env python3
"""PR 18825 mechanism probe 2: do this GPU's kernels give the same bits for the same values when the tensors sit at a
different address (storage offset / alignment), under memory pressure, or on repeat? A kernel whose result depends on
where its operands live would make a kept block (activations at first-pass addresses) differ from a recomputed one.
JSON to --out only."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--device", default = "cuda")
    ap.add_argument("--hog-gb", type = float, default = 40.0)
    args = ap.parse_args()
    res: dict = {}
    try:
        res = run(args)
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-4000:]
    Path(args.out).write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")
    return 0


def run(args) -> dict:
    import torch
    import torch.nn.functional as F

    dev = args.device
    torch.manual_seed(0)
    bf = torch.bfloat16
    S, D, FF, R, H = 1536, 3072, 9216, 16, 24
    base = {
        "x": torch.randn(S, D, device = dev, dtype = bf),
        "w": torch.randn(D, D, device = dev, dtype = bf) * D ** -0.5,
        "wff": torch.randn(FF, D, device = dev, dtype = bf) * D ** -0.5,
        "a": torch.randn(R, D, device = dev, dtype = bf) * D ** -0.5,
        "b": torch.randn(D, R, device = dev, dtype = bf) * 0.01,
        "g": torch.randn(S, D, device = dev, dtype = bf),
        "q": torch.randn(1, H, S, D // H, device = dev, dtype = bf),
        "k": torch.randn(1, H, S, D // H, device = dev, dtype = bf),
        "v": torch.randn(1, H, S, D // H, device = dev, dtype = bf),
    }

    def ops(t):
        out = {}
        out["linear"] = F.linear(t["x"], t["w"])
        out["linear_ff"] = F.linear(t["x"], t["wff"])
        out["lora_h"] = torch.mm(t["x"], t["a"].t())
        out["lora_out"] = torch.mm(out["lora_h"], t["b"].t())
        out["dgrad_w"] = torch.mm(t["g"].t(), t["x"])  # weight-grad shape: K = tokens
        out["dgrad_lora_b"] = torch.mm(t["g"].t(), out["lora_h"])
        out["dgrad_lora_a"] = torch.mm(torch.mm(t["g"], t["b"]).t(), t["x"])
        out["dgrad_x"] = torch.mm(t["g"], t["w"])
        out["layer_norm"] = F.layer_norm(t["x"], (D,))
        out["rms_norm"] = F.rms_norm(t["x"], (D,))
        out["softmax"] = torch.softmax(t["x"].float(), -1)
        out["gelu"] = F.gelu(t["x"], approximate = "tanh")
        q = t["q"].detach().requires_grad_()
        k = t["k"].detach().requires_grad_()
        v = t["v"].detach().requires_grad_()
        o = F.scaled_dot_product_attention(q, k, v)
        gq, gk, gv = torch.autograd.grad(o, (q, k, v), t["q"])
        out["sdpa"] = o.detach()
        out["sdpa_dq"], out["sdpa_dk"], out["sdpa_dv"] = gq, gk, gv
        with torch.autocast(dev, dtype = bf):
            out["autocast_linear_fp32in"] = F.linear(t["x"].float(), t["w"].float())
        (torch.cuda.synchronize() if dev == "cuda" else None)
        return out

    def placed(offset_bytes: int):
        """Same values, each tensor in a fresh buffer at ``offset_bytes`` past an allocation's start."""
        t = {}
        keep = []
        for name, v in base.items():
            n_off = offset_bytes // v.element_size()
            buf = torch.empty(v.numel() + n_off + 64, device = dev, dtype = v.dtype)
            keep.append(buf)
            view = buf[n_off:n_off + v.numel()].view(v.shape)
            view.copy_(v)
            t[name] = view
        return t, keep

    ref = ops(base)
    try:
        blas = str(torch.backends.cuda.preferred_blas_library())
    except Exception as exc:  # noqa: BLE001
        blas = repr(exc)
    res: dict = {"torch": torch.__version__, "hip": torch.version.hip, "platform": sys.platform,
                 "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "cases": {}, "preferred_blas": blas}

    def compare(tag, out):
        diffs = {}
        for k, v in ref.items():
            same = bool(torch.equal(v, out[k]))
            if not same:
                diffs[k] = float((v.float() - out[k].float()).abs().max())
        res["cases"][tag] = {"identical": not diffs, "differs": diffs}

    compare("repeat", ops(base))
    for off in (2, 16, 64, 128, 256, 512, 1024, 4096, 65536):
        t, keep = placed(off)
        compare(f"offset_{off}B", ops(t))
        del t, keep
    # Memory pressure: hold most of the card, then rerun in place and at offsets.
    hog = []
    try:
        for _ in range(int(args.hog_gb)):
            hog.append(torch.empty(1 << 30, dtype = torch.uint8, device = dev))
    except (torch.OutOfMemoryError, RuntimeError):
        pass
    res["hog_gb_held"] = len(hog)
    compare("under_hog_repeat", ops(base))
    for off in (256, 512, 4096):
        t, keep = placed(off)
        compare(f"under_hog_offset_{off}B", ops(t))
        del t, keep
    del hog
    torch.cuda.empty_cache()
    compare("after_hog_repeat", ops(base))
    # The autograd engine thread with offsets (where a recompute's GEMMs run).
    box = {}

    class InBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            return t.clone()

        @staticmethod
        def backward(ctx, g):
            t, keep = placed(512)
            with torch.enable_grad():
                box["out"] = ops(t)
            return g

    z = torch.ones(4, device = dev, requires_grad = True)
    InBackward.apply(z).sum().backward()
    compare("engine_thread_offset_512B", box["out"])
    return res


if __name__ == "__main__":
    sys.exit(main())
