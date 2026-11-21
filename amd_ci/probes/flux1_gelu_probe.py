#!/usr/bin/env python3
"""Probe (gfx1151): the ConvRot GELU(tanh) Triton helper on ROCm and the FLUX.1 wiring around it.

Per state: (1) builds ``diffusion_convrot_quant._kernels()`` on this backend (the head defines its ROCm GELU helper
there, from the portable ``triton.language.extra.libdevice``), runs ``gelu_bf16`` over every finite bf16 input and
counts bit differences vs ``F.gelu(approximate="tanh")``; (2) tries one ``rotq_h16x`` launch (ACT 2 at the head, 0 at
the base) to record whether the producer kernel itself compiles on HIP (it uses *_rn libdevice calls CUDA only has;
the RDNA W8A8 route serves ConvRot on gfx1151 instead); (3) a tiny FLUX.1 transformer forward before / after
``diffusion_flux_convrot_fused.install`` (must not change: no rotated Linears) and whether the GELU producer armed.
Observes only (criteria/flux1_gelu.py judges). JSON to --out, never stdout."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F

    res: dict = {"state": args.state, "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                 "platform": sys.platform, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        res["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    _write(args.out, res)
    sys.path.insert(0, str(Path(args.checkout) / "studio" / "backend"))
    try:
        import triton
        res["triton"] = triton.__version__
        from core.inference import diffusion_convrot_quant as cq
        k = cq._kernels()
        res["kernels_built"] = k is not None
        res["has_gelu_helper"] = bool(k is not None and hasattr(k, "gelu_bf16"))
        if res["has_gelu_helper"] and torch.cuda.is_available():
            dev = torch.device("cuda", 0)
            x = torch.arange(-(2**15), 2**15, dtype = torch.int32).to(torch.int16).view(torch.bfloat16)
            x = x[torch.isfinite(x)].to(dev)
            out = torch.empty_like(x)
            k.gelu_bf16[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK = 1024, num_warps = 4,
                                                         enable_fp_fusion = False)
            ref = F.gelu(x, approximate = "tanh")
            diff = out.view(torch.int16) != ref.view(torch.int16)
            res["gelu_n"] = int(x.numel())
            res["gelu_bit_diffs"] = int(diff.sum())
            idx = diff.nonzero().flatten()[:8].tolist()
            res["gelu_diff_examples"] = [[float(x[i]), float(out[i]), float(ref[i])] for i in idx]
            res["gelu_max_abs_diff"] = float((out.float() - ref.float()).abs().max())
    except Exception as exc:  # noqa: BLE001
        res["kernel_error"] = repr(exc)[:1500]
        res["kernel_tb"] = traceback.format_exc()[-3000:]
    _write(args.out, res)
    if torch.cuda.is_available():
        try:
            from core.inference import diffusion_convrot_quant as cq
            dev = torch.device("cuda", 0)
            g = torch.randn(64, 1024, device = dev).to(torch.bfloat16)
            act = 2 if res.get("has_gelu_helper") else 0
            cq._h16x_launch(None, g, g, "v2", act = act)
            torch.cuda.synchronize()
            res["h16x_compiles"] = True
        except Exception as exc:  # noqa: BLE001
            res["h16x_compiles"] = False
            res["h16x_error"] = repr(exc)[:600]
    _write(args.out, res)
    try:
        from diffusers.models.transformers import transformer_flux as f1
        from core.inference import diffusion_convrot_quant as cq
        from core.inference import diffusion_flux_convrot_fused as fused

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(0)
        model = f1.FluxTransformer2DModel(patch_size = 1, in_channels = 8, num_layers = 1, num_single_layers = 2,
                                          attention_head_dim = 64, num_attention_heads = 2, joint_attention_dim = 32,
                                          pooled_projection_dim = 16, axes_dims_rope = (16, 24, 24)).to(dev, torch.bfloat16).eval()
        g = torch.Generator().manual_seed(3)
        kw = {"hidden_states": torch.randn(1, 64, 8, generator = g), "encoder_hidden_states": torch.randn(1, 16, 32, generator = g),
              "pooled_projections": torch.randn(1, 16, generator = g), "timestep": torch.tensor([0.5]),
              "img_ids": torch.randn(64, 3, generator = g), "txt_ids": torch.zeros(16, 3)}
        kw = {n: (v.to(dev, torch.bfloat16) if v.is_floating_point() and n not in ("img_ids", "txt_ids") else v.to(dev)) for n, v in kw.items()}
        with torch.no_grad():
            a = model(**kw, return_dict = False)[0]
            inst = fused.install(model)
            b = model(**kw, return_dict = False)[0]
        fused.uninstall()
        res["flux1_install"] = {k: v for k, v in inst.items() if k != "producer_cards"}
        res["flux1_forward_unchanged"] = bool(torch.equal(a, b))
        res["flux1_out_sum"] = float(a.float().sum())
        res["gelu_armed"] = bool(getattr(cq, "gelu_armed", lambda i: False)(0))
    except Exception as exc:  # noqa: BLE001
        res["flux1_error"] = repr(exc)[:1500]
    _write(args.out, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
