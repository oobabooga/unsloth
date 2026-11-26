#!/usr/bin/env python3
"""R3-05: VAE decode on ROCm with and without PYTORCH_MIOPEN_SUGGEST_NHWC=1, one fresh process per arm.

  vae.py --backend B --model wan5b|qwen21 --dir D --arm contig|cl|cl_nhwc --out F --save OUT.pt

Arms: contig = Studio's ROCm layout today (no fused VAE on ROCm, so channels_last(_3d) is skipped);
cl = Studio's ``_vae_channels_last_3d`` relayout (post_quant_conv + decoder conv weights) and a channels_last_3d latent;
cl_nhwc = cl with PYTORCH_MIOPEN_SUGGEST_NHWC=1 (the caller sets it in this process's environment, before torch).
Warm decode(s) first (MIOpen find), engagement counted there with pre-hooks (conv inputs that arrive channels_last),
then timed decodes. Output saved fp16 for PSNR in the orchestrator."""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

CFG = {"wan5b": {"cls": "AutoencoderKLWan", "latent": (1, 48, 13, 44, 80), "warm": 1, "reps": 2},  # 1280x704x49f
       "qwen21": {"cls": "AutoencoderKLQwenImage21", "latent": (1, 64, 1, 64, 64), "warm": 2, "reps": 5}}  # 1024^2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--model", required = True)
    ap.add_argument("--dir", required = True)
    ap.add_argument("--arm", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--save")
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    import torch
    import diffusers

    c = CFG[a.model]
    obs = {"model": a.model, "arm": a.arm, "nhwc_env": os.environ.get("PYTORCH_MIOPEN_SUGGEST_NHWC"),
           "torch": torch.__version__, "hip": torch.version.hip, "diffusers": diffusers.__version__,
           "latent": list(c["latent"])}
    try:
        vae = getattr(diffusers, c["cls"]).from_pretrained(a.dir, subfolder = "vae", torch_dtype = torch.bfloat16)
        vae = vae.to("cuda").eval()
        g = torch.Generator(device = "cuda").manual_seed(7)
        z = torch.randn(c["latent"], generator = g, device = "cuda", dtype = torch.float32).to(torch.bfloat16)
        if a.arm in ("cl", "cl_nhwc"):
            from core.inference.diffusion_speed import _vae_channels_last_3d
            obs["relaid"] = bool(_vae_channels_last_3d(vae, None, fused = True))
            z = z.contiguous(memory_format = torch.channels_last_3d)
        seen = {"n": 0, "cl": 0}

        def pre(mod, args):
            x = args[0]
            seen["n"] += 1
            fmt = torch.channels_last_3d if x.dim() == 5 else torch.channels_last
            if x.dim() in (4, 5) and x.is_contiguous(memory_format = fmt) and not x.is_contiguous():
                seen["cl"] += 1
        hooks = [m.register_forward_pre_hook(pre) for m in vae.modules()
                 if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d))]
        with torch.no_grad():
            t = time.perf_counter()
            vae.decode(z).sample
            torch.cuda.synchronize()
            obs["first_s"] = time.perf_counter() - t
            for h in hooks:
                h.remove()
            obs["conv_calls"], obs["conv_cl_inputs"] = seen["n"], seen["cl"]
            for _ in range(c["warm"] - 1):
                vae.decode(z).sample
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(c["reps"]):
                torch.cuda.synchronize()
                t = time.perf_counter()
                y = vae.decode(z).sample
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
        obs["s"], obs["med_s"] = ts, st.median(ts)
        obs["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        obs["out_shape"] = list(y.shape)
        if a.save:
            torch.save(y.float().clamp(-1, 1).to(torch.float16).contiguous().cpu(), a.save)
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-1500:]
    open(a.out, "w", encoding = "utf-8").write(json.dumps(obs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
