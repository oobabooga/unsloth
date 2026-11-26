#!/usr/bin/env python3
"""M8 (c): MiniMax-H3 VAE single-image (T = 1) encode on ROCm, stock Conv3d vs the exact Conv3d -> Conv2d last-tap
lowering, one fresh process per arm. Observes only.

The H3 encoder's causal convs prepend ``kt - 1`` zero frames; with one input frame every earlier tap multiplies zeros,
so ``conv3d(pad_t(x), w) == conv2d(x[:, :, 0], w[:, :, -1])`` (spatial reflect pad kept). Arm ``lowered`` patches
every ``MiniMaxH3VideoCausalConv3d`` whose input has T = 1 and ``temporal_padding == kt - 1`` (kt > 1) to that conv2d;
other calls stay stock. This is the identity Studio already uses on CUDA (``video_minimax_h3_vae._last_tap``) and gates
off on ROCm. Both arms call Studio's ``apply_h3_vae_speedups`` first (stock on ROCm: recorded). Input: a seeded
deterministic 1024 x 1024 image in [-1, 1]; output: the posterior mean (fp32 on the host for PSNR).

  run_h3_ref_encode.py --backend B --dir SNAPSHOT --arm stock|lowered --out F [--save OUT.pt] [--dtype bf16|fp32]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
import types
from pathlib import Path


def lower(vae, counts: dict) -> int:
    import torch.nn.functional as F
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as H

    stock = H.MiniMaxH3VideoCausalConv3d.forward
    n = 0

    def fwd(self, x):
        kt = self.kernel_size[0]
        if x.shape[2] == 1 and kt > 1 and int(self.temporal_padding) == kt - 1 and self.padding == (0, 0, 0):
            counts["lowered"] += 1
            x2 = x[:, :, 0]
            if self.spatial_padding > 0:
                p = self.spatial_padding
                x2 = F.pad(x2, (p, p, p, p), mode = self.spatial_padding_mode)
            y = F.conv2d(x2, self.weight[:, :, -1], self.bias, stride = self.stride[1:], padding = 0,
                         dilation = self.dilation[1:])
            return y.unsqueeze(2)
        counts["stock"] += 1
        return stock(self, x)

    for m in vae.modules():
        if isinstance(m, H.MiniMaxH3VideoCausalConv3d):
            m.forward = types.MethodType(fwd, m)
            n += 1
    return n


def image(side: int, device, dtype):
    import torch
    g = torch.Generator(device = "cpu").manual_seed(11)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, side), torch.linspace(-1, 1, side), indexing = "ij")
    base = torch.stack([torch.sin(3.1 * xx + 1.7 * yy), torch.cos(2.3 * yy - 0.9 * xx), xx * yy], 0)
    noise = torch.randn((3, side // 8, side // 8), generator = g)
    noise = torch.nn.functional.interpolate(noise[None], size = (side, side), mode = "bilinear", align_corners = False)[0]
    img = (0.7 * base + 0.3 * noise).clamp(-1, 1)
    return img[None, :, None].to(device = device, dtype = dtype)  # [1, 3, 1, H, W]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--dir", required = True)
    ap.add_argument("--arm", required = True, choices = ("stock", "lowered"))
    ap.add_argument("--out", required = True)
    ap.add_argument("--save")
    ap.add_argument("--side", type = int, default = 1024)
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--dtype", default = "bf16", choices = ("bf16", "fp32"))
    ap.add_argument("--random-init", action = "store_true")
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    import torch
    import diffusers

    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    obs = {"arm": a.arm, "dtype": a.dtype, "side": a.side, "torch": torch.__version__, "hip": torch.version.hip,
           "diffusers": diffusers.__version__}
    counts = {"lowered": 0, "stock": 0}
    try:
        cls = diffusers.AutoencoderKLMiniMaxH3
        if a.random_init:
            torch.manual_seed(0)
            vae = cls.from_config(cls.load_config(a.dir, subfolder = "vae")).to(dt)
        else:
            vae = cls.from_pretrained(a.dir, subfolder = "vae", torch_dtype = dt)
        vae = vae.to("cuda").eval()
        try:
            from core.inference import video_minimax_h3_vae as SH
            obs["studio_fast_path_available"] = bool(SH.cuda_fast_path_available(vae))
            obs["studio_levers"] = list(SH.apply_h3_vae_speedups(vae, speed_mode = "default", workflow = "i2v") or [])
        except Exception as exc:  # noqa: BLE001
            obs["studio_levers_error"] = repr(exc)[:300]
        obs["use_tiling"] = bool(getattr(vae, "use_tiling", False))
        if a.arm == "lowered":
            obs["patched_convs"] = lower(vae, counts)
        x = image(a.side, "cuda", dt)
        with torch.no_grad():
            torch.cuda.synchronize()
            t = time.perf_counter()
            vae.encode(x)
            torch.cuda.synchronize()
            obs["first_s"] = time.perf_counter() - t
            obs["calls_per_encode"] = dict(counts)
            ts = []
            for _ in range(a.reps):
                torch.cuda.synchronize()
                t = time.perf_counter()
                post = vae.encode(x).latent_dist
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
        mean = post.mean if hasattr(post, "mean") else post
        obs["s"], obs["med_s"] = ts, st.median(ts)
        obs["latent_shape"] = list(mean.shape)
        obs["finite"] = bool(torch.isfinite(mean).all())
        if a.save:
            torch.save(mean.float().cpu(), a.save)
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-2000:]
    Path(a.out).write_text(json.dumps(obs), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
