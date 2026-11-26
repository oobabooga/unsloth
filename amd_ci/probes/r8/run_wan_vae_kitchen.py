#!/usr/bin/env python3
"""M8 (a): Wan 2.2 TI2V-5B VAE decode, stock vs comfy-kitchen channel-first Triton ops (PRs 296 / 308 / 309), one
fresh process per arm. Observes only.

Both arms: diffusers AutoencoderKLWan in bf16 from the vae/ folder given, Studio's own residual no-clone patch
(``diffusion_vae_noclone.install``, what Studio installs on every video pipe), untiled decode of a seeded latent.
Arm ``kitchen`` additionally, in the DECODER only:
  * every WanResidualBlock norm1 / norm2 + SiLU and the decoder's norm_out + SiLU -> ``l2_norm_silu`` (one pass)
  * every WanResidualUpBlock ``x + DupUp3D(x_copy)``                              -> ``depth_to_space3d_residual_add``
  * every WanResample upsample2d / upsample3d interleave + rearrange + nearest 2x  -> ``temporal_interleave_upsample``
Engagement is counted per op. One warm decode (MIOpen find), then --reps timed decodes.

  run_wan_vae_kitchen.py --backend B --dir SNAPSHOT --arm stock|kitchen --out F [--save OUT.pt] [--latent 1,48,13,44,80]
                         [--dtype bf16|fp32]   (fp32 stock = the quality reference; the latent is the same bf16 draw)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def install_kitchen(vae, counts: dict) -> dict:
    import torch
    from torch import nn

    sys.path.insert(0, str(HERE / "kitchen"))
    import k_d2s3d
    import k_interleave_up
    import k_l2_norm_silu
    from diffusers.models.autoencoders import autoencoder_kl_wan as W

    dec = vae.decoder
    stats = {"norm_silu_modules": 0, "up_blocks": 0, "resamples": 0}

    def fused_norm(norm):
        bias = norm.bias
        if not norm.channel_first or (isinstance(bias, torch.Tensor)) or float(bias) != 0.0:
            return False

        def fwd(self, x):
            counts["norm_silu"] += 1
            return k_l2_norm_silu.l2_norm_silu(x, self.gamma.reshape(-1), float(self.scale))
        norm.forward = types.MethodType(fwd, norm)
        return True

    for m in dec.modules():
        if isinstance(m, W.WanResidualBlock) and isinstance(m.nonlinearity, nn.SiLU):
            if fused_norm(m.norm1) and fused_norm(m.norm2):
                m.nonlinearity = nn.Identity()
                stats["norm_silu_modules"] += 2
    if isinstance(dec.nonlinearity, nn.SiLU) and fused_norm(dec.norm_out):
        dec.nonlinearity = nn.Identity()  # WanDecoder3d uses it only after norm_out
        stats["norm_silu_modules"] += 1

    def up_forward(self, x, feat_cache = None, feat_idx = [0], first_chunk = False):  # noqa: B006 - diffusers' signature
        x_copy = x  # residual blocks never write their input in place (Studio's no-clone patch, bitwise stock)
        for resnet in self.resnets:
            x = resnet(x, feat_cache = feat_cache, feat_idx = feat_idx) if feat_cache is not None else resnet(x)
        if self.upsampler is not None:
            x = (self.upsampler(x, feat_cache = feat_cache, feat_idx = feat_idx) if feat_cache is not None
                 else self.upsampler(x))
        if self.avg_shortcut is not None:
            s = self.avg_shortcut
            counts["d2s"] += 1
            x = k_d2s3d.depth_to_space3d_residual_add(x_copy, x, s.factor_t, s.factor_s, s.repeats, bool(first_chunk))
        return x

    for m in dec.modules():
        if isinstance(m, W.WanResidualUpBlock):
            m.forward = types.MethodType(up_forward, m)
            stats["up_blocks"] += 1

    cache_t = W.CACHE_T

    def resample_forward(self, x, feat_cache = None, feat_idx = [0]):  # noqa: B006
        b, c, t, h, w = x.size()
        interleave = False
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -cache_t:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                                        dim = 2)
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim = 2)
                x = self.time_conv(x) if feat_cache[idx] == "Rep" else self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                interleave = True
        counts["interleave_up"] += 1
        y = k_interleave_up.temporal_interleave_upsample_nearest2x(x, interleave = interleave)
        t2 = t * 2 if interleave else t
        y = self.resample[1](y)
        return y.view(b, t2, y.size(1), y.size(2), y.size(3)).permute(0, 2, 1, 3, 4)

    for m in dec.modules():
        if isinstance(m, W.WanResample) and m.mode in ("upsample2d", "upsample3d"):
            if isinstance(m.resample[0], W.WanUpsample) and isinstance(m.resample[1], nn.Conv2d):
                m.forward = types.MethodType(resample_forward, m)
                stats["resamples"] += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--dir", required = True)
    ap.add_argument("--arm", required = True, choices = ("stock", "kitchen"))
    ap.add_argument("--out", required = True)
    ap.add_argument("--save")
    ap.add_argument("--latent", default = "1,48,13,44,80")  # 1280x704x49f
    ap.add_argument("--reps", type = int, default = 1)
    ap.add_argument("--dtype", default = "bf16", choices = ("bf16", "fp32"), help = "fp32: the reference decode")
    ap.add_argument("--random-init", action = "store_true", help = "local wiring test: config only, random weights")
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    import torch
    import diffusers

    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    obs = {"arm": a.arm, "dtype": a.dtype, "torch": torch.__version__, "hip": torch.version.hip, "diffusers": diffusers.__version__,
           "latent": [int(v) for v in a.latent.split(",")]}
    counts = {"norm_silu": 0, "d2s": 0, "interleave_up": 0}
    try:
        if a.random_init:
            cfg = diffusers.AutoencoderKLWan.load_config(a.dir, subfolder = "vae")
            torch.manual_seed(0)
            vae = diffusers.AutoencoderKLWan.from_config(cfg).to(dt)
        else:
            vae = diffusers.AutoencoderKLWan.from_pretrained(a.dir, subfolder = "vae", torch_dtype = dt)
        vae = vae.to("cuda").eval()
        try:
            from core.inference import diffusion_vae_noclone as NC
            obs["noclone_patched"] = NC.install(vae)
        except Exception as exc:  # noqa: BLE001
            obs["noclone_error"] = repr(exc)[:300]
        if a.arm == "kitchen":
            obs["kitchen"] = install_kitchen(vae, counts)
        g = torch.Generator(device = "cuda").manual_seed(7)
        z = torch.randn(obs["latent"], generator = g, device = "cuda", dtype = torch.float32).to(torch.bfloat16).to(dt)
        with torch.no_grad():
            torch.cuda.synchronize()
            t = time.perf_counter()
            vae.decode(z).sample
            torch.cuda.synchronize()
            obs["first_s"] = time.perf_counter() - t
            obs["calls_per_decode"] = dict(counts)
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(a.reps):
                torch.cuda.synchronize()
                t = time.perf_counter()
                y = vae.decode(z).sample
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
        obs["s"], obs["med_s"] = ts, st.median(ts)
        obs["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        obs["out_shape"] = list(y.shape)
        obs["finite"] = bool(torch.isfinite(y).all())
        if a.save:
            torch.save(y.float().clamp(-1, 1).to(torch.float16).contiguous().cpu(), a.save)
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-2000:]
    Path(a.out).write_text(json.dumps(obs), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
_ = os
