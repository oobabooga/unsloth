#!/usr/bin/env python3
"""Probe: Studio's VAE decode-compile gate and the compiled decode itself on this GPU.

Two tiny random AutoencoderKL VAEs: mid-block attention at head dim 64 (narrow) and 512 (wide, the width Windows ROCm's
fused SDPA gets wrong). For each it records the checkout's ``_vae_decode_compile_allowed`` on a U-Net pipe (default
mode) and on a DiT pipe with UNSLOTH_DIFFUSION_COMPILE_VAE=1. For the narrow VAE it also runs the decode eager and under
``torch.compile`` on the GPU and records relative L2 between them. Observes only (criteria judge). JSON to --out."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import types
from pathlib import Path


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def _vae(top: int):
    from diffusers import AutoencoderKL

    return AutoencoderKL(
        down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels = (32, top),
        norm_num_groups = 8,
        latent_channels = 4,
    ).eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    import torch

    res: dict = {"state": args.state, "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                 "platform": sys.platform, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        res["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    _write(args.out, res)
    try:
        sys.path.insert(0, str(Path(args.checkout) / "studio" / "backend"))
        from core.inference import diffusion_speed as S

        S._fused_vae_planned = lambda pipe: False  # the Triton norm fusion is not what is under test

        class UNet2DConditionModel(torch.nn.Module):
            pass

        S._UNET_WHOLE_COMPILE = frozenset({"UNet2DConditionModel"})
        torch.manual_seed(0)
        vaes = {"narrow": _vae(64), "wide": _vae(512)}
        allowed = {}
        for name, vae in vaes.items():
            os.environ.pop(S.COMPILE_VAE_ENV, None)
            unet = S._vae_decode_compile_allowed(types.SimpleNamespace(unet = UNet2DConditionModel(), vae = vae), "default")
            os.environ[S.COMPILE_VAE_ENV] = "1"
            forced = S._vae_decode_compile_allowed(types.SimpleNamespace(vae = vae), "default")
            os.environ.pop(S.COMPILE_VAE_ENV, None)
            allowed[name] = {"unet": unet, "dit_env_forced": forced}
        res["allowed"] = allowed
        _write(args.out, res)
        if torch.cuda.is_available():
            vae = vaes["narrow"].to("cuda")
            z = torch.randn(1, 4, 32, 32, device = "cuda")
            with torch.no_grad():
                eager = vae.decode(z).sample.float()
                compiled = torch.compile(vae.decode)
                out = compiled(z).sample.float()
            res["compiled_finite"] = bool(torch.isfinite(out).all())
            res["compiled_rel_l2"] = float((out - eager).norm() / eager.norm().clamp_min(1e-12))
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-3000:]
    _write(args.out, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
