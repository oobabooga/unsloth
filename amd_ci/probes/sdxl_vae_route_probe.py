#!/usr/bin/env python3
"""Probe: the SDXL LoRA trainer's training latents on this GPU against a CPU fp32 encode of the same pixels.

Runs the checkout's ``run_diffusion_lora_training`` (1 step) on the tiny SDXL pipeline with its VAE swapped for the
real SDXL VAE (single-head mid-block attention at head dim 512, the width Windows ROCm's fused SDPA gets wrong). The
first ``vae.encode`` is captured (pixels in, latent mean out); the same pixels are encoded on CPU fp32 with stock
attention as the reference. Observes only (criteria/sdxl_vae_route.py judges). JSON to --out, never stdout."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

VAE_REPO = "stabilityai/sdxl-vae"
PIPE_REPO = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


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
        if not torch.cuda.is_available():
            sys.modules["unsloth_zoo"] = None  # a CPU dry run: the trainer's optional zoo import needs an accelerator
        import diffusers
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline
        from huggingface_hub import snapshot_download
        from PIL import Image

        from core.training import diffusion_lora_trainer as trainer
        from core.training.diffusion_train_common import DiffusionLoraConfig

        pipe_dir = snapshot_download(PIPE_REPO)
        real_vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype = torch.float32)
        ref_vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype = torch.float32).eval()
        res["diffusers"] = diffusers.__version__
        captured: dict = {}
        real_from_pretrained = StableDiffusionXLPipeline.from_pretrained.__func__

        def from_pretrained(cls, *a, **k):
            pipe = real_from_pretrained(cls, *a, **k)
            pipe.vae = real_vae
            real_encode = real_vae.encode

            def encode(x, *ea, **ek):
                out = real_encode(x, *ea, **ek)
                if "pixels" not in captured:
                    captured["pixels"] = x.detach().float().cpu()
                    captured["mean"] = out.latent_dist.mean.detach().float().cpu()
                    captured["patched"] = sum(
                        1 for m in real_vae.modules() if getattr(m, "_unsloth_vae_attn_chunked", False)
                    )
                return out

            real_vae.encode = encode
            return pipe

        StableDiffusionXLPipeline.from_pretrained = classmethod(from_pretrained)
        trainer._assert_trusted_base_model = lambda *a, **k: None
        tmp = Path(tempfile.mkdtemp(prefix = "sdxl_vae_route_"))
        data = tmp / "data"
        data.mkdir()
        g = torch.Generator().manual_seed(0)
        pix = (torch.rand(256, 256, 3, generator = g) * 255).to(torch.uint8).numpy()
        Image.fromarray(pix).save(data / "img0.png")
        (data / "img0.txt").write_text("a tile", encoding = "utf-8")
        import core.inference.diffusion_lora as dl

        (tmp / "catalog").mkdir()
        dl.loras_dir = lambda: tmp / "catalog"
        cfg = DiffusionLoraConfig(base_model = pipe_dir, model_family = "sdxl", data_dir = str(data),
                                  output_dir = str(tmp / "out"), resolution = 256, train_steps = 1,
                                  learning_rate = 1e-4, lora_rank = 4, seed = 0, compile_transformer = "off",
                                  cache_latents = False)
        events = []
        trainer.run_diffusion_lora_training(cfg, on_event = events.append)
        res["errors"] = [e for e in events if e.get("type") == "error"][:3]
        if "pixels" not in captured:
            res["error"] = "no vae.encode captured"
        else:
            with torch.no_grad():
                ref = ref_vae.encode(captured["pixels"]).latent_dist.mean.float()
            got = captured["mean"]
            res["finite"] = bool(torch.isfinite(got).all())
            res["rel_l2"] = float((got - ref).norm() / ref.norm().clamp_min(1e-12)) if res["finite"] else None
            res["max_abs"] = float((got - ref).abs().max()) if res["finite"] else None
            res["patched"] = captured["patched"]
            res["latent_shape"] = list(got.shape)
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        res["traceback"] = traceback.format_exc()[-3000:]
    _write(args.out, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
