#!/usr/bin/env python3
"""M8 (d): Qwen-Image-2.1-Turbo VAE decode UNTILED at --side px on gfx1151 (sglang #41655 reports a hang at >= 896 px
on Strix Halo), one decode per fresh process (the caller's timeout is the hang detector). Correctness only.

  run_qvae_untiled.py --dir SNAPSHOT --side 1024|1280 --out F
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required = True)
    ap.add_argument("--side", type = int, required = True)
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    obs = {"side": a.side}
    out = Path(a.out)
    try:
        import torch
        import diffusers
        obs["torch"], obs["hip"], obs["diffusers"] = torch.__version__, torch.version.hip, diffusers.__version__
        cfg = diffusers.AutoencoderKLQwenImage21.load_config(a.dir, subfolder = "vae")
        vae = diffusers.AutoencoderKLQwenImage21.from_pretrained(a.dir, subfolder = "vae", torch_dtype = torch.bfloat16)
        vae = vae.to("cuda").eval()
        vae.disable_tiling()
        obs["use_tiling"] = bool(getattr(vae, "use_tiling", False))
        z_dim = int(cfg.get("z_dim") or 64)
        f = 2 ** (len(cfg.get("dim_mult") or [1, 2, 4, 8, 8]) - 1)
        lat = (1, z_dim, 1, a.side // f, a.side // f)
        obs["latent"] = list(lat)
        g = torch.Generator(device = "cuda").manual_seed(5)
        z = torch.randn(lat, generator = g, device = "cuda").to(torch.bfloat16)
        obs["stage"] = "decode"
        out.write_text(json.dumps(obs), encoding = "utf-8")  # a hang leaves this "decode" record behind
        with torch.no_grad():
            torch.cuda.synchronize()
            t = time.perf_counter()
            y = vae.decode(z).sample
            torch.cuda.synchronize()
            obs["decode_s"] = round(time.perf_counter() - t, 3)
        obs["out_shape"] = list(y.shape)
        obs["finite"] = bool(torch.isfinite(y).all())
        obs["md5"] = hashlib.md5(y.float().cpu().numpy().tobytes()).hexdigest()
        obs["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        obs["stage"] = "done"
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-1500:]
    out.write_text(json.dumps(obs), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
