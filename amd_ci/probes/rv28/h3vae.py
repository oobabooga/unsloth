#!/usr/bin/env python3
"""R3b-a: MiniMax-H3 video VAE decode, stock vs Studio's Triton fast path, one fresh process per arm.

  h3vae.py --backend <tree>/studio/backend --dir MODEL --arm stock|fast|fast_noint8|fp32 --out F [--save OUT.pt]

(rv28 copy: arm fp32 = the reference, VAE in float32 with no Studio speedups, decoding the same bf16-rounded latent
in float32; outputs are saved as float32 so the PSNR vs the reference carries no fp16 storage rounding.)

Every arm loads the VAE in bf16 and calls the tree's ``apply_h3_vae_speedups(vae, speed_mode="default")`` (on the
base tree that engages nothing on ROCm); fast_noint8 sets UNSLOTH_H3_VAE_INT8=0 (the caller's env). 480p 5 s:
latent [1, 24, 37, 30, 52] (121 frames at 832x480). One warm decode, then 2 timed."""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--dir", required = True)
    ap.add_argument("--arm", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--save")
    ap.add_argument("--reps", type = int, default = 2)
    ap.add_argument("--lat-t", type = int, default = 37, help = "latent frames (37 = 121 frames at 832x480)")
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    import torch
    import diffusers

    obs = {"arm": a.arm, "backend": a.backend, "torch": torch.__version__, "diffusers": diffusers.__version__}
    try:
        from core.inference import video_minimax_h3_vae as H
        obs["module_file"] = H.__file__
        fp32 = a.arm == "fp32"
        vdt = torch.float32 if fp32 else torch.bfloat16
        vae = diffusers.AutoencoderKLMiniMaxH3.from_pretrained(a.dir, subfolder = "vae", torch_dtype = vdt)
        vae = vae.to("cuda").eval()
        obs["fast_path_available"] = bool(H.cuda_fast_path_available(vae))
        obs["levers"] = [] if fp32 else list(H.apply_h3_vae_speedups(vae, speed_mode = "default", workflow = "t2v",
                                                                     consumer_gpu = True, logger = None))
        obs["vae_dtype"] = str(vdt)
        g = torch.Generator(device = "cuda").manual_seed(11)
        z = torch.randn((1, 24, a.lat_t, 30, 52), generator = g, device = "cuda", dtype = torch.float32).to(torch.bfloat16)
        z = z.to(vdt)
        with torch.no_grad():
            t = time.perf_counter()
            y = vae.decode(z).sample
            torch.cuda.synchronize()
            obs["first_s"] = time.perf_counter() - t
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
        obs["nonfinite"] = int((~torch.isfinite(y)).sum())
        dec = getattr(vae, "decoder", None)
        enc = getattr(vae, "encoder", None)
        obs["decoder_fast_failed"] = bool(getattr(dec, "_unsloth_fast_failed", False))
        obs["encoder_fast_failed"] = bool(getattr(enc, "_unsloth_fast_failed", False))
        if a.save:
            torch.save(y.float().clamp(-1, 1).contiguous().cpu(), a.save)
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-2000:]
    open(a.out, "w", encoding = "utf-8").write(json.dumps(obs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
