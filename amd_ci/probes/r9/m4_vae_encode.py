#!/usr/bin/env python3
"""M4: FLUX.3 klein reference ``vae.encode`` run-to-run determinism on ROCm (observes only).

One fresh process per arm. Loads only the FLUX.3 klein VAE (AutoencoderKLFlux2, public unsloth/flux-3-image-klein
``vae/``) in --dtype, builds two fixed reference images from a seeded CPU generator (1024x1024 and 768x1344, smooth
gradients + noise, in [-1, 1]), and encodes each 1 warm-up + --reps times exactly as the FLUX edit path does
(``latent_dist.mode()``, the argmax latent). Arm U: cuDNN / MIOpen flags as the process has them (Studio's FLUX.3
family runs cudnn.benchmark False). Arm D: each encode inside the scoped context of sglang #51466
(``cudnn.deterministic = True, cudnn.benchmark = False``, restored in ``finally``). Records per encode the latent
sha256 and wall ms, and the flags actually in force inside the encode (engagement proof).

  python m4_vae_encode.py --repo-dir SNAPSHOT --arm U|D --dtype bf16|fp32 --reps 5 --out o.json
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time


def _save(path, obj):
    with open(path, "w", encoding = "utf-8") as fh:
        json.dump(obj, fh, indent = 1, default = str)


def ref_image(torch, h, w, seed):
    g = torch.Generator().manual_seed(seed)
    yy = torch.linspace(-1, 1, h).view(h, 1)
    xx = torch.linspace(-1, 1, w).view(1, w)
    base = torch.stack([torch.sin(3.1 * xx + 2 * yy), torch.cos(2.3 * yy - xx), xx * yy])
    noise = torch.rand(3, h, w, generator = g) * 0.6 - 0.3
    return (base * 0.7 + noise).clamp(-1, 1).unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required = True)
    ap.add_argument("--arm", choices = ("U", "D"), required = True)
    ap.add_argument("--dtype", choices = ("bf16", "fp32"), default = "bf16")
    ap.add_argument("--reps", type = int, default = 5)
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    out: dict = {"arm": a.arm, "dtype": a.dtype, "reps": a.reps, "verdict": "running", "images": {}}
    try:
        import torch
        import diffusers
        p = torch.cuda.get_device_properties(0)
        out.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None), diffusers = diffusers.__version__,
                   arch = getattr(p, "gcnArchName", None), platform = sys.platform)
        cudnn = torch.backends.cudnn
        cudnn.benchmark = False  # Studio's FLUX.3 family flag (cudnn_benchmark = False)
        out["flags_process"] = {"deterministic": bool(cudnn.deterministic), "benchmark": bool(cudnn.benchmark),
                                "enabled": bool(cudnn.enabled)}
        dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
        t = time.perf_counter()
        vae = diffusers.AutoencoderKLFlux2.from_pretrained(a.repo_dir, subfolder = "vae", torch_dtype = dt).cuda().eval()
        out["load_s"] = round(time.perf_counter() - t, 2)

        @contextlib.contextmanager
        def scoped():
            if a.arm != "D":
                yield
                return
            saved = (cudnn.deterministic, cudnn.benchmark)
            cudnn.deterministic, cudnn.benchmark = True, False
            try:
                yield
            finally:
                cudnn.deterministic, cudnn.benchmark = saved
        for name, (h, w) in (("sq1024", (1024, 1024)), ("tall768x1344", (1344, 768))):
            img = ref_image(torch, h, w, 7 if name == "sq1024" else 11).to("cuda", dt)
            recs = []
            out["images"][name] = recs
            for i in range(a.reps + 1):
                torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.inference_mode(), scoped():
                    inside = {"deterministic": bool(cudnn.deterministic), "benchmark": bool(cudnn.benchmark)}
                    lat = vae.encode(img).latent_dist.mode()
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - t) * 1e3
                buf = lat.float().contiguous().cpu()
                recs.append({"i": i, "warm": i == 0, "ms": round(ms, 3), "inside": inside,
                             "sha": hashlib.sha256(buf.numpy().tobytes()).hexdigest()[:20],
                             "shape": list(lat.shape), "finite": bool(torch.isfinite(buf).all())})
                if i == 0:
                    first = buf
                else:
                    recs[-1]["maxabs_vs_warm"] = float((buf - first).abs().max())
                _save(a.out, out)
            out["flags_after"] = {"deterministic": bool(cudnn.deterministic), "benchmark": bool(cudnn.benchmark)}
        out["verdict"] = "ok"
    except Exception:  # noqa: BLE001
        import traceback
        out["verdict"] = "error"
        out["error"] = traceback.format_exc()[-2000:]
    _save(a.out, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
