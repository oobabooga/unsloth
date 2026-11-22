#!/usr/bin/env python3
"""Probe: FLUX.2-class VAE whole-map decode of a would-be-tiled latent, for one checkout.

Observes only (judged by criteria/flux2_full_map.py). Uses the checkout's own ``core.inference.diffusion_vae_tiling``.

Real FLUX.3 Klein VAE (diffusers ``AutoencoderKLFlux2``, pinned revision, no token) in bf16. A real photo
(huggingface/documentation-images ``diffusers/sdxl_reference_input_cat.jpg``, pinned revision), cover-resized to
``--size`` (2048), as 2x-1, encoded untiled (``latent_dist.mode()``). The latent is decoded three ways:

  untiled      ``vae.use_tiling = False`` (the reference)
  tiled        ``vae.enable_tiling(); diffusion_vae_tiling.install(vae)``, kill switch unset
  tiled_kill   the same with ``UNSLOTH_FLUX_VAE_FULL_MAP=0``

plus a ``--small-size`` (1024) photo decoded with tiling enabled + installed (the tiled path not taken; its sha256
is compared base vs head by the criteria, and repeated in-state as an A/A determinism check).

Per decode: PSNR (8-bit levels) vs the untiled decode and vs the photo, ``vae._unsloth_vae_full_map_calls`` delta,
``tiled_decode`` calls, peak allocated MiB over the pre-decode allocation, wall time, sha256 of the 8-bit image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

VAE_REPO = "unsloth/flux-3-image-klein-FP8"
PHOTO_REPO = "huggingface/documentation-images"
PHOTO_FILE = "diffusers/sdxl_reference_input_cat.jpg"
PHOTO_REV = "9a5332498b493c7a37c9dfc167b710bfe8ebcf9b"
FULL_MAP_ENV = "UNSLOTH_FLUX_VAE_FULL_MAP"


def _write(path: Path, obs: dict) -> None:
    path.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")


def _vae_revision(checkout: Path) -> str:
    cfg = checkout / "studio" / "backend" / "tests" / "fixtures" / "vae_tile_configs.json"
    return json.loads(cfg.read_text(encoding = "utf-8"))["families"]["flux.3-klein"]["revision"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--size", type = int, default = 2048)
    ap.add_argument("--small-size", type = int, default = 1024)
    ap.add_argument("--device", default = "cuda:0", help = "cpu: harness smoke test only")
    ap.add_argument("--dtype", default = "bfloat16")
    ap.add_argument("--tile-latent", type = int, default = 0,
                    help = "smoke only: override the stock tile_latent_min_size so a small --size tiles")
    args = ap.parse_args()

    t0 = time.time()
    obs: dict = {"state": args.state, "checkout": str(args.checkout.resolve()), "size": args.size,
                 "small_size": args.small_size, "device": args.device, "decodes": {}}
    os.environ.pop(FULL_MAP_ENV, None)
    sys.path.insert(0, str(args.checkout / "studio" / "backend"))
    try:
        import numpy as np
        import torch
        from huggingface_hub import hf_hub_download
        from PIL import Image

        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        if args.device != "cpu":
            props = torch.cuda.get_device_properties(0)
            obs["arch"] = (getattr(props, "gcnArchName", "") or "").split(":")[0]
            obs["device_name"] = torch.cuda.get_device_name(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import diffusers
        from diffusers import AutoencoderKLFlux2

        obs["diffusers"] = diffusers.__version__
        from core.inference import diffusion_vae_strips as S
        from core.inference import diffusion_vae_tiling as T
        from core.inference import diffusion_vae_fused as FU
        from core.inference import diffusion_vae_attn_chunked as CH

        real_mode = CH.chunked_attention_mode
        modes: list = []

        def counting_mode(*a, **kw):
            m = real_mode(*a, **kw)
            modes.append(m)
            return m

        CH.chunked_attention_mode = counting_mode  # observe the chunked mode's engagement (strips call it by attribute)

        obs["module_file"] = os.path.realpath(T.__file__)
        obs["strips_module_file"] = S.__file__
        obs["has_full_map_hook"] = hasattr(T, "full_map_decode")
        obs["has_full_map_strips"] = hasattr(S, "full_map_decode")
        try:
            obs["fused_runtime_ok"] = bool(FU.runtime_ok())
        except Exception as e:  # noqa: BLE001
            obs["fused_runtime_ok"] = f"{type(e).__name__}: {e}"[:200]
        rev = _vae_revision(args.checkout)
        obs["vae_revision"] = rev
        dev = torch.device(args.device)
        dtype = getattr(torch, args.dtype)
        vae = AutoencoderKLFlux2.from_pretrained(VAE_REPO, subfolder = "vae", revision = rev, token = False,
                                                 torch_dtype = dtype).to(dev).eval()
        obs["vae_class"] = type(vae).__name__
        photo_path = hf_hub_download(PHOTO_REPO, PHOTO_FILE, repo_type = "dataset", revision = PHOTO_REV,
                                     token = False)
        photo = Image.open(photo_path).convert("RGB")
        obs["photo_size"] = list(photo.size)
    except Exception as e:  # noqa: BLE001
        obs["setup_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
        _write(args.out, obs)
        return 0

    is_cuda = dev.type == "cuda"

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    def cover(size: int):
        w, h = photo.size
        s = max(size / w, size / h)
        nw, nh = max(size, round(w * s)), max(size, round(h * s))
        im = photo.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - size) // 2, (nh - size) // 2
        return np.asarray(im.crop((left, top, left + size, top + size)), dtype = np.uint8)

    def to8(x) -> "np.ndarray":
        return ((x[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    def psnr(a, b) -> float:
        mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
        return round(10 * math.log10(255.0 ** 2 / mse), 3) if mse > 0 else float("inf")

    def encode(img8):
        x = torch.from_numpy(img8).permute(2, 0, 1)[None].to(dev, torch.float32) / 127.5 - 1
        vae.use_tiling = False
        with torch.no_grad():
            z = vae.encode(x.to(dtype)).latent_dist.mode()
        sync()
        return z

    calls = {"tiled": 0}

    def counter() -> int:
        return int(vae.__dict__.get("_unsloth_vae_full_map_calls", 0))

    def decode(z, tiling: bool, kill: bool) -> tuple:
        vae.use_tiling = tiling
        if kill:
            os.environ[FULL_MAP_ENV] = "0"
        else:
            os.environ.pop(FULL_MAP_ENV, None)
        rec: dict = {"tiling": tiling, "kill_switch": kill}
        try:
            c0, t_calls0 = counter(), calls["tiled"]
            modes.clear()
            if is_cuda and tiling and obs.get("has_full_map_strips"):
                try:
                    rec["full_map_decode_mib"] = round(float(S.full_map_decode_mib(vae, z)), 1)
                except Exception as e:  # noqa: BLE001
                    rec["full_map_decode_mib"] = f"{type(e).__name__}: {e}"[:200]
            if is_cuda:
                sync()
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info(dev)
                rec["free_mib_before"] = round(free / 2**20, 1)
                torch.cuda.reset_peak_memory_stats(dev)
                base_alloc = torch.cuda.memory_allocated(dev)
            t = time.perf_counter()
            with torch.no_grad():
                out = vae.decode(z).sample
            sync()
            rec["time_s"] = round(time.perf_counter() - t, 3)
            if is_cuda:
                rec["peak_alloc_mib"] = round((torch.cuda.max_memory_allocated(dev) - base_alloc) / 2**20, 1)
            rec["full_map_calls"] = counter() - c0
            rec["tiled_decode_calls"] = calls["tiled"] - t_calls0
            rec["chunked_modes"] = len(modes)
            rec["chunked_sdpa_calls"] = int(sum(getattr(m, "chunked", 0) for m in modes))
            rec["chunked_half_calls"] = int(sum(getattr(m, "half", 0) for m in modes))
            modes.clear()
            rec["out_shape"] = list(out.shape)
            rec["finite"] = bool(torch.isfinite(out).all())
            img = to8(out)
            rec["sha256"] = hashlib.sha256(img.tobytes()).hexdigest()
            del out
            return img, rec
        finally:
            os.environ.pop(FULL_MAP_ENV, None)

    D = obs["decodes"]
    try:
        photo_big = cover(args.size)
        z = encode(photo_big)
        obs["latent_shape"] = list(z.shape)
        obs["latent_sha256"] = hashlib.sha256(z.float().cpu().numpy().tobytes()).hexdigest()
        ref, D["untiled"] = decode(z, tiling = False, kill = False)
        D["untiled"]["psnr_vs_photo"] = psnr(ref, photo_big)
        _write(args.out, obs)

        vae.enable_tiling()
        if args.tile_latent:
            vae.tile_latent_min_size = args.tile_latent
            vae.tile_sample_min_size = args.tile_latent * 8
        obs["stock_tile_latent"] = int(vae.tile_latent_min_size)
        obs["installed"] = bool(T.install(vae))
        inner = vae.tiled_decode

        def counted(*a, **kw):
            calls["tiled"] += 1
            return inner(*a, **kw)

        vae.tiled_decode = counted
        if is_cuda and obs["has_full_map_strips"]:
            try:
                obs["full_map_decode_mib_estimate"] = round(float(S.full_map_decode_mib(vae, z)), 1)
            except Exception as e:  # noqa: BLE001
                obs["full_map_decode_mib_estimate"] = f"{type(e).__name__}: {e}"[:200]
        for name, kill in (("tiled", False), ("tiled_kill", True)):
            try:
                img, D[name] = decode(z, tiling = True, kill = kill)
                D[name]["psnr_vs_untiled"] = psnr(img, ref)
                D[name]["psnr_vs_photo"] = psnr(img, photo_big)
                D[name]["last_decode_tile"] = list(vae.__dict__.get("_unsloth_last_decode_tile") or [])
                del img
            except Exception as e:  # noqa: BLE001
                D[name] = {"error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"}
            _write(args.out, obs)
        del z, ref, photo_big
        if is_cuda:
            torch.cuda.empty_cache()

        photo_small = cover(args.small_size)
        zs = encode(photo_small)
        obs["small_latent_shape"] = list(zs.shape)
        obs["small_latent_sha256"] = hashlib.sha256(zs.float().cpu().numpy().tobytes()).hexdigest()
        for name in ("small", "small_aa"):
            img, D[name] = decode(zs, tiling = True, kill = False)
            D[name]["psnr_vs_photo"] = psnr(img, photo_small)
            del img
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
    obs["elapsed_s"] = round(time.time() - t0, 1)
    _write(args.out, obs)
    print(json.dumps({k: {kk: v.get(kk) for kk in ("psnr_vs_untiled", "full_map_calls", "time_s")}
                      for k, v in D.items()}), flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
