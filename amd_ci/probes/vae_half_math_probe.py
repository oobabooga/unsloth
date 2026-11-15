#!/usr/bin/env python3
"""Probe: Studio's ROCm VAE attention half-reduction math (unsloth #17252) on this GPU, for one checkout.

Observes only (judged by criteria/vae_half_math.py). Uses the checkout's own
``core.inference.diffusion_vae_attn_chunked`` (C):

  attn cells   bf16 q/k/v [1, 1, S, 512] (the FLUX VAE mid-block: one head of 512) through ``C.chunked_attention_mode()``
               arms: half (default), half_aa (same again, the A/A noise arm), kill (UNSLOTH_VAE_ATTN_HALF_MATH=0, the
               fp32 upcast); plus an fp32 reference (q/k/v upcast, stock SDPA).
  vae decode   a diffusers AutoencoderKL with the FLUX.1 VAE layout (16 latent ch, 128/256/512/512), random weights
               (seeded), bf16, attention patched with ``C.patch_attention_modules``; decode a [1, 16, 128, 128] latent to
               1024x1024. Same three arms; fp32 reference = an fp32 copy of the same weights, unpatched.

Per arm: CUDA-event times (interleaved, rotating order, after warm-up), peak allocated memory, the mode's ``half`` /
``chunked`` engagement counters, sha256 of the output (cross-state bit-equality), max-abs / PSNR / LPIPS (when the
``lpips`` package imports) vs the fp32 reference. It also samples the process-global
``torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()`` at every aten op of one half-arm call (a dispatch mode),
and after it: the merged #17252 flips that flag for the duration of the call.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


def _write(path: Path, obs: dict) -> None:
    path.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")


def _sha(t) -> str:
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--tokens", default = "16384,32768")
    ap.add_argument("--reps", type = int, default = 7)
    ap.add_argument("--warmup", type = int, default = 2)
    ap.add_argument("--device", default = "cuda:0", help = "cpu: harness smoke test only (no timing evidence)")
    args = ap.parse_args()

    t0 = time.time()
    obs: dict = {"state": args.state, "cells": [], "vae": {}}
    sys.path.insert(0, str(args.checkout / "studio" / "backend"))
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils._python_dispatch import TorchDispatchMode

        if args.device != "cpu":
            props = torch.cuda.get_device_properties(0)
            obs["arch"] = (getattr(props, "gcnArchName", "") or "").split(":")[0]
            obs["device_name"] = torch.cuda.get_device_name(0)
        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        from core.inference import diffusion_vae_attn_chunked as C

        obs["module_file"] = C.__file__
        obs["has_half_math_sdpa"] = hasattr(C, "half_math_sdpa")
        obs["has_half_reduction_math"] = hasattr(C, "half_reduction_math")
        get_flag = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
        obs["flag_at_start"] = bool(get_flag())
    except Exception as e:  # noqa: BLE001
        obs["setup_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        _write(args.out, obs)
        return 0

    dev = torch.device(args.device)
    if dev.type == "cpu":  # smoke test: no-op CUDA calls, wall-clock events
        class _Ev:
            def __init__(self, **_):
                self.t = 0.0

            def record(self):
                self.t = time.perf_counter()

            def elapsed_time(self, other):
                return (other.t - self.t) * 1e3

        torch.cuda.synchronize = lambda *a: None
        torch.cuda.reset_peak_memory_stats = lambda *a: None
        torch.cuda.memory_allocated = lambda *a: 0
        torch.cuda.max_memory_allocated = lambda *a: 0
        torch.cuda.empty_cache = lambda: None
        torch.cuda.Event = _Ev
        C.fused_sdpa_available = lambda q, k, v: False
        C._half_math_call = lambda q: q.dtype == torch.bfloat16 and C.half_math_enabled()
    env = C.VAE_ATTN_HALF_MATH_ENV

    class FlagSampler(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.samples = []

        def __torch_dispatch__(self, func, types_, args = (), kwargs = None):
            self.samples.append(bool(get_flag()))
            return func(*args, **(kwargs or {}))

    def run_arm(name: str, fn):
        """One call of an arm; returns (output, mode counters) with the kill switch set per arm."""
        if name == "kill":
            os.environ[env] = "0"
        else:
            os.environ.pop(env, None)
        try:
            return fn()
        finally:
            os.environ.pop(env, None)

    def timed(arms: dict, reps: int, warmup: int) -> dict:
        res = {n: {"times_ms": [], "peak_mib": 0.0} for n in arms}
        for _ in range(warmup):
            for n, fn in arms.items():
                run_arm(n, fn)
        torch.cuda.synchronize()
        names = list(arms)
        for rep in range(reps):
            order = names[rep % len(names):] + names[:rep % len(names)]
            for n in order:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base_alloc = torch.cuda.memory_allocated()
                s, e = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                s.record()
                out = run_arm(n, arms[n])
                e.record()
                torch.cuda.synchronize()
                res[n]["times_ms"].append(round(s.elapsed_time(e), 4))
                res[n]["peak_mib"] = max(res[n]["peak_mib"],
                                         round((torch.cuda.max_memory_allocated() - base_alloc) / 2**20, 1))
                del out
        for n in res:
            ts = res[n]["times_ms"]
            res[n]["median_ms"] = round(statistics.median(ts), 4)
            res[n]["spread_rel"] = round((max(ts) - min(ts)) / statistics.median(ts), 4)
        return res

    def numerics(out, ref) -> dict:
        d = (out.float() - ref.float()).abs()
        return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "finite": bool(torch.isfinite(out).all())}

    # ---- attention cells
    for S in [int(x) for x in args.tokens.split(",") if x]:
        cell: dict = {"tokens": S, "head_dim": 512}
        try:
            g = torch.Generator(device = dev).manual_seed(1000 + S)
            q, k, v = (torch.randn(1, 1, S, 512, device = dev, generator = g).to(torch.bfloat16) for _ in range(3))
            cell["fused_sdpa_available"] = bool(C.fused_sdpa_available(q, k, v))
            modes: dict = {}

            def make(name):
                def fn():
                    m = C.chunked_attention_mode()
                    with m, torch.no_grad():
                        out = F.scaled_dot_product_attention(q, k, v)
                    modes[name] = {"half": m.half, "chunked": m.chunked}
                    return out
                return fn

            arms = {n: make(n) for n in ("half", "half_aa", "kill")}
            with torch.no_grad():
                ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
            outs = {n: run_arm(n, fn) for n, fn in arms.items()}
            torch.cuda.synchronize()
            cell["engagement"] = dict(modes)
            cell["sha256"] = {n: _sha(o) for n, o in outs.items()}
            cell["vs_fp32"] = {n: numerics(o, ref) for n, o in outs.items()}
            cell["half_aa_bitwise"] = bool(torch.equal(outs["half"], outs["half_aa"]))
            del outs, ref
            sampler = FlagSampler()
            with sampler:
                run_arm("half", arms["half"])
            torch.cuda.synchronize()
            cell["flag_samples_during_half"] = {"n": len(sampler.samples), "true": sum(sampler.samples)}
            cell["flag_after_half"] = bool(get_flag())
            cell["timing"] = timed(arms, args.reps, args.warmup)
            cell["engagement_last"] = dict(modes)
        except Exception as e:  # noqa: BLE001
            cell["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        torch.cuda.empty_cache()
        obs["cells"].append(cell)
        _write(args.out, obs)
        print(f"attn S={S}: {cell.get('timing', {}).get('half', {}).get('median_ms')} "
              f"kill {cell.get('timing', {}).get('kill', {}).get('median_ms')} {cell.get('error', '')}", flush = True)

    # ---- full VAE decode (random weights, FLUX.1 VAE layout)
    vae_obs = obs["vae"]
    try:
        import diffusers
        from diffusers import AutoencoderKL

        vae_obs["diffusers"] = diffusers.__version__
        torch.manual_seed(0)
        vae32 = AutoencoderKL(
            in_channels = 3, out_channels = 3, latent_channels = 16,
            down_block_types = ("DownEncoderBlock2D",) * 4, up_block_types = ("UpDecoderBlock2D",) * 4,
            block_out_channels = (128, 256, 512, 512), layers_per_block = 2, norm_num_groups = 32,
        ).eval().to(dev)
        vae = copy.deepcopy(vae32).to(torch.bfloat16)
        vae_obs["patched_modules"] = C.patch_attention_modules(vae)
        z = torch.randn(1, 16, 128, 128, device = dev, generator = torch.Generator(device = dev).manual_seed(5))
        counters = {"half": 0, "chunked": 0}
        real_mode = C.chunked_attention_mode

        def counting_mode(*a, **kw):
            m = real_mode(*a, **kw)
            counters.setdefault("_modes", []).append(m)
            return m

        C.chunked_attention_mode = counting_mode  # observe the engagement counters of the patched forward's mode

        def drain() -> dict:
            ms = counters.pop("_modes", [])
            return {"half": sum(m.half for m in ms), "chunked": sum(m.chunked for m in ms), "calls": len(ms)}

        def decode():
            with torch.no_grad():
                return vae.decode(z.to(torch.bfloat16)).sample

        arms = {"half": decode, "half_aa": decode, "kill": decode}
        with torch.no_grad():
            ref = vae32.decode(z).sample
        drain()
        outs, eng = {}, {}
        for n, fn in arms.items():
            outs[n] = run_arm(n, fn)
            eng[n] = drain()
        torch.cuda.synchronize()
        vae_obs["engagement"] = eng
        vae_obs["sha256"] = {n: _sha(o) for n, o in outs.items()}
        vae_obs["half_aa_bitwise"] = bool(torch.equal(outs["half"], outs["half_aa"]))
        img = lambda t: (t.float() / 2 + 0.5).clamp(0, 1)  # noqa: E731
        ref_img = img(ref)
        num = {}
        for n, o in outs.items():
            d = numerics(o, ref)
            mse = float(((img(o) - ref_img) ** 2).mean())
            d["psnr_db"] = round(10 * math.log10(1.0 / mse), 2) if mse > 0 else float("inf")
            num[n] = d
        vae_obs["ref_abs_max"] = float(ref.float().abs().max())
        try:
            import lpips

            net = lpips.LPIPS(net = "alex", verbose = False).to(dev).eval()
            with torch.no_grad():
                for n, o in outs.items():
                    num[n]["lpips"] = float(net(img(o) * 2 - 1, ref_img * 2 - 1).item())
            del net
        except Exception as e:  # noqa: BLE001
            vae_obs["lpips_error"] = f"{type(e).__name__}: {e}"[:300]
        vae_obs["vs_fp32"] = num
        del outs, ref, vae32
        torch.cuda.empty_cache()
        sampler = FlagSampler()
        with sampler:
            run_arm("half", decode)
        torch.cuda.synchronize()
        drain()
        vae_obs["flag_samples_during_half"] = {"n": len(sampler.samples), "true": sum(sampler.samples)}
        vae_obs["flag_after_half"] = bool(get_flag())
        vae_obs["timing"] = timed(arms, args.reps, args.warmup)
        drain()
        C.chunked_attention_mode = real_mode
    except Exception as e:  # noqa: BLE001
        vae_obs["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    obs["flag_at_end"] = bool(get_flag())
    obs["elapsed_s"] = round(time.time() - t0, 1)
    _write(args.out, obs)
    print(f"vae: {vae_obs.get('timing', {}).get('half', {}).get('median_ms')} "
          f"kill {vae_obs.get('timing', {}).get('kill', {}).get('median_ms')} {vae_obs.get('error', '')}", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
