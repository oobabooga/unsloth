#!/usr/bin/env python3
"""One FLUX.2-family VAE decode arm in a fresh process (M05, AMD coverage Nov 24). Observes only.

Loads AutoencoderKLFlux2 from ``--repo`` (subfolder vae, public, token=False), decodes every latent in ``--latents``
(a list of .pt files holding {"z": fp32 tensor} or a "rand" spec) and saves each output (fp32, CPU) to
``<out>/<tag>__<latent name>.pt``. Modes:
  cpu_fp32 / cpu_bf16            reference / CPU bf16
  gpu_raw_bf16 / gpu_raw_fp32    stock diffusers decode, MIOpen immediate mode (as Studio sets it)
  gpu_studio_bf16 / _fp32        + Studio's VAE attention routing (diffusion_vae_attn_chunked.patch_attention_modules)
  gpu_math_bf16                  stock decode inside sdpa_kernel(MATH)
``--local-err``: per-module LOCAL error against a CPU fp32 twin: every Conv2d / GroupNorm / Linear / attention module /
upsampler is re-run on the CPU in fp32 from the GPU module's own input, so each number is that op's error alone
(nothing accumulated from upstream). Env (MIOPEN_DEBUG_*) is set by the caller."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path


def rel(a, b):
    import torch

    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required = True)
    ap.add_argument("--mode", required = True)
    ap.add_argument("--tag", required = True)
    ap.add_argument("--latents", required = True, help = "JSON list of [name, path] (path 'rand:SEED' for N(0,1) bf16)")
    ap.add_argument("--out", required = True)
    ap.add_argument("--studio-backend", default = "")
    ap.add_argument("--local-err", action = "store_true")
    ap.add_argument("--repeats", type = int, default = 2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents = True, exist_ok = True)
    rec: dict = {"tag": a.tag, "mode": a.mode, "repo": a.repo, "ok": False, "latents": {},
                 "env": {k: v for k, v in os.environ.items() if k.startswith(("MIOPEN_", "HIP_", "ROCBLAS", "HIPBLASLT",
                                                                              "TORCH_BLAS", "PYTORCH_"))}}
    jp = out / f"{a.tag}.json"
    try:
        import torch
        import diffusers

        rec["torch"], rec["hip"], rec["diffusers"] = torch.__version__, getattr(torch.version, "hip", None), diffusers.__version__
        cpu = a.mode.startswith("cpu")
        dev = "cpu" if cpu else "cuda"
        dt = torch.float32 if a.mode.endswith("fp32") else torch.bfloat16
        if not cpu:
            rec["gpu"] = torch.cuda.get_device_name(0)
            try:
                torch.backends.miopen.immediate = True
            except Exception:  # noqa: BLE001
                pass
            rec["miopen_immediate"] = bool(getattr(torch.backends.miopen, "immediate", None))
            rec["preferred_blas"] = str(torch.backends.cuda.preferred_blas_library())
        vae = diffusers.AutoencoderKLFlux2.from_pretrained(a.repo, subfolder = "vae", torch_dtype = dt,
                                                           token = False).to(dev).eval()
        if "studio" in a.mode:
            sys.path.insert(0, a.studio_backend)
            from core.inference import diffusion_vae_attn_chunked as C

            rec["studio_patched"] = int(C.patch_attention_modules(vae))
            rec["studio_windows_rocm"] = bool(C.windows_rocm())
            rec["studio_untrusted_head"] = bool(C.vae_has_untrusted_head(vae))
        ctx = None
        if a.mode.startswith("gpu_math"):
            from torch.nn.attention import SDPBackend, sdpa_kernel

            ctx = lambda: sdpa_kernel([SDPBackend.MATH])  # noqa: E731
        hooks = []
        local: list = []
        if a.local_err:
            twin = diffusers.AutoencoderKLFlux2.from_pretrained(a.repo, subfolder = "vae", torch_dtype = torch.float32,
                                                                token = False).eval()
            tw = dict(twin.named_modules())

            def kind(m):
                n = type(m).__name__
                if isinstance(m, (torch.nn.Conv2d, torch.nn.GroupNorm, torch.nn.Linear)):
                    return n
                if n in ("Attention", "Upsample2D"):
                    return n
                return None

            def mk(name, k):
                def hook(mod, args, kwargs, output):
                    try:
                        cargs = [x.detach().float().cpu() if torch.is_tensor(x) else x for x in args]
                        ckw = {kk: (v.detach().float().cpu() if torch.is_tensor(v) else v) for kk, v in kwargs.items()}
                        with torch.no_grad():
                            ref = tw[name](*cargs, **ckw)
                        o = output[0] if isinstance(output, (tuple, list)) else output
                        o = o.detach().float().cpu()
                        x0 = args[0] if args else next(iter(kwargs.values()))
                        local.append({"name": name, "kind": k, "rel": rel(o, ref), "max_abs": float((o - ref).abs().max()),
                                      "ref_absmax": float(ref.abs().max()), "nan": bool(torch.isnan(o).any()),
                                      "in_shape": list(x0.shape), "in_dtype": str(x0.dtype),
                                      "in_channels_last": bool(x0.dim() == 4 and x0.is_contiguous(memory_format = torch.channels_last)
                                                               and not x0.is_contiguous())})
                    except Exception as exc:  # noqa: BLE001
                        local.append({"name": name, "kind": k, "error": repr(exc)[:300]})
                return hook

            for name, m in vae.decoder.named_modules():
                k = kind(m)
                if k:
                    hooks.append(m.register_forward_hook(mk("decoder." + name if name else "decoder", k), with_kwargs = True))
            for name in ("post_quant_conv",):
                m = getattr(vae, name, None)
                if m is not None:
                    hooks.append(m.register_forward_hook(mk(name, "Conv2d"), with_kwargs = True))
        lat = json.loads(a.latents)
        for li, (lname, lpath) in enumerate(lat):
            if lpath.startswith("rand:"):
                g = torch.Generator(device = "cpu").manual_seed(int(lpath.split(":")[1]))
                z = torch.randn(1, 32, 128, 128, generator = g).to(torch.bfloat16).float()
            else:
                z = torch.load(lpath, map_location = "cpu")["z"].float()
            zz = z.to(dev, dt)
            ts = []
            reps = 1 if (cpu or a.local_err) else max(1, a.repeats)
            img = None
            for i in range(reps):
                if not cpu:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                with torch.no_grad():
                    if ctx is not None:
                        with ctx():
                            img = vae.decode(zz).sample
                    else:
                        img = vae.decode(zz).sample
                if not cpu:
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            row = {"z_shape": list(z.shape), "z_std": float(z.std()), "z_absmax": float(z.abs().max()),
                   "first_s": round(ts[0], 3), "decode_s": round(statistics.median(ts[1:] or ts), 4),
                   "finite": bool(torch.isfinite(img).all())}
            if not cpu:
                row["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            if local:
                # keep only this latent's first pass records
                mine = [x for x in local if "rel" in x]
                srt = sorted(mine, key = lambda x: -x["rel"])
                row["local_top"] = srt[:25]
                row["local_n"] = len(mine)
                row["local_by_kind"] = {}
                for x in mine:
                    d = row["local_by_kind"].setdefault(x["kind"], [])
                    d.append(x["rel"])
                row["local_by_kind"] = {k: {"n": len(v), "median": statistics.median(v), "max": max(v)}
                                        for k, v in row["local_by_kind"].items()}
                row["local_order"] = [[x["name"], round(x["rel"], 6)] for x in mine]
                row["local_errors"] = [x for x in local if "error" in x][:5]
                local.clear()
            torch.save(img.float().cpu(), out / f"{a.tag}__{lname}.pt")
            rec["latents"][lname] = row
            jp.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")
        rec["ok"] = True
    except BaseException as exc:  # noqa: BLE001
        rec["error"] = repr(exc)[:1500]
        rec["tail"] = traceback.format_exc().splitlines()[-12:]
    jp.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")
    print("VAE_ARM_DONE", a.tag, rec["ok"], flush = True)
    os._exit(0)


if __name__ == "__main__":
    main()
