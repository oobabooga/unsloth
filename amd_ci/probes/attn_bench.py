#!/usr/bin/env python3
"""Fused attention candidates on a ROCm card, timed at DiT shapes: attention alone AND a whole synthetic DiT block
(LayerNorm -> QKV -> attention -> out proj -> LayerNorm -> 4x GELU MLP, residuals), bf16, B=1, head_dim 128.

Arms (each recorded with an error string when it cannot run):
  sdpa_default   F.scaled_dot_product_attention on (B,H,S,D) transposed views of BSHD projections: what diffusers'
                 native backend (Studio's ROCm default) does.
  sdpa_flash / sdpa_efficient   the same, forced through sdpa_kernel.
  sdpa_ck        sdpa default with torch.backends.cuda.preferred_rocm_fa_library("ck") (torch's own CK FA, if built).
  fa             DAO-AILab flash_attn_func on BSHD (CK or Triton build, decided by FLASH_ATTENTION_TRITON_AMD_ENABLE
                 at import; recorded as fa_backend).
  aiter_hub      kernels-community/aiter-flash-attn (aiter Triton MHA) via `kernels`, when that package is importable.
Accuracy: every arm vs fp32 math at S=1024 H=4, and vs sdpa_default at the real shape. JSON to --out only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

SHAPES = {
    # name: (heads, tokens, hidden)
    "klein_1024": (24, 4096 + 512, 3072),  # FLUX.2-klein-4B 1024^2: 64x64 image tokens + 512 text tokens
    "zimage_1024": (30, 4096 + 256, 3840),  # Z-Image 1024^2: 4096 image tokens + padded caption
    "wan5b_704x1280x121": (24, 31 * 44 * 40 // 2, 3072),  # Wan2.2-TI2V-5B 704x1280x121 -> 27280 tokens
}


def err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--iters", type = int, default = 20)
    ap.add_argument("--block-iters", type = int, default = 10)
    ap.add_argument("--skip-video-block", action = "store_true")
    args = ap.parse_args()
    out: dict = {"env": {k: os.environ.get(k) for k in ("FLASH_ATTENTION_TRITON_AMD_ENABLE",
                                                        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")}}
    try:
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        out["torch"] = torch.__version__
        out["hip"] = getattr(torch.version, "hip", None)
        p = torch.cuda.get_device_properties(0)
        out["device"] = {"name": torch.cuda.get_device_name(0), "arch": getattr(p, "gcnArchName", None)}
        dev, dt = "cuda", torch.bfloat16

        def med_ms(fn, iters) -> float:
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(iters):
                a = torch.cuda.Event(enable_timing = True)
                b = torch.cuda.Event(enable_timing = True)
                a.record()
                fn()
                b.record()
                b.synchronize()
                ts.append(a.elapsed_time(b))
            ts.sort()
            return round(ts[len(ts) // 2], 3)

        arms: dict = {}

        def sdpa_with(backend):
            def f(q, k, v):  # q,k,v BSHD
                qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
                if backend is None:
                    o = F.scaled_dot_product_attention(qt, kt, vt)
                else:
                    with sdpa_kernel([backend]):
                        o = F.scaled_dot_product_attention(qt, kt, vt)
                return o.transpose(1, 2)
            return f

        arms["sdpa_default"] = sdpa_with(None)
        arms["sdpa_flash"] = sdpa_with(SDPBackend.FLASH_ATTENTION)
        arms["sdpa_efficient"] = sdpa_with(SDPBackend.EFFICIENT_ATTENTION)
        try:
            out["preferred_rocm_fa_library_default"] = str(torch.backends.cuda.preferred_rocm_fa_library())
        except Exception as exc:  # noqa: BLE001
            out["preferred_rocm_fa_library_default"] = err(exc)

        def sdpa_ck(q, k, v):
            prev = torch.backends.cuda.preferred_rocm_fa_library()
            torch.backends.cuda.preferred_rocm_fa_library("ck")
            try:
                return sdpa_with(SDPBackend.FLASH_ATTENTION)(q, k, v)
            finally:
                torch.backends.cuda.preferred_rocm_fa_library(prev)
        arms["sdpa_ck"] = sdpa_ck

        try:
            import importlib.metadata as md

            import flash_attn
            from flash_attn import flash_attn_func
            out["fa_version"] = md.version("flash_attn")
            out["fa_module"] = str(Path(flash_attn.__file__).parent)
            try:
                from flash_attn import flash_attn_interface as fai
                out["fa_backend"] = "triton" if getattr(fai, "USE_TRITON_ROCM", False) else "ck"
            except Exception as exc:  # noqa: BLE001
                out["fa_backend"] = err(exc)
            arms["fa"] = lambda q, k, v: flash_attn_func(q, k, v)
        except Exception as exc:  # noqa: BLE001
            out["fa_error"] = err(exc)

        try:
            from kernels import get_kernel
            hub, tried = None, []
            for repo, ver in (("kernels-community/aiter-flash-attn", 1), ("kernels-community/aiter-flash-attn-ck", 1)):
                try:
                    hub = get_kernel(repo, version = ver)
                    out["aiter_hub_repo"] = repo
                    break
                except Exception as exc:  # noqa: BLE001
                    tried.append(f"{repo}: {err(exc)}")
            if hub is None:
                raise RuntimeError("; ".join(tried))
            fn = getattr(hub, "flash_attn_func", None)
            if fn is None:
                raise RuntimeError(f"no flash_attn_func in {dir(hub)[:30]}")

            def aiter_hub(q, k, v, fn = fn, hub = hub):
                try:
                    r = fn(q, k, v)
                except FileNotFoundError as exc:
                    # The hub build ships tuned configs for gfx942 / gfx950 / gfx1250 only. Borrow the RDNA (gfx1250)
                    # one under this arch's name, recorded, so the kernel itself can be timed.
                    missing = Path(str(exc.filename or ""))
                    donor = missing.with_name("gfx1250-MHA-DEFAULT.json")
                    if not missing.name.endswith("-MHA-DEFAULT.json") or not donor.exists():
                        raise
                    import shutil
                    shutil.copyfile(donor, missing)
                    out["aiter_hub_config_borrowed"] = f"{donor.name} -> {missing.name}"
                    r = fn(q, k, v)
                return r[0] if isinstance(r, (tuple, list)) else r
            arms["aiter_hub"] = aiter_hub
        except Exception as exc:  # noqa: BLE001
            out["aiter_hub_error"] = err(exc)

        def rand(shape, seed):
            g = torch.Generator(device = "cpu").manual_seed(seed)
            return torch.randn(*shape, generator = g).to(dev, dt)

        # Accuracy vs fp32 math at a small shape.
        q, k, v = (rand((1, 1024, 4, 128), s) for s in (1, 2, 3))
        k = k + 2.0 * rand((1, 1, 1, 128), 4)  # per-channel key offset, like real keys
        with sdpa_kernel([SDPBackend.MATH]):
            ref = F.scaled_dot_product_attention(*(x.float().transpose(1, 2) for x in (q, k, v))).transpose(1, 2)
        res: dict = {name: {} for name in arms}
        for name, fn in arms.items():
            try:
                o = fn(q, k, v).float()
                torch.cuda.synchronize()
                d = (o - ref).abs()
                res[name]["acc_small"] = {"max_abs": round(float(d.max()), 5), "mean_abs": round(float(d.mean()), 6)}
            except Exception as exc:  # noqa: BLE001
                res[name]["error"] = err(exc)
        for shape, (heads, tokens, hidden) in SHAPES.items():
            q, k, v = (rand((1, tokens, heads, 128), s) for s in (11, 12, 13))
            base_o = None
            for name, fn in arms.items():
                if "error" in res[name]:
                    continue
                r: dict = {}
                try:
                    o = fn(q, k, v)
                    torch.cuda.synchronize()
                    if base_o is None and name == "sdpa_default":
                        base_o = o.float()
                    elif base_o is not None:
                        d = (o.float() - base_o).abs()
                        r["vs_default"] = {"max_abs": round(float(d.max()), 5), "mean_abs": round(float(d.mean()), 6)}
                    del o
                    r["attn_ms"] = med_ms(lambda: fn(q, k, v), args.iters)
                except Exception as exc:  # noqa: BLE001
                    r["error"] = err(exc)
                res[name][shape] = r
                torch.cuda.empty_cache()
            del q, k, v, base_o
            torch.cuda.empty_cache()

        # Whole synthetic DiT block.
        for shape, (heads, tokens, hidden) in SHAPES.items():
            if args.skip_video_block and shape.startswith("wan"):
                continue
            torch.manual_seed(0)
            ln1 = torch.nn.LayerNorm(hidden, elementwise_affine = False).to(dev, dt)
            qkv = torch.nn.Linear(hidden, 3 * hidden).to(dev, dt)
            proj = torch.nn.Linear(hidden, hidden).to(dev, dt)
            ln2 = torch.nn.LayerNorm(hidden, elementwise_affine = False).to(dev, dt)
            fc1 = torch.nn.Linear(hidden, 4 * hidden).to(dev, dt)
            fc2 = torch.nn.Linear(4 * hidden, hidden).to(dev, dt)
            for m in (qkv, proj, fc1, fc2):
                torch.nn.init.normal_(m.weight, std = hidden ** -0.5)
            x = rand((1, tokens, hidden), 21)

            def block(attn, x = x):
                with torch.no_grad():
                    h = qkv(ln1(x)).view(1, tokens, 3, heads, hidden // heads)
                    a = attn(h[:, :, 0], h[:, :, 1], h[:, :, 2]).reshape(1, tokens, hidden)
                    x2 = x + proj(a)
                    return x2 + fc2(F.gelu(fc1(ln2(x2)), approximate = "tanh"))

            base_y = None
            for name, fn in arms.items():
                if "error" in res[name] or "error" in res[name].get(shape, {}):
                    continue
                r = res[name][shape]
                try:
                    y = block(fn).float()
                    if name == "sdpa_default":
                        base_y = y
                    elif base_y is not None:
                        rel = float((y - base_y).abs().mean() / base_y.abs().mean())
                        r["block_vs_default_rel_l1"] = round(rel, 6)
                    del y
                    r["block_ms"] = med_ms(lambda: block(fn), args.block_iters)
                except Exception as exc:  # noqa: BLE001
                    r["block_error"] = err(exc)
                torch.cuda.empty_cache()
            del x, base_y, ln1, qkv, proj, ln2, fc1, fc2
            torch.cuda.empty_cache()
        # Speedups vs sdpa_default.
        for name in arms:
            for shape in SHAPES:
                r = res[name].get(shape, {})
                b = res["sdpa_default"].get(shape, {})
                for key in ("attn_ms", "block_ms"):
                    if r.get(key) and b.get(key):
                        r[key.replace("_ms", "_speedup")] = round(b[key] / r[key], 3)
        out["results"] = res
    except Exception as exc:  # noqa: BLE001
        out["fatal"] = err(exc)
        out["tail"] = traceback.format_exc().strip().splitlines()[-6:]
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(out, indent = 1, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
