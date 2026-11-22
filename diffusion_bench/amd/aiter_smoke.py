#!/usr/bin/env python3
"""AITER flash_attn_func vs torch SDPA at the FLUX.3 klein DiT attention shape on this GPU.

Observes only: import, diffusers' aiter gate, kernel output error vs an fp32 reference, and timing.
Writes JSON to --out (never stdout).
"""

import argparse
import json
import os
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--seq", type = int, default = 4608)  # 64x64 image tokens + 512 text tokens at 1024^2
    ap.add_argument("--heads", type = int, default = 24)
    ap.add_argument("--dim", type = int, default = 128)
    ap.add_argument("--iters", type = int, default = 20)
    args = ap.parse_args()
    rec = {"env": {k: os.environ.get(k) for k in ("ENABLE_CK", "AITER_JIT_DIR", "ROCM_PATH", "GPU_ARCHS")}}
    import torch
    import torch.nn.functional as F

    rec["torch"] = torch.__version__
    rec["hip"] = torch.version.hip
    rec["device"] = torch.cuda.get_device_name(0)
    rec["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    try:
        import importlib.metadata as md

        rec["aiter_version"] = md.version("amd-aiter") if _has(md, "amd-aiter") else md.version("aiter")
    except Exception as exc:  # noqa: BLE001
        rec["aiter_version"] = f"unknown: {exc}"
    try:
        t0 = time.time()
        import aiter  # noqa: F401
        from aiter import flash_attn_func

        rec["aiter_import_s"] = round(time.time() - t0, 2)
        rec["aiter_file"] = aiter.__file__
        try:
            from aiter.jit.core import ENABLE_CK

            rec["aiter_enable_ck"] = bool(ENABLE_CK)
        except Exception as exc:  # noqa: BLE001
            rec["aiter_enable_ck"] = f"unknown: {exc}"
    except Exception as exc:  # noqa: BLE001
        rec["aiter_import_error"] = repr(exc)
        rec["aiter_import_tb"] = traceback.format_exc()[-4000:]
        _write(args.out, rec)
        return
    try:
        import diffusers.models.attention_dispatch as ad

        rec["diffusers"] = __import__("diffusers").__version__
        rec["diffusers_can_use_aiter"] = bool(getattr(ad, "_CAN_USE_AITER_ATTN", False))
    except Exception as exc:  # noqa: BLE001
        rec["diffusers_error"] = repr(exc)

    torch.manual_seed(0)
    b, s, h, d = 1, args.seq, args.heads, args.dim
    q, k, v = (torch.randn(b, s, h, d, device = "cuda", dtype = torch.bfloat16) for _ in range(3))
    ref = F.scaled_dot_product_attention(
        q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
    ).transpose(1, 2)

    def sdpa():
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    def ait():
        return flash_attn_func(q, k, v, dropout_p = 0.0, softmax_scale = None, causal = False)

    for name, fn in (("sdpa", sdpa), ("aiter", ait)):
        try:
            t0 = time.time()
            with torch.no_grad():
                out = fn()
                if isinstance(out, (tuple, list)):
                    out = out[0]
            torch.cuda.synchronize()
            first = time.time() - t0
            err = (out.float() - ref).abs()
            times = []
            with torch.no_grad():
                for _ in range(args.iters):
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t1)
            times.sort()
            rec[name] = {"first_call_s": round(first, 3), "median_ms": round(1000 * times[len(times) // 2], 3),
                         "min_ms": round(1000 * times[0], 3), "max_abs_vs_fp32": float(err.max()),
                         "mean_abs_vs_fp32": float(err.mean()), "finite": bool(torch.isfinite(out).all())}
        except Exception as exc:  # noqa: BLE001
            rec[name] = {"error": repr(exc), "tb": traceback.format_exc()[-4000:]}
    _write(args.out, rec)


def _has(md, name):
    try:
        md.version(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def _write(path, rec):
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(rec, f, indent = 1, default = str)


if __name__ == "__main__":
    main()
