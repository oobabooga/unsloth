#!/usr/bin/env python3
"""One fresh-process arm: a torch.compile'd diffusers WanTransformerBlock at Wan 2.2 TI2V 5B dims (dim 3072, ffn 14336,
24 heads, random weights from a fixed seed, bf16), timed with CUDA events and profiled with torch.profiler.

  --layout contig        weights as loaded (main)
  --layout pitch         [N, K] view of an [N, K + P] buffer (the weight-pitch branch's re-store)
  --layout trans         transposed problem: weight held as a contiguous [K, N] buffer, Linear sees its .t() view
                         (strides (1, N)), so the GEMM runs the NN problem instead of NT
  --layout trans_pitch   as trans, buffer [K, N + P]
  --flags none|cdt|mag|maxat   Inductor flags set before compile (coordinate_descent_tuning; max_autotune_gemm;
                         max_autotune + coordinate_descent_tuning)
Observes only; writes one JSON to --out. Drift is measured against an fp32 eager reference of the same block/input
(written by the first arm to --ref-dir, read by the rest).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
import traceback
from pathlib import Path

DIMS = {"wan5b": dict(dim = 3072, ffn = 14336, heads = 24, in_ch = 48, text = 512),
        "tiny": dict(dim = 1024, ffn = 2048, heads = 8, in_ch = 16, text = 64)}
# latent (frames, h, w) after the Wan 2.2 VAE (4x16x16); patch (1, 2, 2)
SHAPES = {"s480": (13, 30, 52), "s704": (13, 44, 80), "tiny": (2, 8, 8)}


def classify(name: str) -> str:
    n = name.lower()
    if any(s in n for s in ("attn_fwd", "fmha", "flash", "aotriton", "attention", "sdpa")):
        return "attention"
    if n.startswith("triton_tem") or "triton_mm" in n or n.startswith("cijk") or "gemm" in n or "hipblaslt" in n \
            or "wvsplitk" in n or n.startswith("ck_") or "matmul" in n or "kernel_mm" in n:
        return "gemm"
    if n.startswith("triton_"):
        return "inductor_glue"
    if any(s in n for s in ("copy", "memcpy", "memset", "transpose", "clone", "contiguous")):
        return "copy"
    if any(s in n for s in ("elementwise", "vectorized", "reduce", "cat", "fill", "index", "unrolled")):
        return "aten_glue"
    return "other"


def relayout(block, layout: str, p: int, torch) -> list:
    out = []
    with torch.no_grad():
        for name, m in block.named_modules():
            if type(m) is not torch.nn.Linear or m.weight.shape[1] < 1024:
                continue
            w = m.weight.data
            N, K = w.shape
            if layout == "pitch":
                buf = torch.empty((N, K + p), dtype = w.dtype, device = w.device)
                buf[:, :K].copy_(w)
                m.weight.data = buf[:, :K]
            elif layout == "trans":
                m.weight.data = w.t().contiguous().t()
            elif layout == "trans_pitch":
                buf = torch.empty((K, N + p), dtype = w.dtype, device = w.device)
                buf[:, :N].copy_(w.t())
                m.weight.data = buf[:, :N].t()
            out.append([name, list(m.weight.shape), list(m.weight.stride())])
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default = "contig", choices = ("contig", "pitch", "trans", "trans_pitch"))
    ap.add_argument("--pitch", type = int, default = 64)
    ap.add_argument("--flags", default = "none", choices = ("none", "cdt", "mag", "maxat"))
    ap.add_argument("--dims", default = "wan5b")
    ap.add_argument("--shapes", default = "s480,s704")
    ap.add_argument("--iters", type = int, default = 20)
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--no-compile", action = "store_true")
    ap.add_argument("--ref-dir", required = True)
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    rec: dict = {"args": vars(a), "verdict": "error"}
    outp = Path(a.out)

    def save():
        outp.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")
    try:
        import torch
        import torch._inductor.config as ic
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed, WanTransformerBlock
        import diffusers
        rec["env"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                      "diffusers": diffusers.__version__,
                      "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)}
        if a.flags in ("cdt", "maxat"):
            ic.coordinate_descent_tuning = True
        if a.flags == "mag":
            ic.max_autotune_gemm = True
        if a.flags == "maxat":
            ic.max_autotune = True
        rec["inductor_flags"] = {k: getattr(ic, k, None) for k in
                                 ("max_autotune", "max_autotune_gemm", "max_autotune_pointwise",
                                  "coordinate_descent_tuning", "benchmark_kernel")}
        d = DIMS[a.dims]
        torch.manual_seed(0)
        block = WanTransformerBlock(d["dim"], d["ffn"], d["heads"], "rms_norm_across_heads", True, 1e-6, None)
        with torch.no_grad():
            for prm in block.parameters():  # LayerNorm / RMSNorm weights default to ones; keep scale_shift realistic
                if prm.dim() == 2:
                    prm.normal_(0, 0.02)
            block.scale_shift_table.normal_(0, 0.02)
        block = block.eval().requires_grad_(False)
        rope = WanRotaryPosEmbed(d["dim"] // d["heads"], (1, 2, 2), 1024)
        rec["shapes"] = {}
        dev = "cuda"
        ref_dir = Path(a.ref_dir)
        ref_dir.mkdir(parents = True, exist_ok = True)
        inputs = {}
        for sh in a.shapes.split(","):
            f, h, w = SHAPES[sh]
            g = torch.Generator().manual_seed(1)
            lat = torch.zeros(1, d["in_ch"], f, h * 2, w * 2)
            rot = rope(lat)
            S = f * h * w
            hs = torch.randn(1, S, d["dim"], generator = g)
            enc = torch.randn(1, d["text"], d["dim"], generator = g)
            temb = torch.randn(1, 6, d["dim"], generator = g) * 0.1
            inputs[sh] = (hs, enc, temb, rot)
            refp = ref_dir / f"ref32_{a.dims}_{sh}.pt"
            if not refp.is_file():  # fp32 eager reference, written once (first arm), same seeds in every arm
                b32 = block.to(dev, torch.float32)
                F = torch.nn.functional
                sdpa = F.scaled_dot_product_attention

                def chunked(*args, **kw):  # fp32 math SDPA over all queries at once does not fit
                    args = list(args)
                    q = args.pop(0) if args else kw.pop("query")
                    k = args.pop(0) if args else kw.pop("key")
                    v = args.pop(0) if args else kw.pop("value")
                    m = args.pop(0) if args else kw.pop("attn_mask", None)
                    outs = []
                    for i in range(0, q.shape[-2], 512):
                        mi = m[..., i:i + 512, :] if m is not None and m.dim() >= 2 and m.shape[-2] == q.shape[-2] else m
                        outs.append(sdpa(q[..., i:i + 512, :], k, v, mi, *args, **kw))
                    return torch.cat(outs, dim = -2)
                F.scaled_dot_product_attention = chunked
                with torch.no_grad():
                    r32 = b32(hs.to(dev), enc.to(dev), temb.to(dev),
                              tuple(t.to(dev) for t in rot) if isinstance(rot, tuple) else rot.to(dev))
                F.scaled_dot_product_attention = sdpa
                torch.save(r32.float().cpu(), refp)
                del r32
                rec["wrote_ref"] = True
        block = block.to(dev, torch.bfloat16)
        torch.cuda.empty_cache()
        rec["weights"] = relayout(block, a.layout, a.pitch, torch)
        rec["n_relayout"] = len(rec["weights"])
        model = block if a.no_compile else torch.compile(block)
        for sh, (hs, enc, temb, rot) in inputs.items():
            r: dict = {"S": hs.shape[1]}
            rec["shapes"][sh] = r
            xs = (hs.to(dev, torch.bfloat16), enc.to(dev, torch.bfloat16), temb.to(dev, torch.bfloat16),
                  tuple(t.to(dev) for t in rot) if isinstance(rot, tuple) else rot.to(dev))
            t0 = time.time()
            with torch.no_grad():
                y = model(*xs)
                torch.cuda.synchronize()
                r["first_call_s"] = round(time.time() - t0, 2)
                for _ in range(3):
                    model(*xs)
                torch.cuda.synchronize()
                reps = []
                for _ in range(a.reps):
                    ev = [(torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True))
                          for _ in range(a.iters)]
                    for s, e in ev:
                        s.record()
                        model(*xs)
                        e.record()
                    torch.cuda.synchronize()
                    reps.append(st.median(s.elapsed_time(e) for s, e in ev))
                r["ms_reps"] = [round(x, 4) for x in reps]
                r["ms"] = round(st.median(reps), 4)
                ref = torch.load(ref_dir / f"ref32_{a.dims}_{sh}.pt").to(dev)
                yf = y.float()
                r["finite"] = bool(torch.isfinite(yf).all())
                r["rel_err_vs_fp32"] = float((yf - ref).norm() / ref.norm())
                r["max_abs_vs_fp32"] = float((yf - ref).abs().max())
                del ref
                prof_iters = 5
                from torch.profiler import ProfilerActivity, profile
                with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(prof_iters):
                        model(*xs)
                    torch.cuda.synchronize()
            kern: dict = {}
            for e in prof.events():
                if getattr(e, "device_type", None) != torch.autograd.DeviceType.CUDA:
                    continue
                us = e.time_range.elapsed_us() if getattr(e, "time_range", None) is not None else 0
                k = kern.setdefault(e.name, [0, 0.0])
                k[0] += 1
                k[1] += us
            tot = sum(v[1] for v in kern.values()) or 1.0
            cat: dict = {}
            for nm, (c, us) in kern.items():
                cc = classify(nm)
                cat.setdefault(cc, [0, 0.0])
                cat[cc][0] += c
                cat[cc][1] += us
            r["prof_iters"] = prof_iters
            r["device_ms_per_iter"] = round(tot / 1000 / prof_iters, 4)
            r["share"] = {k: round(v[1] / tot, 4) for k, v in sorted(cat.items())}
            r["launches_per_iter"] = {k: round(v[0] / prof_iters, 2) for k, v in sorted(cat.items())}
            r["kernels"] = sorted(([nm[:160], round(c / prof_iters, 2), round(us / tot, 4), classify(nm)]
                                   for nm, (c, us) in kern.items()), key = lambda x: -x[2])[:60]
            r["n_kernel_names"] = len(kern)
            r["glue_share"] = round(sum(v[1] for k, v in cat.items() if k in ("inductor_glue", "aten_glue", "copy"))
                                    / tot, 4)
            del xs, y
            torch.cuda.empty_cache()
            save()
        rec["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        rec["verdict"] = "ok"
    except Exception:  # noqa: BLE001
        rec["error"] = traceback.format_exc()[-3000:]
    save()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
