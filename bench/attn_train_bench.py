#!/usr/bin/env python3
"""Wan 2.2 TI2V-5B self-attention fwd+bwd on gfx1151: Studio's training path vs FeatherAttn (and AITER Triton).

Shapes: B=1, H=24, D=128, bf16, q/k/v leaves in [B, S, H, D] (what diffusers' Wan attention hands the native backend).
Arms (each timed including every layout conversion):
  studio      SDPA on the permuted [B, H, S, D] views, exactly diffusers' native backend (Studio training today)
  studio_A2   the same arm again (A/A noise)
  sdpa_contig SDPA on contiguous [B, H, S, D] copies (Studio's inference-side gfx1151 K/V packing idea, q too)
  feather     feather_attn.feather_attn_func on [B, S, H, D]
  aiter       aiter flash_attn_func (Triton) on [B, S, H, D] (the author's baseline)
Rounds interleave arms. Errors of out/dQ/dK/dV against an fp32 per-head reference. Writes JSON to --out only.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback

import torch
import torch.nn.functional as F

B, H, D = 1, 24, 128
DT = torch.bfloat16


def make_inputs(S, seed):
    g = torch.Generator(device = "cuda")
    g.manual_seed(seed)
    shp = (B, S, H, D)
    q = torch.randn(shp, device = "cuda", dtype = DT, generator = g)
    k = torch.randn(shp, device = "cuda", dtype = DT, generator = g)
    v = torch.randn(shp, device = "cuda", dtype = DT, generator = g)
    do = torch.randn(shp, device = "cuda", dtype = DT, generator = g)
    return q, k, v, do


def arm_fns(avail):
    def studio(q, k, v):
        return F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

    def sdpa_contig(q, k, v):
        c = lambda t: t.permute(0, 2, 1, 3).contiguous()
        return F.scaled_dot_product_attention(c(q), c(k), c(v)).permute(0, 2, 1, 3)

    fns = {"studio": studio, "studio_A2": studio, "sdpa_contig": sdpa_contig}
    if "feather" in avail:
        fns["feather"] = avail["feather"]
    if "aiter" in avail:
        fns["aiter"] = avail["aiter"]
    return fns


def load_optional(info):
    avail = {}
    try:
        t0 = time.time()
        import feather_attn
        import feather_attn.ops as fops

        counter = {"feather_bwd": 0, "aiter_bwd": 0}
        orig_fb, orig_ab = fops._feather_bwd, fops._aiter_bwd

        def counted_fb(*a, **kw):
            counter["feather_bwd"] += 1
            return orig_fb(*a, **kw)

        def counted_ab(*a, **kw):
            counter["aiter_bwd"] += 1
            return orig_ab(*a, **kw)

        fops._feather_bwd = counted_fb
        fops._aiter_bwd = counted_ab
        info["feather_counter"] = counter
        info["feather_file"] = feather_attn.__file__
        info["feather_import_s"] = round(time.time() - t0, 2)
        avail["feather"] = lambda q, k, v: feather_attn.feather_attn_func(q, k, v)
    except Exception as e:  # noqa: BLE001
        info["feather_error"] = f"{type(e).__name__}: {e}"[:2000]
        info["feather_tb"] = traceback.format_exc()[-3000:]
    try:
        from aiter.ops.triton.mha import flash_attn_func as aiter_fa

        avail["aiter"] = lambda q, k, v: aiter_fa(q, k, v)
    except Exception as e:  # noqa: BLE001
        info["aiter_error"] = f"{type(e).__name__}: {e}"[:1000]
    return avail


def fp32_reference(q, k, v, do, heads):
    """Exact attention + grads in fp32 for the listed heads (math, per head, autograd)."""
    outs, dqs, dks, dvs = [], [], [], []
    scale = 1.0 / math.sqrt(D)
    for h in heads:
        qh = q[0, :, h].float().clone().requires_grad_(True)
        kh = k[0, :, h].float().clone().requires_grad_(True)
        vh = v[0, :, h].float().clone().requires_grad_(True)
        p = torch.softmax((qh @ kh.T) * scale, dim = -1)
        o = p @ vh
        o.backward(do[0, :, h].float())
        outs.append(o.detach()); dqs.append(qh.grad); dks.append(kh.grad); dvs.append(vh.grad)
        del p, o
    st = lambda xs: torch.stack(xs, 0)
    return st(outs), st(dqs), st(dks), st(dvs)


def err(x, ref, heads):
    x = torch.stack([x[0, :, h].float() for h in heads], 0)
    d = (x - ref)
    return {
        "max_abs": float(d.abs().max()),
        "rel_fro": float(d.norm() / ref.norm().clamp_min(1e-30)),
        "finite": bool(torch.isfinite(x).all()),
    }


def one_iter(fn, q, k, v, do):
    q.grad = k.grad = v.grad = None
    e0, e1, e2 = (torch.cuda.Event(enable_timing = True) for _ in range(3))
    e0.record()
    out = fn(q, k, v)
    e1.record()
    out.backward(do)
    e2.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1), e1.elapsed_time(e2), out


def profile_kernels(fn, q, k, v, do):
    from torch.profiler import ProfilerActivity, profile

    q.grad = k.grad = v.grad = None
    with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = fn(q, k, v)
        out.backward(do)
        torch.cuda.synchronize()
    rows = []
    for ev in prof.key_averages():
        dev_t = getattr(ev, "device_time_total", None)
        if dev_t is None:
            dev_t = getattr(ev, "cuda_time_total", 0)
        if dev_t and dev_t > 0:
            rows.append((dev_t, ev.key))
    rows.sort(reverse = True)
    return [{"us": round(t, 1), "name": n[:160]} for t, n in rows[:10]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--seqs", default = "1024,4096,5070,27280")
    ap.add_argument("--arms", default = "studio,studio_A2,sdpa_contig,feather,aiter")
    ap.add_argument("--rounds", type = int, default = 5)
    ap.add_argument("--iters", type = int, default = 8)
    ap.add_argument("--long-iters", type = int, default = 2)
    ap.add_argument("--long-rounds", type = int, default = 3)
    ap.add_argument("--no-ref", action = "store_true")
    ap.add_argument("--tag", default = "")
    a = ap.parse_args()

    res = {"tag": a.tag, "env": {k: os.environ.get(k) for k in ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "HIP_VISIBLE_DEVICES")}}
    res["torch"] = torch.__version__
    res["hip"] = torch.version.hip
    res["device"] = torch.cuda.get_device_name(0)
    res["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "?")
    free, total = torch.cuda.mem_get_info()
    res["free_mib_start"], res["total_mib"] = free // 2**20, total // 2**20
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info = {}
    avail = load_optional(info)
    fns = {n: f for n, f in arm_fns(avail).items() if n in a.arms.split(",")}
    res["arms"] = list(fns)
    res["load"] = {k: v for k, v in info.items() if k != "feather_counter"}
    res["shapes"] = {}

    def dump():
        if "feather_counter" in info:
            res["feather_counter"] = dict(info["feather_counter"])
        with open(a.out, "w", encoding = "utf-8") as fh:
            json.dump(res, fh, indent = 1)

    for S in [int(s) for s in a.seqs.split(",")]:
        long = S > 8192
        iters = a.long_iters if long else a.iters
        rounds = a.long_rounds if long else a.rounds
        sr = {"S": S, "iters": iters, "rounds": rounds, "arms": {}}
        res["shapes"][str(S)] = sr
        q0, k0, v0, do = make_inputs(S, 1234 + S)
        q, k, v = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
        # correctness + kernel check per arm (fresh grads)
        heads = list(range(H)) if S <= 8192 else [0, 1]
        ref = None
        if not a.no_ref:
            try:
                t0 = time.time()
                ref = fp32_reference(q0, k0, v0, do, heads)
                sr["ref_s"] = round(time.time() - t0, 1)
                sr["ref_heads"] = len(heads)
            except Exception as e:  # noqa: BLE001
                sr["ref_error"] = f"{type(e).__name__}: {e}"[:500]
            torch.cuda.empty_cache()
        for name, fn in fns.items():
            ar = {"fwd_ms": [], "bwd_ms": [], "tot_ms": []}
            sr["arms"][name] = ar
            try:
                c0 = dict(info.get("feather_counter", {}))
                t0 = time.time()
                _, _, out = one_iter(fn, q, k, v, do)
                ar["first_call_s"] = round(time.time() - t0, 2)
                _, _, out = one_iter(fn, q, k, v, do)
                if ref is not None:
                    ar["err"] = {
                        "out": err(out.detach(), ref[0], heads),
                        "dq": err(q.grad, ref[1], heads),
                        "dk": err(k.grad, ref[2], heads),
                        "dv": err(v.grad, ref[3], heads),
                    }
                if name == "feather":
                    c1 = info["feather_counter"]
                    ar["engaged"] = {kk: c1[kk] - c0.get(kk, 0) for kk in c1}
                if not long:
                    ar["kernels"] = profile_kernels(fn, q, k, v, do)
                del out
            except Exception as e:  # noqa: BLE001
                ar["error"] = f"{type(e).__name__}: {e}"[:1500]
                ar["tb"] = traceback.format_exc()[-2500:]
            dump()
        live = [n for n in fns if "error" not in sr["arms"][n]]
        for r in range(rounds):
            order = live if r % 2 == 0 else live[::-1]
            for name in order:
                fn = fns[name]
                ar = sr["arms"][name]
                try:
                    one_iter(fn, q, k, v, do)  # warm
                    f_, b_ = [], []
                    for _ in range(iters):
                        tf, tb, out = one_iter(fn, q, k, v, do)
                        del out
                        f_.append(tf); b_.append(tb)
                    ar["fwd_ms"].append(statistics.median(f_))
                    ar["bwd_ms"].append(statistics.median(b_))
                    ar["tot_ms"].append(statistics.median([x + y for x, y in zip(f_, b_)]))
                except Exception as e:  # noqa: BLE001
                    ar["error"] = f"round {r}: {type(e).__name__}: {e}"[:1500]
            dump()
        for name in fns:
            ar = sr["arms"][name]
            for key in ("fwd_ms", "bwd_ms", "tot_ms"):
                xs = ar[key]
                if xs:
                    ar[key + "_med"] = round(statistics.median(xs), 3)
                    ar[key + "_spread_pct"] = round(100 * (max(xs) - min(xs)) / statistics.median(xs), 2)
        sr["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        del q, k, v, q0, k0, v0, do, ref
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dump()
    res["done"] = True
    dump()


if __name__ == "__main__":
    sys.exit(main())
