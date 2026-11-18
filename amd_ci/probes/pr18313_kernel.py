#!/usr/bin/env python3
"""Kernel-level cell for unsloth #18313 (ROCm packed rows on torch's flash varlen). Observes only; JSON to --out.

Loads this checkout's unsloth/utils/hf_packed_attention.py either through the full package (--mode full) or
through stub `unsloth` / `unsloth.utils` packages (--mode stub: only flash_impl.py and hf_packed_attention.py run,
so a box without the training stack can still answer). Records:
  route      : _torch_varlen(cuda:0) is not None, first / second call seconds; window verdict (head only)
  selfcheck  : head only. verdict under autocast bf16 / fp16, injected raising / leaking varlen and a raising
               aten entry (all must turn the route off), 4 threads racing a cold cache
  wrapper    : full mode only. the real sdpa wrapper on packed rows (no window and window 512): fast / fallback
               counter deltas and the output vs the wrapped sdpa over the dense mask
  kernels    : fwd+bwd ms (median of 5, each arm timed twice = A/A) for dense-mask SDPA (what main runs) vs
               flash varlen (eager varlen_attn and the aten op used in graphs), and max relative error of out /
               dq / dk / dv against an fp32 per-document reference written here (not the PR's)
  efficient  : head and base: _efficient_varlen_takes() for a head dim 512 row (must stay False on ROCm)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import threading
import time
import traceback
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--mode", choices = ("full", "stub"), default = "full")
ap.add_argument("--kernels", type = int, default = 1)
a = ap.parse_args()
ck = Path(a.checkout).resolve()
res: dict = {"mode": a.mode, "stage": "start"}


def dump():
    a.out.write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")


def err(e):
    return f"{type(e).__name__}: {str(e)[:600]}"


try:
    import torch

    res["versions"] = {"torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
                       "python": sys.version.split()[0], "platform": sys.platform}
    try:
        props = torch.cuda.get_device_properties(0)
        res["versions"]["arch"] = getattr(props, "gcnArchName", None) or f"sm{props.major}{props.minor}"
        res["versions"]["device"] = torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        res["versions"]["arch"] = f"ERR {e}"
    if a.mode == "full":
        import unsloth  # noqa: F401
        from unsloth.utils import hf_packed_attention as hpa
        res["unsloth_file"] = unsloth.__file__
    else:
        for name, sub in (("unsloth", "unsloth"), ("unsloth.utils", "unsloth/utils")):
            m = types.ModuleType(name)
            m.__path__ = [str(ck / sub)]
            sys.modules[name] = m
        import unsloth.utils.hf_packed_attention as hpa
    res["hpa_file"] = hpa.__file__
    res["has_selfcheck"] = hasattr(hpa, "_hip_varlen_matches")
    res["stage"] = "imported"
    dump()
    dev = torch.device("cuda", 0)
    try:
        from torch.nn.attention.varlen import varlen_attn  # noqa: F401
        import inspect
        res["varlen_params"] = sorted(inspect.signature(varlen_attn).parameters)
    except Exception as e:  # noqa: BLE001
        res["varlen_params"] = err(e)

    # ---- route (both states) ----
    r = {}
    hpa._TORCH_VARLEN_DEVICES.clear()
    if hasattr(hpa, "_HIP_WINDOW_OK"):
        hpa._HIP_WINDOW_OK.clear()
    torch.cuda.synchronize()
    t = time.perf_counter()
    try:
        fn = hpa._torch_varlen(dev)
        r["on"] = fn is not None
    except Exception as e:  # noqa: BLE001
        r["on"], r["error"] = None, err(e)
    torch.cuda.synchronize()
    r["first_s"] = time.perf_counter() - t
    t = time.perf_counter()
    hpa._torch_varlen(dev)
    r["second_s"] = time.perf_counter() - t
    r["fn_loaded"] = hpa._TORCH_VARLEN[0] is not None if hpa._TORCH_VARLEN else None
    if hasattr(hpa, "_hip_window_ok"):
        t = time.perf_counter()
        r["window_512"] = hpa._hip_window_ok(dev, 512)
        torch.cuda.synchronize()
        r["window_first_s"] = time.perf_counter() - t
        t = time.perf_counter()
        r["window_1024"] = hpa._hip_window_ok(dev, 1024)
        r["window_second_s"] = time.perf_counter() - t
    res["route"] = r
    res["stage"] = "route"
    dump()

    # ---- efficient varlen stays off on ROCm (both states) ----
    try:
        q512 = torch.zeros(1, 4, 64, 512, device = dev, dtype = torch.bfloat16)
        res["efficient_takes_hd512"] = hpa._efficient_varlen_takes(q512, None)
    except Exception as e:  # noqa: BLE001
        res["efficient_takes_hd512"] = err(e)

    # ---- self-check robustness (head only) ----
    if res["has_selfcheck"]:
        sc = {}
        real_fn = hpa._TORCH_VARLEN[0]

        def fresh():
            hpa._TORCH_VARLEN_DEVICES.clear()
            hpa._HIP_WINDOW_OK.clear()

        for label, dtype in (("autocast_bf16", torch.bfloat16), ("autocast_fp16", torch.float16)):
            try:
                with torch.autocast("cuda", dtype = dtype):
                    sc[label] = hpa._hip_varlen_matches(dev, None)
            except Exception as e:  # noqa: BLE001
                sc[label] = err(e)
        try:
            with torch.no_grad():
                sc["no_grad"] = hpa._hip_varlen_matches(dev, None)
            with torch.inference_mode():
                sc["inference_mode"] = hpa._hip_varlen_matches(dev, None)
        except Exception as e:  # noqa: BLE001
            sc["grad_modes_error"] = err(e)

        def raiser(*args, **kwargs):
            raise RuntimeError("injected: no varlen kernel")

        def leaky(q, k, v, cu_q, cu_k, max_q, max_k, scale = None, window_size = (-1, 0), **kw):
            # whole-row causal attention: crosses documents
            qh, kh, vh = (x.transpose(0, 1)[None] for x in (q, k, v))
            if kh.shape[1] != qh.shape[1]:
                g = qh.shape[1] // kh.shape[1]
                kh, vh = kh.repeat_interleave(g, 1), vh.repeat_interleave(g, 1)
            o = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal = True, scale = scale)
            return o[0].transpose(0, 1)

        for label, fake in (("injected_raise", raiser), ("injected_leak", leaky)):
            fresh()
            hpa._TORCH_VARLEN[:] = [fake]
            try:
                sc[label + "_route_on"] = hpa._torch_varlen(dev) is not None
            except Exception as e:  # noqa: BLE001
                sc[label + "_route_on"] = err(e)
            finally:
                hpa._TORCH_VARLEN[:] = [real_fn]
        fresh()
        orig_flash = hpa._flash_varlen
        hpa._flash_varlen = raiser
        try:
            sc["injected_aten_raise_route_on"] = hpa._torch_varlen(dev) is not None
        except Exception as e:  # noqa: BLE001
            sc["injected_aten_raise_route_on"] = err(e)
        finally:
            hpa._flash_varlen = orig_flash
        fresh()
        sc["restored_route_on"] = hpa._torch_varlen(dev) is not None
        # threads racing a cold cache
        fresh()
        outs, errs = [], []

        def worker():
            try:
                outs.append(hpa._torch_varlen(dev) is not None)
            except Exception as e:  # noqa: BLE001
                errs.append(err(e))

        th = [threading.Thread(target = worker) for _ in range(4)]
        [x.start() for x in th]
        [x.join() for x in th]
        sc["threads"] = {"results": outs, "errors": errs, "cache": dict(hpa._TORCH_VARLEN_DEVICES)}
        # per-call cost once cached (eager path calls these every attention layer)
        n = 20000
        t = time.perf_counter()
        for _ in range(n):
            hpa._torch_varlen(dev)
        sc["cached_torch_varlen_us"] = (time.perf_counter() - t) / n * 1e6
        t = time.perf_counter()
        for _ in range(n):
            hpa._hip_window_ok(dev, 512)
        sc["cached_window_ok_us"] = (time.perf_counter() - t) / n * 1e6
        res["selfcheck"] = sc
        dump()

    # ---- independent fp32 per-document reference ----
    def reference(q, k, v, lengths, scale, window):
        """q (1,H,T,D) k/v (1,Hkv,T,D) fp32 -> (1,T,H,D)"""
        g = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
        outs, s = [], 0
        for n in lengths:
            sl = slice(s, s + n)
            i = torch.arange(n, device = q.device)
            allow = i[None, :] <= i[:, None]
            if window is not None:
                allow = allow & (i[None, :] > i[:, None] - window)
            w = (q[:, :, sl] @ k[:, :, sl].transpose(-1, -2)) * scale
            outs.append(w.masked_fill(~allow, float("-inf")).softmax(-1) @ v[:, :, sl])
            s += n
        return torch.cat(outs, 2).transpose(1, 2)

    def dense_mask(lengths, window, device):
        seg = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths)).to(device)
        pos = torch.arange(sum(lengths), device = device)
        m = (seg[:, None] == seg[None, :]) & (pos[None, :] <= pos[:, None])
        if window is not None:
            m &= pos[None, :] > pos[:, None] - window
        return m[None, None]

    def lengths_for(total, docs, seed):
        g = torch.Generator().manual_seed(seed)
        cuts = sorted(torch.randperm(total - 1, generator = g)[: docs - 1].add(1).tolist())
        b = [0, *cuts, total]
        return [b[i + 1] - b[i] for i in range(docs)]

    def make(total, hq, hkv, hd, seed, dtype = torch.bfloat16):
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(1, hq, total, hd, generator = g).to(dev, dtype).requires_grad_()
        k = torch.randn(1, hkv, total, hd, generator = g).to(dev, dtype).requires_grad_()
        v = torch.randn(1, hkv, total, hd, generator = g).to(dev, dtype).requires_grad_()
        go = torch.randn(1, total, hq, hd, generator = g).to(dev, dtype)
        return q, k, v, go

    def rel(got, want):
        return float((got.float() - want).abs().max() / want.abs().max().clamp_min(1e-6))

    def arm_dense(q, k, v, mask, scale):
        g = q.shape[1] // k.shape[1]
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), attn_mask = mask, scale = scale)
        return out.transpose(1, 2)

    def time_arm(run, q, k, v, go, reps = 5, warm = 2):
        ts = []
        for i in range(warm + reps):
            for x in (q, k, v):
                x.grad = None
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing = True)
            e = torch.cuda.Event(enable_timing = True)
            s.record()
            out = run()
            out.backward(go)
            e.record()
            torch.cuda.synchronize()
            if i >= warm:
                ts.append(s.elapsed_time(e))
        return statistics.median(ts), min(ts), max(ts)

    if a.kernels and res["route"].get("on") is not None:
        shapes = [
            ("qwen35_full_8192_22docs", 8192, 22, 16, 4, 256, None),
            ("qwen35_full_2048_7docs", 2048, 7, 16, 4, 256, None),
            ("hd256_window1024_8192", 8192, 4, 16, 4, 256, 1024),
            ("hd128_window512_4096", 4096, 3, 32, 8, 128, 512),
        ]
        fn = hpa._TORCH_VARLEN[0] if hpa._TORCH_VARLEN else None
        kern = {}
        for name, total, docs, hq, hkv, hd, window in shapes:
            row = {"total": total, "docs": docs, "hq": hq, "hkv": hkv, "hd": hd, "window": window}
            try:
                lengths = lengths_for(total, docs, total + docs)
                q, k, v, go = make(total, hq, hkv, hd, total)
                scale = hd ** -0.5
                cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype = torch.int32, device = dev)
                mx = max(lengths)
                refin = [x.detach().float().requires_grad_() for x in (q, k, v)]
                ref = reference(*refin, lengths, scale, window)
                wants = (ref, *torch.autograd.grad(ref, refin, go.float()))
                del ref
                mask = dense_mask(lengths, window, dev)
                arms = {"dense_sdpa": lambda: arm_dense(q, k, v, mask, scale)}
                if fn is not None:
                    arms["varlen_eager"] = lambda: hpa._call_torch_varlen(fn, q, k, v, cu, mx, scale, window)
                arms["aten_flash"] = lambda: hpa._flash_varlen(q, k, v, cu, mx, scale, window)
                for an, run in arms.items():
                    try:
                        out = run()
                        grads = torch.autograd.grad(out, (q, k, v), go)
                        row[an + "_rel_err"] = [rel(gt, w) for gt, w in zip((out, *grads), wants)]
                        row[an + "_finite"] = all(bool(torch.isfinite(x).all()) for x in (out, *grads))
                        del out, grads
                        row[an + "_ms"] = [time_arm(run, q, k, v, go), time_arm(run, q, k, v, go)]
                    except Exception as e:  # noqa: BLE001
                        row[an + "_error"] = err(e)
                del mask, wants, refin
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                row["error"] = err(e)
            kern[name] = row
            res["kernels"] = kern
            dump()

    # ---- the real wrapper on packed rows (full mode, both states) ----
    if a.mode == "full":
        wr = {}
        try:
            wr["enabled"] = hpa.enable_hf_packed_attention()

            class Attn(torch.nn.Module):
                def __init__(self, groups, window):
                    super().__init__()
                    self.is_causal = True
                    self.num_key_value_groups = groups
                    self.sliding_window = window
                    self.training = True

            for name, total, docs, hq, hkv, hd, window in (
                ("hd256_nowindow_2048", 2048, 7, 16, 4, 256, None),
                ("hd128_window512_4096", 4096, 3, 32, 8, 128, 512),
            ):
                row = {}
                try:
                    lengths = lengths_for(total, docs, 7 + total)
                    q, k, v, go = make(total, hq, hkv, hd, 11 + total)
                    mod = Attn(hq // hkv, window)
                    mask = dense_mask(lengths, window, dev)
                    seql = torch.tensor(lengths, dtype = torch.int32, device = dev)
                    before = dict(hpa.HF_PACKED_ATTENTION_STATS)
                    out, _ = hpa._sdpa_packed_varlen(mod, q, k, v, mask, packed_seq_lengths = seql, dropout = 0.0, scaling = None)
                    after = dict(hpa.HF_PACKED_ATTENTION_STATS)
                    row["delta"] = {kk: after[kk] - before.get(kk, 0) for kk in after if after[kk] != before.get(kk, 0)}
                    g1 = torch.autograd.grad(out, (q, k, v), go)
                    ref_out, _ = hpa._ORIG_SDPA[0](mod, q, k, v, mask, dropout = 0.0, scaling = None)
                    g2 = torch.autograd.grad(ref_out, (q, k, v), go)
                    refin = [x.detach().float().requires_grad_() for x in (q, k, v)]
                    ref = reference(*refin, lengths, hd ** -0.5, window)
                    wants = (ref, *torch.autograd.grad(ref, refin, go.float()))
                    row["wrapper_rel_err"] = [rel(gt, w) for gt, w in zip((out, *g1), wants)]
                    row["orig_sdpa_rel_err"] = [rel(gt, w) for gt, w in zip((ref_out, *g2), wants)]
                except Exception as e:  # noqa: BLE001
                    row["error"] = err(e)
                    row["traceback"] = traceback.format_exc()[-2000:]
                wr[name] = row
        except Exception as e:  # noqa: BLE001
            wr["error"] = err(e)
        res["wrapper"] = wr
    res["stage"] = "done"
except BaseException as e:  # noqa: BLE001
    res["error"] = err(e)
    res["traceback"] = traceback.format_exc()[-4000:]
dump()
