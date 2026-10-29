#!/usr/bin/env python3
"""Probe: SDPA time on this GPU with the strided k/v diffusers' native backend passes vs contiguous copies.

Observes only (judged by criteria/sdpa_contiguous_kv.py). Per (heads, tokens) cell, bf16, head_dim 128, batch 1:

  strided     q/k/v made [B, S, H, D] and permuted to [B, H, S, D] (what diffusers' ``_native_attention`` hands SDPA)
  strided_aa  the same call again, interleaved: the A/A noise arm
  kv_contig   k.contiguous() and v.contiguous() INSIDE the timed region, then SDPA (ComfyUI #16726)
  studio      the checkout's own path: Studio ``apply_attention_backend(..., target=<this GPU>)`` configured, then
              diffusers ``dispatch_attention_fn`` on the [B, S, H, D] tensors (every diffusers processor calls this)
  wan_attn    a diffusers ``WanAttention`` self-attention module (Wan 14B width at 40 heads) through its stock
              processor, same configuration; projections included, so only its base-vs-head delta matters

Arms are interleaved with a rotating order. It also records: which kernels SDPA launched (torch.profiler, one call per
arm), whether SDPA saw contiguous k/v on the studio arm (a wrapper on F.scaled_dot_product_attention), bitwise
equality / max-abs between the arms' outputs, and a sha256 of the wan_attn output for a cross-state comparison.
Results are written after every cell, so a timeout keeps what finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

# Studio sets this before torch is imported (studio/backend/main.py, core/inference/__init__.py).
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


def _write(path: Path, obs: dict) -> None:
    path.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")


def _kernel_names(torch, fn) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities = [ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names: dict[str, float] = {}
    for ev in prof.key_averages():
        dt = getattr(ev, "device_time_total", None)
        if dt is None:
            dt = getattr(ev, "cuda_time_total", 0)
        if dt and dt > 0:
            names[ev.key] = names.get(ev.key, 0.0) + float(dt)
    return [k[:120] for k, _ in sorted(names.items(), key = lambda kv: -kv[1])[:4]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--heads", default = "24,40")
    ap.add_argument("--tokens", default = "4096,8192,16384,32768")
    ap.add_argument("--min-reps", type = int, default = 5)
    ap.add_argument("--max-reps", type = int, default = 15)
    ap.add_argument("--cell-budget-s", type = float, default = 120.0)
    ap.add_argument("--state-budget-s", type = float, default = 1800.0)
    args = ap.parse_args()

    t_start = time.time()
    obs: dict = {"state": args.state, "cells": [], "skipped": []}
    sys.path.insert(0, str(args.checkout / "studio" / "backend"))
    try:
        import torch
        import torch.nn.functional as F

        props = torch.cuda.get_device_properties(0)
        obs["arch"] = (getattr(props, "gcnArchName", "") or "").split(":")[0]
        obs["device_name"] = torch.cuda.get_device_name(0)
        obs["l2_cache_size"] = int(getattr(props, "L2_cache_size", 0) or 0)
        obs["is_integrated"] = getattr(props, "is_integrated", None)
        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        obs["aotriton_experimental"] = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
        try:
            import diffusers

            obs["diffusers"] = diffusers.__version__
        except Exception as e:  # noqa: BLE001
            obs["diffusers_error"] = repr(e)
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"
        _write(args.out, obs)
        return 0

    dev = torch.device("cuda:0")
    dtype = torch.bfloat16

    # ---- configure the checkout's Studio attention exactly as a diffusion load does (native on ROCm).
    dispatch = None
    try:
        from types import SimpleNamespace

        from core.inference import diffusion_attention as da
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            _AttentionBackendRegistry,
            dispatch_attention_fn,
        )

        target = SimpleNamespace(device = "cuda", torch_device = "cuda:0", ordinal = 0, dtype = dtype)
        engaged = da.apply_attention_backend(SimpleNamespace(), None, target = target)
        native = _AttentionBackendRegistry._backends[AttentionBackendName.NATIVE]
        obs["studio_engaged_backend"] = engaged
        obs["studio_native_wrapped"] = bool(getattr(native, "_unsloth_rocm_contiguous_kv", False))
        obs["studio_has_contiguous_kv"] = hasattr(da, "install_rocm_contiguous_kv")
        obs["sdp_flags"] = {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        }
        dispatch = dispatch_attention_fn
    except Exception as e:  # noqa: BLE001
        obs["studio_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"

    # Record what layout SDPA itself receives on the studio arm.
    seen: list = []
    real_sdpa = F.scaled_dot_product_attention

    def spy(*a, **kw):
        q, k, v = (a[i] if len(a) > i else kw[n] for i, n in enumerate(("query", "key", "value")))
        seen.append((q.is_contiguous(), k.is_contiguous(), v.is_contiguous()))
        return real_sdpa(*a, **kw)

    def sdpa_backend_hint(q, k, v) -> dict:
        out = {}
        try:
            try:
                params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
            except TypeError:
                params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False)
            out["can_flash"] = bool(torch.backends.cuda.can_use_flash_attention(params, False))
            out["can_efficient"] = bool(torch.backends.cuda.can_use_efficient_attention(params, False))
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"[:200]
        return out

    heads_list = [int(x) for x in args.heads.split(",") if x]
    tokens_list = [int(x) for x in args.tokens.split(",") if x]
    D = 128

    for S in tokens_list:
        for H in heads_list:
            if time.time() - t_start > args.state_budget_s:
                obs["skipped"].append({"heads": H, "tokens": S, "why": "state budget"})
                continue
            cell: dict = {"heads": H, "tokens": S, "head_dim": D,
                          "kv_strided_bytes_per_head": 2 * S * D * 2}
            try:
                g = torch.Generator(device = dev).manual_seed(1234 + S + H)
                q, k, v = (torch.randn(1, S, H, D, device = dev, dtype = dtype, generator = g) for _ in range(3))
                qt, kt, vt = (t.permute(0, 2, 1, 3) for t in (q, k, v))
                cell["sdpa_sees_contiguous_strided_arm"] = [qt.is_contiguous(), kt.is_contiguous(),
                                                            vt.is_contiguous()]
                cell["backend_hint_strided"] = sdpa_backend_hint(qt, kt, vt)
                cell["backend_hint_contig_kv"] = sdpa_backend_hint(qt, kt.contiguous(), vt.contiguous())

                arms = {
                    "strided": lambda: F.scaled_dot_product_attention(qt, kt, vt),
                    "strided_aa": lambda: F.scaled_dot_product_attention(qt, kt, vt),
                    "kv_contig": lambda: F.scaled_dot_product_attention(qt, kt.contiguous(), vt.contiguous()),
                }
                if dispatch is not None:
                    arms["studio"] = lambda: dispatch(q, k, v)

                wan = None
                if dispatch is not None and H == 40:
                    try:
                        from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

                        torch.manual_seed(7)
                        wan = WanAttention(dim = H * D, heads = H, dim_head = D, eps = 1e-6,
                                           processor = WanAttnProcessor()).to(dev, dtype).eval()
                        x = torch.randn(1, S, H * D, device = dev, dtype = dtype,
                                        generator = torch.Generator(device = dev).manual_seed(99 + S))
                        arms["wan_attn"] = lambda: wan(x)
                    except Exception as e:  # noqa: BLE001
                        cell["wan_error"] = f"{type(e).__name__}: {e}"[:400]

                outs = {}
                with torch.no_grad():
                    # warmup + outputs
                    for name, fn in arms.items():
                        if name == "studio":
                            seen.clear()
                            F.scaled_dot_product_attention = spy
                            try:
                                outs[name] = fn()
                            finally:
                                F.scaled_dot_product_attention = real_sdpa
                            cell["studio_sdpa_saw_contiguous"] = [list(s) for s in seen]
                        elif name == "wan_attn":
                            seen.clear()
                            F.scaled_dot_product_attention = spy
                            try:
                                outs[name] = fn()
                            except Exception as e:  # noqa: BLE001
                                cell["wan_error"] = f"{type(e).__name__}: {e}"[:400]
                            finally:
                                F.scaled_dot_product_attention = real_sdpa
                            cell["wan_sdpa_saw_contiguous"] = [list(s) for s in seen]
                        else:
                            outs[name] = fn()
                    torch.cuda.synchronize()
                    if "wan_attn" in arms and "wan_attn" not in outs:
                        del arms["wan_attn"]

                    ref = outs["strided"]
                    num = {}
                    for name in ("strided_aa", "kv_contig"):
                        num[name] = {"bitwise_equal": bool(torch.equal(outs[name], ref)),
                                     "max_abs": float((outs[name].float() - ref.float()).abs().max())}
                    if "studio" in outs:
                        st = outs["studio"].permute(0, 2, 1, 3)
                        num["studio"] = {"bitwise_equal": bool(torch.equal(st, ref)),
                                         "max_abs": float((st.float() - ref.float()).abs().max())}
                    cell["numerics_vs_strided"] = num
                    cell["ref_max_abs_value"] = float(ref.float().abs().max())
                    if "wan_attn" in outs:
                        w = outs["wan_attn"]
                        w = w[0] if isinstance(w, tuple) else w
                        cell["wan_out_sha256"] = hashlib.sha256(
                            w.detach().float().cpu().numpy().tobytes()).hexdigest()
                    del outs, ref

                    # kernels: one profiled call per arm
                    kern = {}
                    for name, fn in arms.items():
                        try:
                            kern[name] = _kernel_names(torch, fn)
                        except Exception as e:  # noqa: BLE001
                            kern[name] = [f"profiler error {type(e).__name__}: {e}"[:200]]
                    cell["kernels"] = kern

                    # timing, interleaved with rotating order
                    times = {name: [] for name in arms}
                    names = list(arms)
                    cell_t0 = time.time()
                    rep = 0
                    while True:
                        order = names[rep % len(names):] + names[:rep % len(names)]
                        for name in order:
                            s, e = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                            torch.cuda.synchronize()
                            s.record()
                            r = arms[name]()
                            e.record()
                            torch.cuda.synchronize()
                            times[name].append(s.elapsed_time(e))
                            del r
                        rep += 1
                        spent = time.time() - cell_t0
                        if rep >= args.max_reps:
                            break
                        if rep >= args.min_reps and spent > args.cell_budget_s:
                            break
                cell["reps"] = rep
                cell["times_ms"] = {n: [round(t, 4) for t in ts] for n, ts in times.items()}
                cell["median_ms"] = {n: round(statistics.median(ts), 4) for n, ts in times.items()}
                cell["spread_rel"] = {n: round((max(ts) - min(ts)) / statistics.median(ts), 4)
                                      for n, ts in times.items()}
            except Exception as e:  # noqa: BLE001
                cell["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
            torch.cuda.empty_cache()
            obs["cells"].append(cell)
            obs["elapsed_s"] = round(time.time() - t_start, 1)
            _write(args.out, obs)
            print(f"cell H={H} S={S}: {cell.get('median_ms')} {cell.get('error', '')}", flush = True)

    obs["elapsed_s"] = round(time.time() - t_start, 1)
    _write(args.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
