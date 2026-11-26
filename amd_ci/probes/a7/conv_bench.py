#!/usr/bin/env python3
"""A7 item 5: ROCm temporal-unfolded batched Conv2d (sglang rocm.py ``_conv3d_as_batched_conv2d``) vs stock Conv3d.

  conv_bench.py --model wan|ltx --out F [--rounds 3]

Loads the real VAE (Wan 2.2 TI2V-5B vae/ in fp32 as Studio's vae_force_fp32; LTX-2.5 vae/ in bf16), NCHW only.
1. capture: one stock decode of a seeded latent records every Conv3d call shape (input, weight, stride, padding).
2. microbench: each unique kt > 1 shape, stock F.conv3d vs the unfolded Conv2d, interleaved, median ms, plus
   peak-allocated delta and the max rel error vs the stock output.
3. end to end: decode stock vs lowered (nn.Conv3d._conv_forward patched for kt > 1, groups 1, dilation 1, temporal
   stride 1), interleaved over rounds: median s, peak allocated GiB, PSNR of the decoded clip vs stock (not bit-exact).
Lowering: x (time padded) -> unfold(2, kt, 1) -> [N*T_out, kt*C, H, W] copy; weight [Co, C, kt, kh, kw] ->
[Co, kt*C, kh, kw]; one conv2d (spatial stride / padding); output back to [N, Co, T_out, Ho, Wo]. T_out chunked so the
unfolded copy stays under 1 GiB."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a7common import env_info, write  # noqa: E402

REPOS = {"wan": ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "AutoencoderKLWan", "float32", (1, 48, 13, 30, 52)),
         "ltx": ("Lightricks/LTX-2.5-Diffusers", "AutoencoderKLLTX2Video", "bfloat16", (1, 128, 7, 16, 24))}
UNF_BYTES = 1 << 30


def lowered_conv3d(F, torch, x, w, b, stride, padding, cache):
    kt = w.shape[2]
    pt, ph, pw = padding
    if pt:
        x = F.pad(x, (0, 0, 0, 0, pt, pt))
    N, C, T, H, W = x.shape
    To = T - kt + 1
    key = (w.data_ptr(), tuple(w.shape))
    w2 = cache.get(key)
    if w2 is None:
        w2 = w.permute(0, 2, 1, 3, 4).reshape(w.shape[0], kt * C, w.shape[3], w.shape[4]).contiguous()
        cache[key] = w2
    per_t = N * kt * C * H * W * x.element_size()
    step = max(1, min(To, UNF_BYTES // max(per_t, 1)))
    outs = []
    for t0 in range(0, To, step):
        t1 = min(To, t0 + step)
        xs = x[:, :, t0:t1 + kt - 1]
        xu = xs.unfold(2, kt, 1).permute(0, 2, 5, 1, 3, 4).reshape(N * (t1 - t0), kt * C, H, W)
        y = F.conv2d(xu, w2, b, stride[1:], (ph, pw))
        outs.append(y.view(N, t1 - t0, *y.shape[1:]))
    y = outs[0] if len(outs) == 1 else torch.cat(outs, 1)
    return y.permute(0, 2, 1, 3, 4).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True, choices = tuple(REPOS))
    ap.add_argument("--out", required = True)
    ap.add_argument("--rounds", type = int, default = 3)
    a = ap.parse_args()
    obs: dict = {"model": a.model, "shapes": [], "micro": [], "e2e": {}}
    t0 = time.time()
    try:
        import torch
        import torch.nn.functional as F
        import diffusers
        obs["env"] = env_info()
        obs["diffusers"] = diffusers.__version__
        repo, cls_name, dt_name, zshape = REPOS[a.model]
        dt = getattr(torch, dt_name)
        cls = getattr(diffusers, cls_name)
        vae = cls.from_pretrained(repo, subfolder = "vae", torch_dtype = dt).to("cuda").eval()
        obs["vae"] = {"class": cls_name, "dtype": dt_name, "latent": list(zshape)}
        g = torch.Generator(device = "cuda").manual_seed(2026)
        z = torch.randn(*zshape, generator = g, device = "cuda", dtype = torch.float32).to(dt)
        real = torch.nn.Conv3d._conv_forward
        state = {"mode": "stock", "calls": 0, "lowered": 0, "seen": {}}
        wcache: dict = {}

        def patched(self, x, weight, bias):
            state["calls"] += 1
            pad = self.padding if isinstance(self.padding, tuple) else None
            kt = weight.shape[2]
            if state["mode"] == "capture":
                key = (tuple(x.shape), tuple(weight.shape), tuple(self.stride), str(self.padding), tuple(self.dilation),
                       self.groups, self.padding_mode, str(x.dtype))
                state["seen"][key] = state["seen"].get(key, 0) + 1
            ok = (state["mode"] == "lowered" and kt > 1 and self.groups == 1 and tuple(self.dilation) == (1, 1, 1)
                  and self.stride[0] == 1 and self.padding_mode == "zeros" and pad is not None)
            if ok:
                state["lowered"] += 1
                return lowered_conv3d(F, torch, x, weight, bias, tuple(self.stride), pad, wcache)
            return real(self, x, weight, bias)
        torch.nn.Conv3d._conv_forward = patched

        def decode():
            with torch.no_grad():
                out = vae.decode(z, return_dict = False)[0]
            torch.cuda.synchronize()
            return out
        state["mode"] = "capture"
        ref = decode().float()
        obs["e2e"]["out_shape"] = list(ref.shape)
        obs["shapes"] = [{"x": list(k[0]), "w": list(k[1]), "stride": list(k[2]), "padding": k[3], "dil": list(k[4]),
                          "groups": k[5], "pmode": k[6], "dtype": k[7], "count": n} for k, n in state["seen"].items()]
        save = lambda: write(a.out, obs)  # noqa: E731
        save()
        # microbench per unique kt>1 shape
        for s in obs["shapes"]:
            if s["w"][2] <= 1 or s["groups"] != 1 or s["dil"] != [1, 1, 1] or s["stride"][0] != 1:
                continue
            row = {"x": s["x"], "w": s["w"], "count": s["count"]}
            try:
                pad = tuple(int(v) for v in s["padding"].strip("()").split(",") if v.strip())
                x = torch.randn(*s["x"], device = "cuda", dtype = dt)
                w = torch.randn(*s["w"], device = "cuda", dtype = dt) * (1.0 / math.sqrt(s["w"][1] * 27))
                b = torch.randn(s["w"][0], device = "cuda", dtype = dt)
                stride = tuple(s["stride"])
                fs = lambda: F.conv3d(x, w, b, stride, pad)  # noqa: E731
                fl = lambda: lowered_conv3d(F, torch, x, w, b, stride, pad, {})  # noqa: E731
                ys, yl = fs(), fl()
                row["max_rel"] = float((yl.float() - ys.float()).abs().max() / ys.float().abs().max().clamp_min(1e-12))
                res = {"stock": [], "lowered": []}
                for r in range(5):
                    for name, fn in ((("stock", fs), ("lowered", fl)) if r % 2 == 0 else (("lowered", fl), ("stock", fs))):
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                        e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                        e0.record()
                        fn()
                        e1.record()
                        e1.synchronize()
                        res[name].append(e0.elapsed_time(e1))
                        row[f"{name}_peak_extra_mib"] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
                row["stock_ms"], row["lowered_ms"] = st.median(res["stock"][1:]), st.median(res["lowered"][1:])
                row["speedup"] = row["stock_ms"] / row["lowered_ms"]
                del x, w, b, ys, yl
            except Exception as exc:  # noqa: BLE001
                row["error"] = repr(exc)[:300]
            obs["micro"].append(row)
            torch.cuda.empty_cache()
            save()
        # end to end
        state["mode"] = "lowered"
        decode()  # warm the lowered path (MIOpen conv2d solver search)
        times = {"stock": [], "lowered": []}
        peaks = {"stock": [], "lowered": []}
        psnr = []
        for r in range(a.rounds):
            for mode in (("stock", "lowered") if r % 2 == 0 else ("lowered", "stock")):
                state["mode"] = mode
                state["lowered"] = 0
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                out = decode()
                times[mode].append(time.perf_counter() - t)
                peaks[mode].append(round(torch.cuda.max_memory_allocated() / 2 ** 30, 3))
                if mode == "lowered":
                    obs["e2e"]["lowered_calls"] = state["lowered"]
                    o = out.float().clamp(-1, 1)
                    rr = ref.clamp(-1, 1)
                    mse = float(((o - rr) / 2).pow(2).mean())
                    psnr.append(99.0 if mse == 0 else 10 * math.log10(1.0 / mse))
                del out
        obs["e2e"].update({"times_s": times, "peak_gib": peaks, "psnr_db": psnr,
                           "stock_s": st.median(times["stock"]), "lowered_s": st.median(times["lowered"]),
                           "stock_peak_gib": max(peaks["stock"]), "lowered_peak_gib": max(peaks["lowered"])})
        obs["e2e"]["speedup"] = obs["e2e"]["stock_s"] / obs["e2e"]["lowered_s"]
        torch.nn.Conv3d._conv_forward = real
    except Exception:  # noqa: BLE001
        obs["error"] = traceback.format_exc()[-2000:]
    obs["seconds"] = round(time.time() - t0, 1)
    write(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
