#!/usr/bin/env python3
"""carbon6#3: ROCm/aiter Triton conv2d (vendored here from ROCm/aiter@6aa39366b7, MIT, conv files only; the #8504
revert touches only HIP JIT files) vs MIOpen (F.conv2d) on gfx1151 at every decoder Conv2d shape of the Qwen-Image-2.1
(AutoencoderKLQwenImage21), FLUX.2 / FLUX.3 (AutoencoderKLFlux2) and FLUX.1 / Z-Image (AutoencoderKL) VAEs at
1024^2, bf16, including the small mid-block planes (66^2 / 128^2) where comfy-kitchen#324 lost 0.34-0.84x.

Studio process env: core.inference imported first (TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL), MIOPEN_SEARCH_CUTOFF as
Studio's hardware.py sets it on RDNA, and Studio's MIOpen user-db installer called (records whether a db matched).

Per shape, arms: M (F.conv2d NCHW), M_aa (same, A/A), T (aiter conv2d NCHW router), and channels-last M_cl / T_nhwc
(input converted outside the timing). CUDA events, 3 warm-ups then 5 rotated rounds of ITERS calls. Error: rel L2 vs
fp32 F.conv2d of the same bf16 operands. Engagement: torch.profiler kernel names (full) per arm per shape.
Then one full random-weight decode per VAE (bf16, from the shipped config), every nn.Conv2d routed M vs T in
M T M T M T order, routed-call counter, PSNR of T's image vs M's.

  conv_bench.py --root <tree>/studio/backend --out R.json [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as stt
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITERS = 5
ROUNDS = 5
LATENTS = {"qwen21": (1, 64, 1, 64, 64), "flux2": (1, 32, 128, 128), "flux1_zimage": (1, 16, 128, 128)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--quick", action = "store_true")
    ap.add_argument("--no-decode", action = "store_true")
    a = ap.parse_args()
    rec: dict = {"verdict": "running", "rows": [], "decode": {}}
    outp = Path(a.out)

    def save():
        outp.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")
    sys.path.insert(0, str(Path(a.root).resolve()))
    sys.path.insert(0, str(HERE / "aiter_pkg"))
    try:
        os.environ.setdefault("MIOPEN_SEARCH_CUTOFF", "1")
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import core.inference  # noqa: F401  Studio's process env
        import aiter
        from aiter.ops.triton.conv.conv2d import conv2d as aconv
        import triton
        rec["aiter_file"] = aiter.__file__
        import re
        jit = set()
        for f in (HERE / "aiter_pkg" / "aiter" / "ops" / "triton" / "_triton_kernels" / "conv").glob("*.py"):
            jit |= set(re.findall(r"@triton\.jit[^\n]*\n(?:@[^\n]*\n)*def\s+(\w+)", f.read_text(encoding = "utf-8")))
        rec["aiter_jit_names"] = sorted(jit)
        rec["torch"], rec["triton"], rec["hip"] = torch.__version__, triton.__version__, torch.version.hip
        rec["arch"] = torch.cuda.get_device_properties(0).gcnArchName
        try:
            from core.inference import diffusion_rocm_miopen_db as MDB

            class _P:  # stand-in pipe for the installer's ownership bookkeeping
                pass
            rec["_pipe_keep"] = None
            pipe = _P()
            rec["miopen_db_eligible"] = str(MDB.eligible(torch.device("cuda", 0)))[:300]
            MDB.install_for_pipe(pipe, torch.device("cuda", 0), None)
            rec["miopen_db_active"] = MDB.is_active()
        except Exception as exc:  # noqa: BLE001
            rec["miopen_db_error"] = repr(exc)[:300]
        rec["env"] = {k: os.environ.get(k) for k in ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "MIOPEN_SEARCH_CUTOFF",
                                                     "MIOPEN_USER_DB_PATH", "MIOPEN_FIND_MODE")}
        save()
        dev = torch.device("cuda", 0)
        shapes = json.loads((HERE / "vae_convs.json").read_text(encoding = "utf-8"))
        uniq: dict = {}
        for vae, lst in shapes.items():
            for c in lst:
                key = (tuple(c["x"]), tuple(c["w"]), tuple(c["stride"]), tuple(c["padding"]), c["bias"])
                uniq.setdefault(key, {"uses": {}})["uses"][vae] = c["count"]
        keys = list(uniq)
        if a.quick:
            keys = keys[:2]
        gen = torch.Generator(device = "cpu").manual_seed(7)
        from torch.profiler import ProfilerActivity, profile

        def bench(fns: dict) -> dict:
            names = list(fns)
            for _ in range(3):
                for n in names:
                    fns[n]()
            torch.cuda.synchronize()
            t: dict = {n: [] for n in names}
            for r in range(ROUNDS):
                for n in names[r % len(names):] + names[:r % len(names)]:
                    e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                    e0.record()
                    for _ in range(ITERS):
                        fns[n]()
                    e1.record()
                    e1.synchronize()
                    t[n].append(e0.elapsed_time(e1) / ITERS)
            return t

        def kernels(fn) -> list:
            with profile(activities = [ProfilerActivity.CUDA]) as pr:
                fn()
                torch.cuda.synchronize()
            ev = [(e.key, float(getattr(e, "self_device_time_total", 0) or 0)) for e in pr.key_averages()]
            ev = [x for x in ev if x[1] > 0]
            ev.sort(key = lambda x: -x[1])
            return [[k, round(us, 1)] for k, us in ev[:6]]

        for key in keys:
            xs, wsh, st_, pd, has_b = key
            row = {"x": list(xs), "w": list(wsh), "stride": list(st_), "padding": list(pd), "bias": has_b,
                   "uses": uniq[key]["uses"], "plane_px": xs[2] * xs[3]}
            try:
                x = torch.randn(xs, generator = gen).to(torch.bfloat16).to(dev)
                fan = wsh[1] * wsh[2] * wsh[3]
                w = (torch.randn(wsh, generator = gen) / math.sqrt(fan)).to(torch.bfloat16).to(dev)
                b = (torch.randn(wsh[0], generator = gen) * 0.1).to(torch.bfloat16).to(dev) if has_b else None
                x_cl, w_cl = x.contiguous(memory_format = torch.channels_last), w.contiguous(memory_format = torch.channels_last)
                kw = dict(stride = st_, padding = pd)
                fns = {"M": lambda: F.conv2d(x, w, b, **kw), "M_aa": lambda: F.conv2d(x, w, b, **kw),
                       "T": lambda: aconv(x, w, b, st_, pd, (1, 1), layout = "nchw"),
                       "M_cl": lambda: F.conv2d(x_cl, w_cl, b, **kw),
                       "T_nhwc": lambda: aconv(x_cl, w, b, st_, pd, (1, 1), layout = "nhwc")}
                with torch.no_grad():
                    ref = F.conv2d(x.float(), w.float(), b.float() if b is not None else None, **kw)
                    rn = float(ref.norm())
                    errs, outs_ok = {}, {}
                    for n, fn in fns.items():
                        if n == "M_aa":
                            continue
                        try:
                            y = fn()
                            torch.cuda.synchronize()
                            errs[n] = float((y.float() - ref).norm()) / max(rn, 1e-30)
                            outs_ok[n] = True
                            del y
                        except Exception as exc:  # noqa: BLE001
                            row.setdefault("arm_errors", {})[n] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    del ref
                    live = {n: f for n, f in fns.items() if n == "M_aa" or outs_ok.get(n)}
                    row["rel_l2"] = {n: round(v, 7) for n, v in errs.items()}
                    row["kernels"] = {n: kernels(f) for n, f in live.items() if n != "M_aa"}
                    t = bench(live)
                row["ms"] = {n: round(stt.median(v), 4) for n, v in t.items()}
                row["ms_all"] = {n: [round(v, 4) for v in vs] for n, vs in t.items()}
                del x, w, b, x_cl, w_cl
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            rec["rows"].append(row)
            print("CONV", json.dumps({k: row.get(k) for k in ("x", "w", "ms", "rel_l2")}), flush = True)
            save()
            torch.cuda.empty_cache()
        if not a.no_decode:
            import diffusers
            rec["diffusers"] = diffusers.__version__
            route = {"on": False, "routed": 0, "fallback": 0, "errors": []}
            orig = nn.Conv2d._conv_forward

            def fwd(self, input, weight, bias):
                if route["on"] and self.groups == 1 and self.padding_mode == "zeros" and input.dtype == torch.bfloat16 \
                        and not isinstance(self.padding, str):
                    try:
                        y = aconv(input, weight, bias, self.stride, self.padding, self.dilation, layout = "nchw")
                        route["routed"] += 1
                        return y
                    except Exception as exc:  # noqa: BLE001
                        route["errors"].append(f"{type(exc).__name__}: {str(exc)[:200]}")
                route["fallback"] += int(route["on"])
                return orig(self, input, weight, bias)
            nn.Conv2d._conv_forward = fwd
            classes = {"qwen21": diffusers.AutoencoderKLQwenImage21, "flux2": diffusers.AutoencoderKLFlux2,
                       "flux1_zimage": diffusers.AutoencoderKL}
            for vae, cls in classes.items():
                d: dict = {}
                rec["decode"][vae] = d
                try:
                    cfg = json.loads((HERE / "vae_cfg" / f"{vae}.json").read_text(encoding = "utf-8"))
                    torch.manual_seed(0)
                    m = cls.from_config(cfg).to(dev, torch.bfloat16).eval()
                    z = torch.randn(LATENTS[vae], generator = gen).to(torch.bfloat16).to(dev)
                    imgs, times = {}, {"M": [], "T": []}
                    with torch.no_grad():
                        for arm in ("M", "T", "M", "T", "M", "T", "M", "T"):  # the first M / T pair is the warm-up
                            route["on"] = arm == "T"
                            r0 = route["routed"]
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            y = m.decode(z, return_dict = False)[0]
                            torch.cuda.synchronize()
                            times[arm].append(time.perf_counter() - t0)
                            d.setdefault("routed_per_decode", {})[arm] = route["routed"] - r0
                            imgs[arm] = y.float().clamp(-1, 1)
                    route["on"] = False
                    d["s_all"] = {k: [round(x, 4) for x in v] for k, v in times.items()}
                    d["s"] = {k: round(stt.median(v[1:]), 4) for k, v in times.items()}
                    mse = float(((imgs["T"] - imgs["M"]) ** 2).mean())
                    d["psnr_T_vs_M_db"] = round(10 * math.log10(4.0 / max(mse, 1e-20)), 2)
                    d["speedup"] = round(d["s"]["M"] / d["s"]["T"], 4)
                    d["fallback_errors"] = route["errors"][:5]
                    del m, z, imgs, y
                except Exception as exc:  # noqa: BLE001
                    d["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
                    route["on"] = False
                save()
                torch.cuda.empty_cache()
            nn.Conv2d._conv_forward = orig
        rec["verdict"] = "ok"
    except BaseException as exc:  # noqa: BLE001
        rec["verdict"] = "error"
        rec["error"] = repr(exc)[:2000]
        rec["traceback"] = traceback.format_exc()[-4000:]
        print(rec["traceback"], flush = True)
    save()
    os._exit(0 if rec["verdict"] == "ok" else 3)


if __name__ == "__main__":
    sys.exit(main())
