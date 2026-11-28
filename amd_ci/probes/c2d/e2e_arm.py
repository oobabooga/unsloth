#!/usr/bin/env python3
"""One FLUX.2 klein render arm in a fresh process through Studio's own DiffusionBackend load + generate (Studio
defaults; only width / height / steps / guidance from --gen). Records per render: wall s, VAE decode s (pipe.vae.decode
wrapped, synchronised), s/step, peak allocated GiB, md5, the rocm_triton_conv2d status block (routed / cold / ...),
the routed calls during that render's decode, and the manifest's ready-variant count. Saves render 0 and the last
render as PNG. After the renders: the last latent decoded by the pipe's VAE (this arm's conv path) and by a fresh fp32
copy of the same VAE (no router, MIOpen fp32): rel L2 / PSNR / max abs vs fp32.

  e2e_arm.py --root <tree>/studio/backend --model REPO --out DIR --tag T --gen '{"width":1024,"height":1024}' --renders 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

PROMPT = "a red fox sitting in fresh snow at golden hour, detailed fur, shallow depth of field"
SEED = 20261125


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True)
    ap.add_argument("--model", required = True)
    ap.add_argument("--tag", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--gen", default = "{}")
    ap.add_argument("--renders", type = int, default = 5)
    ap.add_argument("--fp32-check", action = "store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents = True, exist_ok = True)
    rec: dict = {"tag": a.tag, "verdict": "running", "root": a.root, "model": a.model, "renders": [],
                 "env": {k: os.environ.get(k) for k in ("UNSLOTH_DIFFUSION_ROCM_TRITON_CONV2D", "TRITON_CACHE_DIR",
                                                        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "MIOPEN_USER_DB_PATH")}}
    recp = out / f"{a.tag}.json"

    def save():
        recp.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")
    sys.path.insert(0, str(Path(a.root).resolve()))
    try:
        import numpy as np
        import torch

        rec["torch"], rec["hip"] = torch.__version__, getattr(torch.version, "hip", None)
        from core.inference.diffusion import DiffusionBackend
        try:
            from core.inference import diffusion_rocm_triton_conv2d as R2
            rec["router_file"] = R2.__file__
        except Exception as exc:  # noqa: BLE001 - main has no router
            R2 = None
            rec["router_import"] = repr(exc)[:200]
        eng = DiffusionBackend()
        t0 = time.perf_counter()
        eng.begin_load(a.model)
        while True:
            p = eng.load_progress() or {}
            if p.get("phase") == "ready":
                break
            if p.get("phase") == "error" or p.get("error"):
                raise RuntimeError(f"load failed: {p.get('error')}")
            if time.perf_counter() - t0 > 7200:
                raise TimeoutError(str(p))
            time.sleep(0.25)
        rec["load_s"] = round(time.perf_counter() - t0, 2)
        rec["t_load_end"] = time.time()
        st = eng.status() or {}
        rec["status_load"] = {k: st.get(k) for k in ("speed_mode", "speed_optims", "transformer_quant",
                                                       "rocm_triton_conv2d", "device", "dtype")}
        rec["status_keys"] = sorted(st)
        pipe = eng._state.pipe
        rec["pipeline_class"] = type(pipe).__name__
        rec["vae_class"] = type(pipe.vae).__name__
        rec["vae_dtype"] = str(next(pipe.vae.parameters()).dtype)
        wrapped = [n for n, m in pipe.vae.named_modules()
                   if getattr(m.__dict__.get("_conv_forward"), "_unsloth_triton_conv2d", False)]
        rec["wrapped_modules"] = len(wrapped)
        step_t: list = []
        sched_cls = type(getattr(pipe, "scheduler", None))
        if hasattr(sched_cls, "step"):
            s0 = sched_cls.step

            def step(self, *x, **k):
                r = s0(self, *x, **k)
                torch.cuda.synchronize()
                step_t.append(time.perf_counter())
                return r
            sched_cls.step = step
        vae_t: list = []
        vae_in: dict = {}
        vdec = pipe.vae.decode

        def decode(*x, **k):
            if x and hasattr(x[0], "shape"):
                vae_in["z"] = x[0].detach().clone()
                vae_in["k"] = dict(k)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            r = vdec(*x, **k)
            torch.cuda.synchronize()
            vae_t.append(time.perf_counter() - t1)
            return r
        pipe.vae.decode = decode
        gen = json.loads(a.gen)
        rec["params"] = gen
        save()
        for r in range(a.renders):
            step_t.clear()
            vae_t.clear()
            c0 = dict(R2.COUNTS) if R2 is not None else {}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            t_wall0 = time.time()
            res = eng.generate(prompt = PROMPT, seed = SEED, **gen)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            img = res["images"][0].convert("RGB")
            arr = np.asarray(img, dtype = np.uint8)
            row = {"r": r, "wall_s": round(wall, 4), "md5": hashlib.md5(arr.tobytes()).hexdigest(),
                   "n_steps": len(step_t),
                   "s_per_step": round((step_t[-1] - step_t[0]) / (len(step_t) - 1), 4) if len(step_t) >= 2 else None,
                   "vae_s": [round(x, 4) for x in vae_t],
                   "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                   "since_load_end_s": round(t_wall0 - rec["t_load_end"], 2)}
            if R2 is not None:
                row["conv2d_counts_delta"] = {k: R2.COUNTS[k] - c0.get(k, 0) for k in R2.COUNTS}
                row["ready_variants"] = len(R2._READY)
            st = eng.status() or {}
            row["status_conv2d"] = st.get("rocm_triton_conv2d")
            row["speed_optims"] = st.get("speed_optims")
            if r == 0 or r == a.renders - 1:
                img.save(out / f"{a.tag}_r{r}.png")
            rec["renders"].append(row)
            print("RENDER", json.dumps(row), flush = True)
            save()
        if a.fp32_check and vae_in.get("z") is not None:
            chk: dict = {}
            try:
                vae = pipe.vae
                ref_vae = type(vae).from_config(vae.config)
                ref_vae.load_state_dict({k: v.float() for k, v in vae.state_dict().items()})
                ref_vae = ref_vae.to(device = vae.device, dtype = torch.float32).eval()
                with torch.no_grad():
                    c0 = dict(R2.COUNTS) if R2 is not None else {}
                    y = vdec(vae_in["z"], **vae_in["k"])
                    routed = (R2.COUNTS["routed"] - c0.get("routed", 0)) if R2 is not None else 0
                    y = y[0] if isinstance(y, (tuple, list)) else getattr(y, "sample", y)
                    yr = ref_vae.decode(vae_in["z"].float(), **vae_in["k"])
                    yr = yr[0] if isinstance(yr, (tuple, list)) else getattr(yr, "sample", yr)
                d = (y.float() - yr.float())
                mse = float((d.clamp(-2, 2) ** 2).mean())
                chk = {"rel_l2": float(d.norm() / yr.float().norm()), "max_abs": float(d.abs().max()),
                       "psnr_db": round(10 * math.log10(4.0 / max(mse, 1e-20)), 3), "routed": routed,
                       "z_shape": list(vae_in["z"].shape), "z_dtype": str(vae_in["z"].dtype)}
                del ref_vae, y, yr, d
            except Exception as exc:  # noqa: BLE001
                chk["error"] = repr(exc)[:500]
            rec["fp32_check"] = chk
            save()
        rec["counts_end"] = dict(R2.COUNTS) if R2 is not None else None
        rec["verdict"] = "ok"
    except BaseException as exc:  # noqa: BLE001
        rec["verdict"] = "error"
        rec["error"] = repr(exc)[:2000]
        rec["traceback"] = traceback.format_exc()[-5000:]
        print(rec["traceback"], flush = True)
    save()
    print("ARM_DONE", a.tag, rec["verdict"], flush = True)
    os._exit(0 if rec["verdict"] == "ok" else 3)


if __name__ == "__main__":
    main()
