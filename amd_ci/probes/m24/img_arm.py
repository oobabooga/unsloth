#!/usr/bin/env python3
"""One image arm in a fresh process (AMD coverage Nov 24, also the NVIDIA reference arm).

Loads through Studio's DiffusionBackend (``--load`` kwargs, {} = Studio default), renders ``--renders`` times and
records per render: wall s, s/step (synchronised timestamps at every scheduler step, steps 2..N), VAE decode s,
peak allocated / reserved GiB, md5 of the final pixels, the PNG (lossless) for LPIPS. Initial noise is drawn on the
CPU from the seed (the pipeline's ``prepare_latents`` gets a CPU generator with the same seed), so the same seed gives
the same noise on gfx1151, NVIDIA and CPU; MOSS-Image already draws CPU noise itself (``comfy_noise``).
``--capture-vae`` saves every ``vae.decode`` input latent and output (fp32, CPU) for the VAE probe.
Warnings / fallbacks from Studio's loggers are kept (``log_lines``). Observes only."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

PROMPTS = [
    "a red fox sitting in fresh snow at golden hour, detailed fur, shallow depth of field",
    "a cluttered watchmaker's workbench under a desk lamp, brass gears, tweezers, a loupe, macro photo",
]
SEED = 20261124


class _Keep(logging.Handler):
    KEYS = ("fallback", "fall back", "falls back", "refus", "disabled", "not installed", "error", "failed", "skip",
            "warning", "unsupported", "offload", "route", "engine", "compile")

    def __init__(self):
        super().__init__(level = logging.INFO)
        self.lines: list = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        low = msg.lower()
        if record.levelno >= logging.WARNING or any(k in low for k in self.KEYS):
            if len(self.lines) < 400:
                self.lines.append(f"{record.levelname} {record.name}: {msg[:600]}")


def _cpu_gen(g, torch):
    if isinstance(g, torch.Generator) and g.device.type != "cpu":
        return torch.Generator("cpu").manual_seed(int(g.initial_seed()))
    if isinstance(g, (list, tuple)):
        return type(g)(_cpu_gen(x, torch) for x in g)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True, help = "studio/backend of the tree under test")
    ap.add_argument("--model", required = True)
    ap.add_argument("--tag", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--load", default = "{}")
    ap.add_argument("--gen", default = "{}")
    ap.add_argument("--renders", type = int, default = 2)
    ap.add_argument("--prompt-schedule", default = "", help = "comma list of prompt indexes per render (default all 0)")
    ap.add_argument("--save-renders", default = "all", help = "'all' or comma list of render indexes to save as PNG")
    ap.add_argument("--capture-vae", action = "store_true")
    ap.add_argument("--no-cpu-noise", action = "store_true")
    ap.add_argument("--noise-fp32", action = "store_true", help = "replace prepare_latents noise with an fp32 CPU draw")
    a = ap.parse_args()
    out = Path(a.out) / a.tag
    out.mkdir(parents = True, exist_ok = True)
    rec: dict = {"tag": a.tag, "verdict": "running", "renders": [], "model": a.model, "load": json.loads(a.load),
                 "seed": SEED, "prompts": PROMPTS,
                 "env": {k: v for k, v in os.environ.items()
                         if k.startswith(("UNSLOTH_", "PYTORCH_", "TORCHINDUCTOR_", "HIP_", "MIOPEN_", "HSA_"))}}
    recp = out / "record.json"
    keep = _Keep()
    logging.getLogger().addHandler(keep)
    rec["log_lines"] = keep.lines

    def save():
        recp.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")

    sys.path.insert(0, str(Path(a.root).resolve()))
    try:
        import numpy as np
        import torch

        rec["torch"] = torch.__version__
        rec["hip"] = getattr(torch.version, "hip", None)
        rec["cuda"] = getattr(torch.version, "cuda", None)
        rec["gpu"] = torch.cuda.get_device_name(0)
        rec["platform"] = sys.platform
        try:
            import diffusers, transformers
            rec["diffusers"], rec["transformers"] = diffusers.__version__, transformers.__version__
        except Exception:  # noqa: BLE001
            pass
        from core.inference.diffusion import DiffusionBackend

        eng = DiffusionBackend()
        t0 = time.perf_counter()
        eng.begin_load(a.model, **json.loads(a.load))
        deadline = time.time() + 7200
        while True:
            p = eng.load_progress() or {}
            if p.get("phase") == "ready":
                break
            if p.get("phase") == "error" or p.get("error"):
                raise RuntimeError(f"load failed: {p.get('error')}")
            if time.time() > deadline:
                raise TimeoutError(str(p))
            time.sleep(0.5)
        rec["load_s"] = round(time.perf_counter() - t0, 2)
        rec["load_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        st = eng.status() or {}
        rec["status_full"] = {k: (v if isinstance(v, (str, int, float, bool, type(None), list)) else str(v))
                              for k, v in st.items() if not isinstance(v, dict)}
        pipe = eng._state.pipe
        rec["pipeline_class"] = type(pipe).__name__
        rec["vae_class"] = type(getattr(pipe, "vae", None)).__name__
        try:
            rec["vae_dtype"] = str(pipe.vae.dtype)
        except Exception:  # noqa: BLE001
            pass
        try:
            tr = pipe.transformer
            rec["blocks_compiled"] = len([m for m in tr.modules() if getattr(m, "_compiled_call_impl", None) is not None])
        except Exception as exc:  # noqa: BLE001
            rec["blocks_compiled"] = repr(exc)[:200]

        # CPU noise: the pipeline's own prepare_latents, CPU generator with the same seed
        noise_hits: list = []
        noise_info: list = []
        rec["noise_info"] = noise_info
        rec["noise_hook"] = "none"
        if not a.no_cpu_noise and callable(getattr(pipe, "prepare_latents", None)):
            rec["noise_hook"] = "prepare_latents"
            pl0 = pipe.prepare_latents

            def prepare_latents(*args, **kw):
                args = tuple(_cpu_gen(x, torch) for x in args)
                kw = {k: _cpu_gen(v, torch) for k, v in kw.items()}
                noise_hits.append(1)
                out = pl0(*args, **kw)
                if a.noise_fp32:
                    # CPU bf16 randn is not stable across torch versions; fp32 CPU randn is. Replace the pipeline's
                    # noise (any layout of iid N(0,1) is valid noise) with an fp32 CPU draw from the seed, cast.
                    t_ = out[0] if isinstance(out, (tuple, list)) else out
                    gcpu = torch.Generator("cpu").manual_seed(SEED)
                    z_ = torch.randn(tuple(t_.shape), generator = gcpu, dtype = torch.float32).to(t_.device, t_.dtype)
                    out = (z_, *out[1:]) if isinstance(out, (tuple, list)) else z_
                try:
                    t0_ = out[0] if isinstance(out, (tuple, list)) else out
                    c = t0_.detach().float().cpu().contiguous()
                    noise_info.append({"shape": list(t0_.shape), "dtype": str(t0_.dtype),
                                       "md5": hashlib.md5(c.numpy().tobytes()).hexdigest(), "std": float(c.std())})
                except Exception as exc:  # noqa: BLE001
                    noise_info.append({"error": repr(exc)[:200]})
                return out
            pipe.prepare_latents = prepare_latents

        # step timestamps: class-level scheduler.step (survives a per-call scheduler copy of the same class)
        step_t: list = []
        sched_cls = type(getattr(pipe, "scheduler", None))
        rec["scheduler_class"] = sched_cls.__name__
        if hasattr(sched_cls, "step"):
            s0 = sched_cls.step

            def step(self, *args, **kw):
                r = s0(self, *args, **kw)
                torch.cuda.synchronize()
                step_t.append(time.perf_counter())
                return r
            sched_cls.step = step

        vae_t: list = []
        caps: list = []
        try:
            vae = pipe.vae
            vdec = vae.decode

            def decode(*x, **k):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                r = vdec(*x, **k)
                torch.cuda.synchronize()
                vae_t.append(time.perf_counter() - t1)
                if a.capture_vae and x and hasattr(x[0], "shape"):
                    try:
                        s = r.sample if hasattr(r, "sample") else (r[0] if isinstance(r, (tuple, list)) else r)
                        caps.append((x[0].detach().float().cpu().clone(), str(x[0].dtype), s.detach().float().cpu().clone()))
                    except Exception as exc:  # noqa: BLE001
                        rec.setdefault("capture_errors", []).append(repr(exc)[:300])
                return r
            vae.decode = decode
            rec["vae_hooked"] = True
        except Exception as exc:  # noqa: BLE001
            rec["vae_hooked"] = repr(exc)[:200]

        gen = json.loads(a.gen)
        params = dict(width = gen.pop("width", 1024), height = gen.pop("height", 1024), steps = gen.pop("steps", 8),
                      guidance = gen.pop("guidance", 0.0))
        params.update(gen)
        rec["params"] = params
        sched = [int(x) for x in a.prompt_schedule.split(",") if x.strip()] or [0] * a.renders
        save_set = None if a.save_renders == "all" else {int(x) for x in a.save_renders.split(",") if x.strip()}
        save()
        for r in range(a.renders):
            vae_t.clear()
            step_t.clear()
            caps.clear()
            n0 = len(noise_hits)
            pi = sched[r] if r < len(sched) else 0
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            res = eng.generate(prompt = PROMPTS[pi], seed = SEED, **params)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            img = res["images"][0].convert("RGB")
            arr = np.asarray(img, dtype = np.uint8)
            sps = None
            if len(step_t) >= 3:
                sps = (step_t[-1] - step_t[0]) / (len(step_t) - 1)
            row = {"r": r, "prompt": pi, "wall_s": round(wall, 3), "md5": hashlib.md5(arr.tobytes()).hexdigest(),
                   "shape": list(arr.shape), "n_steps_seen": len(step_t),
                   "s_per_step": round(sps, 4) if sps is not None else None,
                   "s_per_step_fallback": round((wall - sum(vae_t)) / max(1, params["steps"]), 4),
                   "vae_s": round(sum(vae_t), 4) if vae_t else None, "vae_calls": len(vae_t),
                   "noise_hook_hits": len(noise_hits) - n0,
                   "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                   "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}
            if save_set is None or r in save_set:
                img.save(out / f"r{r}.png")
                row["png"] = f"r{r}.png"
            if caps:
                for j, (z, zdt, sm) in enumerate(caps):
                    torch.save({"z": z, "z_dtype": zdt, "studio_out": sm, "prompt": PROMPTS[pi], "seed": SEED,
                                "model": a.model, "params": params}, out / f"vae_cap_r{r}_{j}.pt")
                row["vae_caps"] = [list(c[0].shape) for c in caps]
            rec["renders"].append(row)
            print("RENDER", json.dumps(row), flush = True)
            save()
        rec["verdict"] = "ok"
    except BaseException as exc:  # noqa: BLE001
        rec["verdict"] = "error"
        rec["error"] = repr(exc)[:2000]
        rec["traceback"] = traceback.format_exc()[-6000:]
        print(rec["traceback"], flush = True)
    save()
    print("ARM_DONE", a.tag, rec["verdict"], flush = True)
    os._exit(0 if rec["verdict"] == "ok" else 3)


if __name__ == "__main__":
    main()
