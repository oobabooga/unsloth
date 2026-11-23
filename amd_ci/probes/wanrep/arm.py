#!/usr/bin/env python3
"""One Wan2.2-TI2V-5B arm in a fresh process: load through Studio's VideoBackend, render the same prompt + seed
``--renders`` times, and hash every stage so processes can be compared bit for bit.

Hashed (sha1 of the raw bytes), outside every compiled region:
  te       text encoder forward outputs
  dit      top-level DiT call inputs (hidden_states, timestep, encoder_hidden_states) and output, when the DiT itself is
           not whole-compiled (hooks on a compiled module would be traced into its graph)
  blk      per-block outputs, only with --block-hooks and only when the blocks are NOT compiled
  sched    scheduler.step(model_output, t, sample) -> prev_sample, per step
  vae      vae.decode input / output
  frames   the final uint8 frames
Saved (fp32, under --tensors): per-step model_output and sample, the VAE input, for cross-process magnitudes.
Recorded: inductor ``*.best_config`` picks (TORCHINDUCTOR_CACHE_DIR), triton Autotuner caches (flash_attn etc.),
Studio's ROCm flash bucket verdicts, TunableOp state.
"""
from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROMPT = "a golden retriever running through a sunny meadow, slow motion, shallow depth of field"
SEED = 20261113


def sha(t) -> str:
    import torch

    x = t.detach()
    if x.dtype == torch.bool:
        x = x.to(torch.uint8)
    x = x.contiguous().cpu()
    try:
        b = x.view(torch.uint8).numpy().tobytes()
    except Exception:  # noqa: BLE001
        b = x.float().numpy().tobytes()
    return hashlib.sha1(b + str(tuple(x.shape)).encode() + str(x.dtype).encode()).hexdigest()[:16]


def leaves(obj, path = ""):
    import torch

    if torch.is_tensor(obj):
        yield path, obj
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            yield from leaves(x, f"{path}.{i}")
    elif isinstance(obj, dict):
        for k, x in obj.items():
            yield from leaves(x, f"{path}.{k}")
    elif hasattr(obj, "to_tuple") and callable(obj.to_tuple):
        yield from leaves(obj.to_tuple(), path + ".t")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True)
    ap.add_argument("--model", default = "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--tag", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--tensors", required = True)
    ap.add_argument("--load", default = "{}")
    ap.add_argument("--gen", default = "{}")
    ap.add_argument("--renders", type = int, default = 2)
    ap.add_argument("--block-hooks", action = "store_true")
    ap.add_argument("--deterministic-algos", action = "store_true")
    a = ap.parse_args()
    out = Path(a.out) / a.tag
    out.mkdir(parents = True, exist_ok = True)
    tdir = Path(a.tensors) / a.tag
    tdir.mkdir(parents = True, exist_ok = True)
    rec: dict = {"tag": a.tag, "verdict": "running", "renders": [],
                 "env": {k: v for k, v in os.environ.items()
                         if k.startswith(("UNSLOTH_", "PYTORCH_", "TORCHINDUCTOR_", "FLASH_", "HIP_", "MIOPEN_"))}}
    recp = out / "record.json"

    def save():
        recp.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")

    sys.path.insert(0, str(Path(a.root).resolve()))
    try:
        import logging

        import torch

        logging.basicConfig(level = logging.INFO, stream = sys.stdout, format = "LOG %(name)s %(message)s")
        if a.deterministic_algos:
            torch.use_deterministic_algorithms(True, warn_only = True)
        rec["torch"] = torch.__version__
        rec["hip"] = getattr(torch.version, "hip", None)
        rec["gpu"] = torch.cuda.get_device_name(0)
        try:
            import torch._inductor.config as icfg
            rec["inductor_has_deterministic"] = hasattr(icfg, "deterministic")
            rec["inductor_has_force_filter"] = hasattr(getattr(icfg, "test_configs", None), "force_filter_reduction_configs")
        except Exception as exc:  # noqa: BLE001
            rec["inductor_cfg_error"] = repr(exc)
        try:
            import flash_attn
            rec["flash_attn"] = getattr(flash_attn, "__version__", None)
        except Exception as exc:  # noqa: BLE001
            rec["flash_attn"] = repr(exc)[:200]
        import diffusers
        rec["diffusers"] = diffusers.__version__
        from core.inference import diffusion_speed as DS
        pins: list = []
        pin0 = getattr(DS, "pin_reduction_configs", None)
        if pin0 is not None:
            def pin(kwargs, logger = None):
                r = pin0(kwargs, logger)
                pins.append(bool(r))
                return r
            DS.pin_reduction_configs = pin
        rec["reduction_pins"] = pins
        from core.inference import video as V

        frames_box: dict = {}
        orig = V.VideoBackend._encode_mp4

        def grab(video_frames, fps, audio, pipe, **k):
            frames_box["frames"] = video_frames
            return orig(video_frames, fps, audio, pipe, **k)

        V.VideoBackend._encode_mp4 = staticmethod(grab)
        engine = V.VideoBackend()
        t0 = time.perf_counter()
        engine.begin_load(a.model, local_files_only = False, **json.loads(a.load))
        deadline = time.time() + 5400
        while True:
            p = engine.load_progress() or {}
            if p.get("phase") == "ready":
                break
            if p.get("phase") == "error":
                raise RuntimeError(f"load failed: {p.get('error')}")
            if time.time() > deadline:
                raise TimeoutError(str(p))
            time.sleep(0.5)
        rec["load_s"] = round(time.perf_counter() - t0, 2)
        st = engine.status() or {}
        res = st.get("resolved") or {}
        rec["status"] = {k: st.get(k) for k in ("attention_backend", "transformer_quant", "text_encoder_quant",
                                                 "speed_mode", "speed_optims", "offload_policy")}
        rec["steps_resolved"] = {s.get("step"): s.get("status") for s in ((res.get("speed_mode") or {}).get("steps") or [])}
        rec["attention_resolved"] = res.get("attention_backend")
        pipe = engine._state.pipe
        from core.inference.diffusion_autoopt.targets import denoiser_dits

        dits = denoiser_dits(pipe)
        log: list = []  # (render, stage, index, key, hash)
        cur = {"r": -1}
        counters: dict = {}

        def note(stage, key, h):
            n = counters.get(stage, 0)
            log.append([cur["r"], stage, n, key, h])

        def bump(stage):
            counters[stage] = counters.get(stage, 0) + 1

        handles = []
        te = getattr(pipe, "text_encoder", None)
        if te is not None:
            def te_hook(m, args, kwargs, output):
                for p, x in leaves(output):
                    if x.is_floating_point():
                        note("te", p, sha(x))
                bump("te")
            handles.append(te.register_forward_hook(te_hook, with_kwargs = True))
        rec["dit_compiled_whole"] = [getattr(d, "_compiled_call_impl", None) is not None for d in dits]
        blocks_compiled = []
        for d in dits:
            for b in getattr(d, "blocks", []) or []:
                blocks_compiled.append(getattr(b, "_compiled_call_impl", None) is not None)
        rec["blocks_compiled"] = sum(blocks_compiled)
        rec["blocks"] = len(blocks_compiled)
        for di, d in enumerate(dits):
            if getattr(d, "_compiled_call_impl", None) is not None:
                continue

            def pre(m, args, kwargs, di = di):
                for p, x in leaves((args, kwargs)):
                    note(f"dit{di}_in", p, sha(x))
                return None

            def post(m, args, kwargs, output, di = di):
                for p, x in leaves(output):
                    note(f"dit{di}_out", p, sha(x))
                bump(f"dit{di}_in")
                bump(f"dit{di}_out")
                return None
            handles.append(d.register_forward_pre_hook(pre, with_kwargs = True))
            handles.append(d.register_forward_hook(post, with_kwargs = True))
            if a.block_hooks:
                for bi, b in enumerate(getattr(d, "blocks", []) or []):
                    if getattr(b, "_compiled_call_impl", None) is not None:
                        continue

                    def bpost(m, args, output, di = di, bi = bi, nb = len(d.blocks)):
                        for p, x in leaves(output):
                            note(f"blk{di}", f"{bi}{p}", sha(x))
                        if bi == nb - 1:
                            bump(f"blk{di}")
                        return None
                    handles.append(b.register_forward_hook(bpost))
        patched: set = set()

        def patch_sched():
            s = getattr(pipe, "scheduler", None)
            cls = type(s) if s is not None else None
            if cls is None or cls in patched:
                return
            fn = cls.step

            @functools.wraps(fn)
            def step(self, model_output, timestep, sample, *x, **k):
                i = counters.get("sched", 0)
                note("sched", "model_output", sha(model_output))
                note("sched", "sample", sha(sample))
                note("sched", "t", sha(timestep) if torch.is_tensor(timestep) else str(timestep))
                torch.save({"model_output": model_output.detach().float().cpu(), "sample": sample.detach().float().cpu()},
                           tdir / f"r{cur['r']}_s{i}.pt")
                r = fn(self, model_output, timestep, sample, *x, **k)
                for p, y in leaves(r):
                    note("sched", "prev" + p, sha(y))
                bump("sched")
                return r
            cls.step = step
            patched.add(cls)

        vae = getattr(pipe, "vae", None)
        if vae is not None:
            vdec = vae.decode

            @functools.wraps(vdec)
            def decode(z, *x, **k):
                if torch.is_tensor(z):
                    note("vae", "in", sha(z))
                    torch.save(z.detach().float().cpu(), tdir / f"r{cur['r']}_vae_in_{counters.get('vae', 0)}.pt")
                r = vdec(z, *x, **k)
                for p, y in leaves(r):
                    note("vae", "out" + p, sha(y))
                bump("vae")
                return r
            vae.decode = decode
        save()
        gen = json.loads(a.gen)
        params = dict(width = gen.pop("width", 832), height = gen.pop("height", 480), num_frames = gen.pop("num_frames", 17),
                      steps = gen.pop("steps", 8), guidance = gen.pop("guidance", 5.0), fps = gen.pop("fps", 24))
        params.update(gen)
        rec["params"] = params
        import numpy as np
        for r in range(a.renders):
            cur["r"] = r
            counters.clear()
            patch_sched()
            frames_box.clear()
            torch.cuda.synchronize()
            t = time.perf_counter()
            engine.generate(prompt = PROMPT, seed = SEED, **params)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            row = {"r": r, "wall_s": round(wall, 3), "counters": dict(counters)}
            fr = frames_box.get("frames")
            if fr is not None:
                arr = np.stack([np.asarray(f) for f in fr]) if not hasattr(fr, "shape") else np.asarray(fr)
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                np.save(tdir / f"r{r}_frames.npy", arr)
                row["frames_sha"] = hashlib.sha1(arr.tobytes()).hexdigest()[:16]
                row["frames_shape"] = list(arr.shape)
                note("frames", "uint8", row["frames_sha"])
            rec["renders"].append(row)
            print("RENDER", json.dumps(row), flush = True)
            save()
        for h in handles:
            h.remove()
        rec["log"] = log
        # per-process choices
        try:
            from core.inference import diffusion_attention as DA
            rec["flash_buckets"] = {str(k): v for k, v in getattr(DA, "_ROCM_FLASH_BUCKETS", {}).items()}
        except Exception as exc:  # noqa: BLE001
            rec["flash_buckets"] = repr(exc)[:300]
        try:
            import triton.runtime.autotuner as AT
            tuners = []
            for o in gc.get_objects():
                if isinstance(o, AT.Autotuner):
                    fn = getattr(o, "fn", None)
                    name = getattr(fn, "__name__", None) or getattr(getattr(fn, "fn", None), "__name__", "?")
                    mod = getattr(getattr(fn, "fn", fn), "__module__", "?")
                    cache = getattr(o, "cache", {}) or {}
                    if cache:
                        tuners.append({"fn": f"{mod}.{name}", "n_configs": len(getattr(o, "configs", []) or []),
                                       "cache": {str(k): str(v) for k, v in cache.items()}})
            rec["triton_autotuners"] = tuners
        except Exception as exc:  # noqa: BLE001
            rec["triton_autotuners"] = repr(exc)[:300]
        ind = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if ind and Path(ind).is_dir():
            picks = {}
            for p in Path(ind).rglob("*.best_config"):
                try:
                    picks[p.name] = json.loads(p.read_text(encoding = "utf-8"))
                except Exception:  # noqa: BLE001
                    picks[p.name] = "unreadable"
            rec["inductor_best_configs"] = picks
            # kernels with more than one candidate config: read the generated source's config list length
            multi = {}
            for p in Path(ind).rglob("*.py"):
                try:
                    txt = p.read_text(encoding = "utf-8", errors = "replace")
                except Exception:  # noqa: BLE001
                    continue
                if "@triton_heuristics." in txt and ("reduction" in txt):
                    head = txt.split("@triton.jit", 1)[0]
                    multi[p.stem] = head[-600:]
            rec["inductor_reduction_kernels"] = len(multi)
        try:
            tun = torch.cuda.tunable
            rec["tunableop"] = {"enabled": tun.is_enabled(), "tuning": tun.tuning_is_enabled(),
                                "n_results": len(tun.get_results() or [])}
        except Exception as exc:  # noqa: BLE001
            rec["tunableop"] = repr(exc)[:200]
        rec["verdict"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["verdict"] = "error"
        rec["error"] = repr(exc)[:2000]
        rec["traceback"] = traceback.format_exc()[-6000:]
        print(rec["traceback"], flush = True)
    save()
    print("ARM_DONE", a.tag, rec["verdict"], flush = True)
    os._exit(0 if rec["verdict"] == "ok" else 3)


if __name__ == "__main__":
    main()
