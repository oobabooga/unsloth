#!/usr/bin/env python3
"""One arm in a fresh process: load a family through Studio at its defaults, render the fixed prompt list timed,
then (optionally) a teacher-forced pass.

  python arm.py --root STUDIO_BACKEND --kind image|video --model REPO --tag T --out DIR [--load JSON] [--gen JSON]
                [--prompts 8] [--warmup 1] [--tf none|save|compare] [--tf-ref DIR] [--tf-prompts N]

The arm's attention is set by the caller's environment (UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL=0 native, unset auto,
1 forced). Teacher forcing: the save arm stores, at three denoise steps of each prompt (15 / 50 / 85 % of the
distinct timesteps), every tensor input and output of each DiT call; a compare arm replaces that call's tensor inputs
with the saved ones and reports relative L2 and cosine of its output against the saved output.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

PROMPTS_IMAGE = [
    "a red fox sitting in fresh snow at dawn, soft golden light, highly detailed fur",
    "a futuristic city skyline at night with neon signs reflected in wet streets",
    "a bowl of ramen with a soft boiled egg, steam rising, studio food photography",
    "an old lighthouse on a rocky cliff during a storm, waves crashing",
    "a watercolor painting of a hummingbird drinking from a purple flower",
    "a cozy wooden cabin interior with a fireplace, bookshelves and a sleeping cat",
    "portrait of an elderly fisherman with a weathered face, dramatic side lighting",
    "a vintage red bicycle leaning against a pastel yellow wall in a sunny street",
    "macro photo of a dew covered spider web in the early morning",
    "an astronaut riding a horse on the surface of mars, cinematic wide shot",
]
PROMPTS_VIDEO = [
    "a golden retriever running through a sunny meadow, slow motion, shallow depth of field",
    "waves rolling onto a black sand beach at sunset, aerial drone shot",
    "a chef tossing vegetables in a flaming wok in a busy restaurant kitchen",
    "time lapse of clouds moving over snowy mountain peaks",
    "a paper boat floating down a rain filled street gutter, close up",
    "a hummingbird hovering next to a red flower, macro, slow motion",
    "a steam train crossing a stone bridge in autumn forest, wide shot",
    "a young woman laughing while walking through a busy night market, handheld camera",
    "a koi fish swimming in a clear pond with lily pads, top down view",
    "candles flickering on a birthday cake in a dark room",
]
SEED0 = 20261113


class Sampler(threading.Thread):
    """nvidia-smi per-process used memory and the process tree's RSS, polled; ``take`` returns and resets peaks."""

    def __init__(self, pid: int):
        super().__init__(daemon = True)
        self.pid = pid
        self.smi = 0.0
        self.rss = 0.0
        self.stop_ev = threading.Event()
        self.start()

    def run(self):
        try:
            import psutil

            proc = psutil.Process(self.pid)
        except Exception:  # noqa: BLE001
            proc = None
        while not self.stop_ev.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    capture_output = True, text = True, timeout = 10).stdout
                used = sum(float(l.split(",")[1]) for l in out.strip().splitlines()
                           if l.strip() and int(l.split(",")[0]) == self.pid)
                self.smi = max(self.smi, used / 1024)
            except Exception:  # noqa: BLE001
                pass
            if proc is not None:
                try:
                    rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive = True))
                    self.rss = max(self.rss, rss / 2**30)
                except Exception:  # noqa: BLE001
                    pass
            self.stop_ev.wait(0.5)

    def take(self) -> dict:
        out = {"smi_gib": round(self.smi, 3), "rss_gib": round(self.rss, 3)}
        self.smi = 0.0
        self.rss = 0.0
        return out


def leaves(obj, path = ()):
    import torch

    if torch.is_tensor(obj):
        yield path, obj
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            yield from leaves(x, path + (i,))
    elif isinstance(obj, dict):
        for k, x in obj.items():
            yield from leaves(x, path + (k,))
    elif hasattr(obj, "to_tuple") and callable(obj.to_tuple):
        yield from leaves(obj.to_tuple(), path + ("t",))


def replace(obj, saved: dict, path = ()):
    import torch

    if torch.is_tensor(obj):
        s = saved.get(path)
        if s is not None and tuple(s.shape) == tuple(obj.shape):
            return s.to(device = obj.device, dtype = obj.dtype)
        return obj
    if isinstance(obj, tuple):
        return tuple(replace(x, saved, path + (i,)) for i, x in enumerate(obj))
    if isinstance(obj, list):
        return [replace(x, saved, path + (i,)) for i, x in enumerate(obj)]
    if isinstance(obj, dict):
        return {k: replace(x, saved, path + (k,)) for k, x in obj.items()}
    return obj


def timestep_of(args, kwargs):
    import torch

    for name in ("timestep", "t", "timesteps", "sigma"):
        v = kwargs.get(name)
        if torch.is_tensor(v):
            return round(float(v.flatten()[0].float()), 4)
    if len(args) > 1 and torch.is_tensor(args[1]):
        return round(float(args[1].flatten()[0].float()), 4)
    return None


class TeacherForcing:
    """Forward hooks on every DiT: probe the calls at the chosen distinct timesteps of each prompt."""

    def __init__(self, dits, mode: str, ref: Path, steps: int, pipe):
        self.dits, self.mode, self.ref, self.pipe = dits, mode, ref, pipe
        n = max(1, int(steps))
        self.picks = sorted({min(n - 1, round(f * (n - 1))) for f in (0.15, 0.5, 0.85)})
        self.handles = []
        self.rows = []
        self.prompt = 0
        self.reset()
        for d, dit in enumerate(dits):
            self.handles.append(dit.register_forward_pre_hook(self._pre(d), with_kwargs = True))
            self.handles.append(dit.register_forward_hook(self._post(d), with_kwargs = True))

    def reset(self):
        self.seen: list = []
        self.count: dict = {}
        self.current = None
        self.probed = 0

    def _key(self, d, t):
        if t not in self.seen:
            self.seen.append(t)
        idx = self.seen.index(t)
        occ = self.count.get((d, t), 0)
        self.count[(d, t)] = occ + 1
        return idx, occ

    def _pre(self, d):
        def hook(module, args, kwargs):
            t = timestep_of(args, kwargs)
            if t is None:
                return None
            idx, occ = self._key(d, t)
            self.current = None
            if idx not in self.picks:
                return None
            name = f"p{self.prompt}_d{d}_s{idx}_o{occ}"
            self.current = (name, idx, occ, d)
            if self.mode == "save":
                saved = {p: x.detach().to("cpu", copy = True) for p, x in leaves((args, kwargs))}
                import torch
                torch.save(saved, self.ref / f"{name}_in.pt")
                return None
            path = self.ref / f"{name}_in.pt"
            if not path.exists():
                self.current = None
                return None
            import torch
            saved = torch.load(path)
            new_args, new_kwargs = replace((args, kwargs), saved)
            return tuple(new_args), new_kwargs
        return hook

    def _post(self, d):
        def hook(module, args, kwargs, output):
            if self.current is None:
                return None
            import torch
            name, idx, occ, dd = self.current
            self.current = None
            outs = [x.detach().float() for _, x in leaves(output)]
            if self.mode == "save":
                torch.save([x.to("cpu", torch.float32) for x in outs], self.ref / f"{name}_out.pt")
            else:
                path = self.ref / f"{name}_out.pt"
                if path.exists():
                    ref = torch.load(path)
                    row = {"name": name, "prompt": self.prompt, "dit": dd, "step": idx, "occ": occ, "leaves": []}
                    for i, (a, r) in enumerate(zip(outs, ref)):
                        r = r.to(a.device)
                        if a.shape != r.shape:
                            row["leaves"].append({"leaf": i, "shape_mismatch": [list(a.shape), list(r.shape)]})
                            continue
                        diff = (a - r).flatten()
                        rel = float(diff.norm() / r.flatten().norm().clamp_min(1e-30))
                        cos = float(torch.nn.functional.cosine_similarity(a.flatten(), r.flatten(), dim = 0))
                        row["leaves"].append({"leaf": i, "numel": a.numel(), "rel_l2": rel, "cos": cos})
                    self.rows.append(row)
            self.probed += 1
            if idx >= max(self.picks) and self.pipe is not None:
                # nothing later is probed: skip the rest of the denoise loop where the pipeline honours it
                try:
                    self.pipe._interrupt = True
                except Exception:  # noqa: BLE001
                    pass
            return None
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True)
    ap.add_argument("--kind", default = "image")
    ap.add_argument("--model", required = True)
    ap.add_argument("--tag", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--load", default = "{}")
    ap.add_argument("--gen", default = "{}")
    ap.add_argument("--prompts", type = int, default = 8)
    ap.add_argument("--warmup", type = int, default = 1)
    ap.add_argument("--tf", default = "none")
    ap.add_argument("--tf-ref", default = "")
    ap.add_argument("--tf-prompts", type = int, default = 8)
    ap.add_argument("--save-frames-every", type = int, default = 8)
    ap.add_argument("--profile-steps", type = int, default = 0)
    a = ap.parse_args()
    out = Path(a.out) / a.tag
    out.mkdir(parents = True, exist_ok = True)
    rec = {"tag": a.tag, "model": a.model, "kind": a.kind, "verdict": "running", "renders": [], "warm": [],
           "env": {k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_") or k.startswith("PYTORCH_")}}
    recp = out / "record.json"

    def save():
        recp.write_text(json.dumps(rec, indent = 1, default = str))

    sampler = Sampler(os.getpid())
    sys.path.insert(0, str(Path(a.root).resolve()))
    try:
        import logging

        import torch

        logging.basicConfig(level = logging.INFO, stream = sys.stdout, format = "LOG %(name)s %(message)s")
        from core.inference import diffusion_int8_attention as I8
        try:
            from core.inference import diffusion_int8_attention_dispatch as D
        except Exception as exc:  # noqa: BLE001 - a base tree
            D = None
            rec["dispatch_module"] = repr(exc)
        shapes: dict = {}
        impl0 = I8.int8_attention_impl

        int8_by_device: dict = {}
        mismatch: list = []

        def counted(q, k, v, *xa, **xk):
            key = f"{tuple(q.shape)}x{tuple(k.shape)} {str(q.dtype)[6:]} {q.device}"
            shapes[key] = shapes.get(key, 0) + 1
            if not (q.device == k.device == v.device):
                mismatch.append(f"{q.device}/{k.device}/{v.device}")
            n0 = I8.STATS.get("int8", 0)
            r = impl0(q, k, v, *xa, **xk)
            if I8.STATS.get("int8", 0) > n0:
                int8_by_device[str(q.device)] = int8_by_device.get(str(q.device), 0) + 1
            if r.device != q.device:
                mismatch.append(f"out {r.device} vs q {q.device}")
            return r

        I8.int8_attention_impl = counted
        rec["gpu"] = torch.cuda.get_device_name(0)
        rec["cap"] = list(torch.cuda.get_device_capability(0))
        rec["torch"] = torch.__version__
        import diffusers
        rec["diffusers"] = diffusers.__version__
        load_kw = json.loads(a.load)
        gen_kw = json.loads(a.gen)
        t0 = time.perf_counter()
        if a.kind == "video":
            from core.inference import video as V

            frames_box: dict = {}
            orig = V.VideoBackend._encode_mp4

            def grab(video_frames, fps, audio, pipe, **k):
                frames_box["frames"], frames_box["audio"] = video_frames, audio
                return orig(video_frames, fps, audio, pipe, **k)

            V.VideoBackend._encode_mp4 = staticmethod(grab)
            engine = V.VideoBackend()
        else:
            from core.inference.diffusion import DiffusionBackend

            engine = DiffusionBackend()
        engine.begin_load(a.model, local_files_only = False, **load_kw)
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
        torch.cuda.synchronize()
        rec["load_s"] = round(time.perf_counter() - t0, 2)
        rec["load_mem"] = sampler.take()
        st = engine.status() or {}
        rec["status"] = {k: st.get(k) for k in (
            "repo_id", "family", "attention_backend", "transformer_quant", "text_encoder_quant", "speed_mode",
            "memory_mode", "offload_policy", "speed_optims", "resolved", "generation_defaults", "defaults",
            "kind", "dtype", "precision", "cuda_graphs", "transformer_cache")}
        rec["status_multigpu"] = {k: v for k, v in st.items() if any(x in k.lower() for x in ("gpu", "split", "cfg", "placement", "device"))}
        pipe = engine._state.pipe
        from core.inference.diffusion_autoopt.targets import denoiser_dits
        dits = denoiser_dits(pipe)
        rec["dits"] = [type(d).__name__ for d in dits]
        rec["dispatch_status_load"] = D.status(pipe) if D is not None else None
        rec["int8_status_load"] = I8.status(dits[0] if dits else None)
        try:
            from core.inference import diffusion_attention as DA
            import importlib.metadata as md
            try:
                sv = md.version("sageattention")
            except Exception:  # noqa: BLE001
                sv = None
            rec["sage_seen"] = {"pip_version": sv, "pip_sage2": bool(DA._pip_sage2_installed()),
                                "hub_modules": {n: getattr(m, "__file__", None) for n, m in sys.modules.items()
                                                if n.startswith("_unsloth_sage_hub_") and n.count(".") == 0},
                                "sageattention_file": getattr(sys.modules.get("sageattention"), "__file__", None)}
        except Exception as exc:  # noqa: BLE001
            rec["sage_seen"] = repr(exc)[:300]
        print("LOADED", json.dumps({k: rec.get(k) for k in ("load_s", "status", "dits", "dispatch_status_load")},
                                   default = str), flush = True)
        save()
        # generation parameters: Studio's own defaults for this load
        if a.kind == "video":
            dfl = st.get("defaults") or {}
            presets = dfl.get("resolution_presets") or []
            w = gen_kw.pop("width", None) or dfl.get("default_width") or (presets[0][0] if presets else 1280)
            h = gen_kw.pop("height", None) or dfl.get("default_height") or (presets[0][1] if presets else 704)
            params = dict(width = int(w), height = int(h), num_frames = gen_kw.pop("num_frames", dfl.get("num_frames")),
                          fps = gen_kw.pop("fps", dfl.get("fps")), steps = gen_kw.pop("steps", dfl.get("steps")),
                          guidance = gen_kw.pop("guidance", dfl.get("guidance")))
            prompts = PROMPTS_VIDEO
        else:
            dfl = st.get("generation_defaults") or {}
            params = dict(width = gen_kw.pop("width", 1024), height = gen_kw.pop("height", 1024),
                          steps = gen_kw.pop("steps", dfl.get("steps", 9)),
                          guidance = gen_kw.pop("guidance", dfl.get("guidance", 0.0)))
            prompts = PROMPTS_IMAGE
        params.update(gen_kw)
        rec["params"] = params
        print("PARAMS", params, flush = True)
        # stage split: synchronised wall time at the end of the prompt encode and the start of the VAE decode
        marks: dict = {}

        def wrap(obj, name, key, at_end):
            fn = getattr(obj, name, None)
            if fn is None or getattr(fn, "_arm_wrapped", False):
                return
            import functools

            @functools.wraps(fn)
            def w(*x, **k):
                if not at_end:
                    torch.cuda.synchronize()
                    marks.setdefault(key, time.perf_counter())
                r = fn(*x, **k)
                if at_end:
                    torch.cuda.synchronize()
                    marks[key] = time.perf_counter()
                return r

            w._arm_wrapped = True
            try:
                setattr(obj, name, w)
            except Exception:  # noqa: BLE001
                pass

        wrap(pipe, "encode_prompt", "te_end", True)
        vae = getattr(pipe, "vae", None)
        if vae is not None:
            wrap(vae, "decode", "vae_start", False)

        def render(i: int, prompt: str, seed: int, kind: str):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            sampler.take()
            marks.clear()
            n0 = I8.STATS.get("int8", 0)
            r0 = dict(D.ROUTE_STATS) if D is not None else {}
            t = time.perf_counter()
            if a.kind == "video":
                frames_box.clear()
                res = engine.generate(prompt = prompt, seed = seed, **params)
            else:
                res = engine.generate(prompt = prompt, seed = seed, **params)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            row = {"i": i, "kind": kind, "seed": seed, "wall_s": round(wall, 3),
                   "alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   "reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                   "int8_calls": I8.STATS.get("int8", 0) - n0, **sampler.take()}
            if D is not None:
                row["route_calls"] = {k: v - r0.get(k, 0) for k, v in D.ROUTE_STATS.items() if v - r0.get(k, 0)}
            te_end, vae_start = marks.get("te_end"), marks.get("vae_start")
            if vae_start is not None:
                row["denoise_s"] = round(vae_start - (te_end if te_end is not None and te_end < vae_start else t), 3)
                row["vae_plus_post_s"] = round(t + wall - vae_start, 3)
            if te_end is not None:
                row["te_s"] = round(te_end - t, 3)
            return row, res

        def keep_media(row, res, i):
            import numpy as np
            if a.kind == "video":
                fr = frames_box.get("frames")
                if fr is not None:
                    arr = np.stack([np.asarray(f) for f in fr]) if not hasattr(fr, "shape") else np.asarray(fr)
                    row["nonfinite"] = bool(np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all())
                    if arr.dtype != np.uint8:
                        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                    row["frame_mean"] = float(arr.mean())
                    sub = arr[:: max(1, a.save_frames_every)]
                    np.savez_compressed(out / f"r{i}.npz", frames = sub)
                    row["frames"] = int(arr.shape[0])
                au = frames_box.get("audio")
                if au is not None:
                    try:
                        wav = au[0] if isinstance(au, (tuple, list)) else au
                        wav = wav.detach().float().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav, dtype = np.float32)
                        np.save(out / f"r{i}_audio.npy", wav.astype(np.float32))
                        row["audio_shape"] = list(wav.shape)
                    except Exception as exc:  # noqa: BLE001
                        row["audio_error"] = repr(exc)[:200]
            else:
                img = res["images"][0]
                if isinstance(img, str):
                    import base64
                    import io

                    from PIL import Image
                    img = Image.open(io.BytesIO(base64.b64decode(img.split(",")[-1])))
                img.save(out / f"r{i}.png")

        for j in range(a.warmup):
            row, res = render(-1 - j, prompts[-1 - j], SEED0 + 999 + j, "warm")
            rec["warm"].append(row)
            print("WARM", json.dumps(row), flush = True)
            save()
        for i in range(a.prompts):
            row, res = render(i, prompts[i % len(prompts)], SEED0 + i, "timed")
            keep_media(row, res, i)
            rec["renders"].append(row)
            print("RENDER", json.dumps(row), flush = True)
            save()
        if a.profile_steps:
            # engagement: CUDA kernel names of a short render (never timed)
            from torch.profiler import ProfilerActivity, profile
            pp = dict(params)
            pp["steps"] = a.profile_steps
            try:
                n0 = I8.STATS.get("int8", 0)
                with profile(activities = [ProfilerActivity.CUDA]) as prof:
                    engine.generate(prompt = prompts[0], seed = SEED0, **pp)
                    torch.cuda.synchronize()
                ks = []
                for e in prof.key_averages():
                    t_us = getattr(e, "device_time_total", None)
                    if t_us is None:
                        t_us = getattr(e, "cuda_time_total", 0)
                    ks.append((e.key, t_us, e.count))
                ks.sort(key = lambda x: -x[1])
                words = ("attn", "attention", "sage", "int8", "flash", "fmha", "cudnn", "sdpa", "softmax", "qk_", "i8")
                rec["profile"] = {"steps": a.profile_steps, "int8_calls": I8.STATS.get("int8", 0) - n0,
                                  "top": [[k[:200], round(t / 1000, 2), c] for k, t, c in ks[:12]],
                                  "attn_kernels": [[k[:200], round(t / 1000, 2), c] for k, t, c in ks
                                                   if any(w in k.lower() for w in words)][:15]}
                print("PROFILE", json.dumps(rec["profile"])[:3000], flush = True)
            except Exception as exc:  # noqa: BLE001
                rec["profile"] = {"error": repr(exc)[:400]}
        rec["shapes"] = shapes
        rec["dispatch_status"] = D.status(pipe) if D is not None else None
        rec["int8_stats"] = dict(I8.STATS)
        rec["int8_by_device"] = int8_by_device
        rec["device_mismatch"] = mismatch[:20]
        print("PER_CARD", json.dumps({"int8_by_device": int8_by_device, "shapes": shapes, "mismatch": mismatch[:20],
                                      "route_calls": (D.ROUTE_STATS if D is not None else None)}, default = str), flush = True)
        rec["int8_status"] = I8.status(dits[0] if dits else None)
        try:
            rec["dit_devices"] = {type(d).__name__: sorted({str(p.device) for p in d.parameters()}) for d in dits}
            rec["route_cards"] = sorted({str(getattr(m.processor._attention_backend, "cards", None)) for d in dits
                                         for m in d.modules() if hasattr(getattr(m, "processor", None), "_attention_backend")})
        except Exception as exc:  # noqa: BLE001
            rec["dit_devices"] = repr(exc)[:200]
        save()
        if a.tf != "none":
            ref = Path(a.tf_ref)
            ref.mkdir(parents = True, exist_ok = True)
            tf = TeacherForcing(dits, a.tf, ref, int(params.get("steps") or 9), pipe)
            rec["tf_picks"] = tf.picks
            t_tf = time.perf_counter()
            for i in range(a.tf_prompts):
                tf.prompt = i
                tf.reset()
                try:
                    engine.generate(prompt = prompts[i % len(prompts)], seed = SEED0 + i, **params)
                except Exception as exc:  # noqa: BLE001
                    rec.setdefault("tf_errors", []).append(repr(exc)[:300])
                print("TF", i, "probed", tf.probed, flush = True)
                rec.setdefault("tf_probed", []).append(tf.probed)
            tf.close()
            rec["tf_s"] = round(time.perf_counter() - t_tf, 1)
            rec["tf_rows"] = tf.rows
        rec["verdict"] = "ok"
    except Exception as exc:  # noqa: BLE001
        rec["verdict"] = "error"
        rec["error"] = repr(exc)[:2000]
        rec["traceback"] = traceback.format_exc()[-6000:]
        print(rec["traceback"], flush = True)
    rec["ru_maxrss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)
    save()
    print("ARM_DONE", a.tag, rec["verdict"], flush = True)
    sampler.stop_ev.set()
    os._exit(0 if rec["verdict"] == "ok" else 3)


if __name__ == "__main__":
    main()
