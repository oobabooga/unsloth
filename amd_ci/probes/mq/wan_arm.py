#!/usr/bin/env python3
"""One Wan arm in a fresh process (M13 MIOpen db A/B on Windows gfx1151, M21 A14B on Linux gfx1151, and the NVIDIA
reference). Loads through Studio's VideoBackend (``--load`` kwargs, {} = Studio default), renders ``--renders`` times
and records per render: wall s, s/step (synchronised timestamps at every scheduler step), VAE decode s (``vae.decode``
wrapped after load, so Studio's load-time decode wrappers are inside the timing and its render-time scopes outside),
the decode tail (last step to return), peak allocated / reserved GiB, host RAM low-water mark, md5 of the uint8 frames,
sha of the VAE input latent. Initial noise: with ``--noise-fp32`` the pipeline's ``prepare_latents`` output is replaced
by an fp32 CPU draw from the seed (same noise on gfx1151, NVIDIA and CPU). Observes only."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path

PROMPT = "a golden retriever running through a sunny meadow, slow motion, shallow depth of field"
SEED = 20261124


class _Keep(logging.Handler):
    KEYS = ("fallback", "fall back", "falls back", "refus", "disabled", "not installed", "error", "failed", "skip",
            "warning", "unsupported", "offload", "route", "engine", "compile", "miopen", "tunableop", "vae", "decode",
            "engagement", "quant", "plan", "resident", "tile", "untiled")

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
            if len(self.lines) < 600:
                self.lines.append(f"{record.levelname} {record.name}: {msg[:600]}")


class _HostMem(threading.Thread):
    """Host RAM low-water mark (unified memory: the GPU heap is host RAM) and this process's peak RSS."""

    def __init__(self):
        super().__init__(daemon = True)
        self.min_avail = None
        self.max_rss = 0
        self.stop = False

    def run(self):
        try:
            import psutil
        except Exception:  # noqa: BLE001
            return
        p = psutil.Process()
        while not self.stop:
            try:
                av = psutil.virtual_memory().available
                self.min_avail = av if self.min_avail is None else min(self.min_avail, av)
                self.max_rss = max(self.max_rss, p.memory_info().rss)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)

    def reset(self):
        self.min_avail, self.max_rss = None, 0


def sha(t) -> str:
    import torch

    x = t.detach().contiguous().cpu()
    try:
        b = x.view(torch.uint8).numpy().tobytes()
    except Exception:  # noqa: BLE001
        b = x.float().numpy().tobytes()
    return hashlib.sha1(b + str(tuple(x.shape)).encode() + str(x.dtype).encode()).hexdigest()[:16]


def dir_listing(d):
    out = {}
    if d and Path(d).is_dir():
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                out[str(p.relative_to(d))] = p.stat().st_size
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True, help = "studio/backend of the tree under test")
    ap.add_argument("--model", required = True)
    ap.add_argument("--tag", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--load", default = "{}")
    ap.add_argument("--gen", default = "{}")
    ap.add_argument("--renders", type = int, default = 2)
    ap.add_argument("--noise-fp32", action = "store_true")
    ap.add_argument("--save-png-every", type = int, default = 0, help = "save every k-th frame of the LAST render")
    ap.add_argument("--save-npy", default = "", help = "save the LAST render's uint8 frames here (.npy)")
    a = ap.parse_args()
    out = Path(a.out) / a.tag
    out.mkdir(parents = True, exist_ok = True)
    rec: dict = {"tag": a.tag, "verdict": "running", "renders": [], "model": a.model, "load": json.loads(a.load),
                 "seed": SEED, "prompt": PROMPT, "t_start": time.time(),
                 "env": {k: v for k, v in os.environ.items()
                         if k.startswith(("UNSLOTH_", "PYTORCH_", "TORCHINDUCTOR_", "HIP_", "MIOPEN_", "HSA_", "HF_HUB_"))}}
    recp = out / "record.json"
    keep = _Keep()
    logging.getLogger().addHandler(keep)
    logging.getLogger().setLevel(logging.INFO)
    rec["log_lines"] = keep.lines
    mem = _HostMem()
    mem.start()

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
            rec["cudnn_or_miopen_version"] = torch.backends.cudnn.version()
        except Exception as exc:  # noqa: BLE001
            rec["cudnn_or_miopen_version"] = repr(exc)[:200]
        try:
            props = torch.cuda.get_device_properties(0)
            rec["gcn_arch"] = getattr(props, "gcnArchName", None)
            rec["multi_processor_count"] = getattr(props, "multi_processor_count", None)
            rec["total_mem_gib"] = round(props.total_memory / 2 ** 30, 2)
        except Exception:  # noqa: BLE001
            pass
        try:
            import diffusers
            import transformers
            rec["diffusers"], rec["transformers"] = diffusers.__version__, transformers.__version__
        except Exception:  # noqa: BLE001
            pass
        from core.inference import video as V

        frames_box: dict = {}
        enc_t: list = []
        orig = V.VideoBackend._encode_mp4

        def grab(video_frames, fps, audio, pipe, **k):
            frames_box["frames"] = video_frames
            t1 = time.perf_counter()
            r = orig(video_frames, fps, audio, pipe, **k)
            enc_t.append(time.perf_counter() - t1)
            return r

        V.VideoBackend._encode_mp4 = staticmethod(grab)
        engine = V.VideoBackend()
        t0 = time.perf_counter()
        engine.begin_load(a.model, local_files_only = False, **json.loads(a.load))
        deadline = time.time() + float(os.environ.get("MQ_LOAD_DEADLINE_S", "14400"))
        last_phase = None
        while True:
            p = engine.load_progress() or {}
            if p.get("phase") != last_phase:
                last_phase = p.get("phase")
                print("LOAD_PHASE", round(time.perf_counter() - t0, 1), json.dumps(p, default = str)[:400], flush = True)
            if p.get("phase") == "ready":
                break
            if p.get("phase") == "error" or p.get("error"):
                raise RuntimeError(f"load failed: {p.get('error')}")
            if time.time() > deadline:
                raise TimeoutError(str(p))
            time.sleep(0.5)
        rec["load_s"] = round(time.perf_counter() - t0, 2)
        rec["load_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        rec["load_min_host_avail_gib"] = round((mem.min_avail or 0) / 2 ** 30, 2)
        st = engine.status() or {}
        rec["status"] = {k: (v if isinstance(v, (str, int, float, bool, type(None), list)) else str(v))
                         for k, v in st.items() if not isinstance(v, dict)}
        res = st.get("resolved") or {}
        rec["resolved"] = json.loads(json.dumps(res, default = str))
        pipe = engine._state.pipe
        rec["pipeline_class"] = type(pipe).__name__
        try:
            rec["vae_dtype"] = str(pipe.vae.dtype)
        except Exception:  # noqa: BLE001
            pass
        rec["miopen_immediate"] = getattr(getattr(torch.backends, "miopen", None), "immediate", None)

        noise_info: list = []
        rec["noise_info"] = noise_info
        if callable(getattr(pipe, "prepare_latents", None)):
            pl0 = pipe.prepare_latents

            def prepare_latents(*args, **kw):
                o = pl0(*args, **kw)
                t_ = o[0] if isinstance(o, (tuple, list)) else o
                if a.noise_fp32 and torch.is_tensor(t_):
                    g = torch.Generator("cpu").manual_seed(SEED)
                    z_ = torch.randn(tuple(t_.shape), generator = g, dtype = torch.float32).to(t_.device, t_.dtype)
                    o = (z_, *o[1:]) if isinstance(o, (tuple, list)) else z_
                    t_ = z_
                try:
                    c = t_.detach().float().cpu().contiguous()
                    noise_info.append({"shape": list(t_.shape), "dtype": str(t_.dtype),
                                       "md5": hashlib.md5(c.numpy().tobytes()).hexdigest(), "std": float(c.std())})
                except Exception as exc:  # noqa: BLE001
                    noise_info.append({"error": repr(exc)[:200]})
                return o
            pipe.prepare_latents = prepare_latents

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
        vae_in: list = []
        try:
            vae = pipe.vae
            vdec = vae.decode

            def decode(*x, **k):
                if x and torch.is_tensor(x[0]):
                    vae_in.append([sha(x[0]), list(x[0].shape), str(x[0].dtype)])
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                r = vdec(*x, **k)
                torch.cuda.synchronize()
                vae_t.append(time.perf_counter() - t1)
                return r
            vae.decode = decode
            rec["vae_hooked"] = True
        except Exception as exc:  # noqa: BLE001
            rec["vae_hooked"] = repr(exc)[:200]

        gen = json.loads(a.gen)
        params = dict(prompt = PROMPT, seed = SEED)
        params.update(gen)
        rec["params"] = params
        save()
        for r in range(a.renders):
            vae_t.clear()
            vae_in.clear()
            step_t.clear()
            enc_t.clear()
            frames_box.clear()
            n0 = len(noise_info)
            mem.reset()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            engine.generate(**params)
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            wall = t_end - t
            row = {"r": r, "wall_s": round(wall, 3), "n_steps_seen": len(step_t),
                   "first_step_s": round(step_t[0] - t, 3) if step_t else None,
                   "s_per_step": round((step_t[-1] - step_t[0]) / (len(step_t) - 1), 4) if len(step_t) >= 2 else None,
                   "step_s": [round(b - a_, 3) for a_, b in zip(step_t, step_t[1:])],
                   "tail_s": round(t_end - step_t[-1], 3) if step_t else None,
                   "vae_s": [round(x, 4) for x in vae_t], "vae_in": list(vae_in),
                   "encode_mp4_s": [round(x, 3) for x in enc_t],
                   "noise_hook_hits": len(noise_info) - n0,
                   "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                   "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3),
                   "min_host_avail_gib": round((mem.min_avail or 0) / 2 ** 30, 2),
                   "max_rss_gib": round(mem.max_rss / 2 ** 30, 2)}
            fr = frames_box.get("frames")
            if fr is not None:
                arr = np.stack([np.asarray(f) for f in fr]) if not hasattr(fr, "shape") else np.asarray(
                    fr.cpu() if torch.is_tensor(fr) else fr)
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                row["md5"] = hashlib.md5(arr.tobytes()).hexdigest()
                row["frames_shape_raw"] = list(arr.shape)
                try:
                    while arr.ndim > 4 and arr.shape[0] == 1:
                        arr = arr[0]
                    if arr.ndim == 4 and arr.shape[-1] not in (3, 4) and arr.shape[1] in (3, 4):
                        arr = arr.transpose(0, 2, 3, 1)
                    row["frames_shape"] = list(arr.shape)
                    if r == a.renders - 1:
                        if a.save_npy:
                            Path(a.save_npy).parent.mkdir(parents = True, exist_ok = True)
                            np.save(a.save_npy, np.ascontiguousarray(arr))
                        if a.save_png_every > 0:
                            from PIL import Image
                            for i in range(0, arr.shape[0], a.save_png_every):
                                Image.fromarray(np.ascontiguousarray(arr[i][..., :3])).save(out / f"f{i:03d}.png")
                except Exception as exc:  # noqa: BLE001
                    row["save_error"] = repr(exc)[:300]
            rec["renders"].append(row)
            print("RENDER", json.dumps(row), flush = True)
            save()
        ud = os.environ.get("MIOPEN_USER_DB_PATH")
        rec["miopen_user_db"] = dir_listing(ud)
        kc = os.environ.get("MIOPEN_CUSTOM_CACHE_DIR")
        rec["miopen_kernel_cache_bytes"] = sum(dir_listing(kc).values()) if kc else None
        rec["verdict"] = "ok"
    except BaseException as exc:  # noqa: BLE001
        rec["verdict"] = "error"
        rec["error"] = repr(exc)[:2000]
        rec["traceback"] = traceback.format_exc()[-6000:]
        print(rec["traceback"], flush = True)
    mem.stop = True
    rec["t_end"] = time.time()
    save()
    print("ARM_DONE", a.tag, rec["verdict"], flush = True)
    os._exit(0 if rec["verdict"] == "ok" else 3)


if __name__ == "__main__":
    main()
