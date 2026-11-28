#!/usr/bin/env python3
"""One image arm in a fresh process through Studio's own DiffusionBackend load + generate at Studio defaults (only
width / height / steps / guidance from --gen are passed).

Before the load every file of the model's hub cache is fsync'd and dropped from the page cache (posix_fadvise), and
the resident fraction is measured (mincore), so the load is cold. Records: load s, peak RSS (VmHWM) at load end and at
the end, per render wall s, text-encode s (``pipe.encode_prompt`` wrapped, synchronised), s/step (scheduler.step,
synchronised), VAE s, md5 of the image, the RDNA W8A8 GEMM launches by row class (engagement of the small-row tiles),
whether the pread scope routed any shard, and the load's resolved status.

  e2e_arm.py --root <tree>/studio/backend --model REPO --out DIR --tag T --gen '{"width":1024,"height":1024}' --renders 5
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import mmap
import os
import sys
import time
import traceback
from pathlib import Path

PROMPT = "a red fox sitting in fresh snow at golden hour, detailed fur, shallow depth of field"
SEED = 20261125


def repo_files(repo: str) -> list:
    hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    d = hub / ("models--" + repo.replace("/", "--"))
    out = []
    for p in (d / "blobs").glob("*") if (d / "blobs").is_dir() else []:
        if p.is_file():
            out.append(str(p))
    return out


def drop_cache(files: list) -> float:
    libc = ctypes.CDLL("libc.so.6", use_errno = True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    page = os.sysconf("SC_PAGE_SIZE")
    tot = res = 0
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            if size == 0:
                continue
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            mm = mmap.mmap(fd, size, access = mmap.ACCESS_COPY)
            n = (size + page - 1) // page
            vec = (ctypes.c_ubyte * n)()
            view = (ctypes.c_char * 1).from_buffer(mm)
            if libc.mincore(ctypes.c_void_p(ctypes.addressof(view)), ctypes.c_size_t(size), vec) == 0:
                res += sum(1 for b in vec if b & 1)
            tot += n
            del view
            mm.close()
        finally:
            os.close(fd)
    return res / max(tot, 1)


def meminfo(key: str) -> float:
    try:
        for line in open("/proc/meminfo", encoding = "utf-8"):
            if line.startswith(key + ":"):
                return round(int(line.split()[1]) / 2 ** 20, 3)
    except Exception:  # noqa: BLE001
        pass
    return -1.0


def install_trace(trace: list) -> None:
    """Wrap every safetensors / torch.load entry point a loader can reach; record file, backend, size and the first
    Studio / diffusers / transformers frame that called it."""
    import functools
    import traceback as tb

    import safetensors
    import safetensors.torch as st_torch
    import torch

    def caller():
        for fr in reversed(tb.extract_stack()[:-3]):
            f = fr.filename.replace("\\", "/")
            for tag in ("core/inference/", "diffusers/", "transformers/"):
                if tag in f and "safetensors_pread" not in f:
                    return f"{f.split(tag)[-1]}:{fr.name}:{fr.lineno}"
        return None

    def wrap(owner, attr, label, path_of):
        real = getattr(owner, attr, None)
        if real is None:
            return

        @functools.wraps(real)
        def traced(*x, **k):
            try:
                path = str(path_of(x, k))
                size = os.path.getsize(path) if os.path.isfile(path) else None
                trace.append({"fn": label, "file": "/".join(path.replace("\\", "/").split("/")[-2:]),
                              "backend": k.get("backend"), "gb": round(size / 1e9, 3) if size else None,
                              "caller": caller()})
            except Exception:  # noqa: BLE001
                pass
            return real(*x, **k)
        setattr(owner, attr, traced)

    first = lambda x, k: x[0] if x else (k.get("filename") or k.get("f"))  # noqa: E731
    wrap(safetensors, "safe_open", "safetensors.safe_open", first)
    wrap(st_torch, "safe_open", "safetensors.torch.safe_open", first)
    wrap(st_torch, "load_file", "safetensors.torch.load_file", first)
    wrap(torch, "load", "torch.load", first)
    try:
        import transformers.modeling_utils as mu
        wrap(mu, "safe_open", "transformers.safe_open", first)
    except Exception:  # noqa: BLE001
        pass
    for mod in ("diffusers.models.model_loading_utils", "diffusers.models.modeling_utils",
                "diffusers.loaders.single_file_utils"):
        try:
            import importlib
            m = importlib.import_module(mod)
            wrap(m, "load_state_dict", f"{mod}.load_state_dict", first)
        except Exception:  # noqa: BLE001
            pass


def vm(key: str) -> float:
    if not sys.platform.startswith("linux"):
        try:
            import psutil
            mi = psutil.Process().memory_info()
            return round(getattr(mi, "peak_wset" if key == "VmHWM" else "rss", mi.rss) / 2 ** 30, 3)
        except Exception:  # noqa: BLE001
            return -1.0
    for line in open("/proc/self/status", encoding = "utf-8"):
        if line.startswith(key + ":"):
            return round(int(line.split()[1]) / 2 ** 20, 3)
    return -1.0


T_START = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required = True)
    ap.add_argument("--model", required = True)
    ap.add_argument("--tag", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--gen", default = "{}")
    ap.add_argument("--load", default = "{}")
    ap.add_argument("--renders", type = int, default = 5)
    ap.add_argument("--cold", action = "store_true")
    ap.add_argument("--trace", action = "store_true", help = "record every safetensors / torch.load open during the load")
    ap.add_argument("--checksum", action = "store_true", help = "checksum of the loaded transformer parameters")
    ap.add_argument("--gemm-probe", action = "store_true", help = "bf16 4096^3 GEMM TFLOP/s before and after every render")
    ap.add_argument("--mark-times", action = "store_true",
                    help = "record CLOCK_BOOTTIME / CLOCK_MONOTONIC ns at each render's start and end (profiler windows)")
    ap.add_argument("--vae-swap", action = "store_true", help = "after the renders: real-VAE decode MIOpen vs aiter conv2d")
    ap.add_argument("--mod-probe", action = "store_true",
                    help = "after the timed renders: one extra hooked render, then time each modulation Linear alone")
    ap.add_argument("--torch-profile", type = int, default = -1,
                    help = "torch.profiler around this render index; kernel table -> <tag>_kernels.json")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents = True, exist_ok = True)
    rec: dict = {"tag": a.tag, "verdict": "running", "root": a.root, "model": a.model, "renders": []}
    recp = out / f"{a.tag}.json"

    def save():
        recp.write_text(json.dumps(rec, indent = 1, default = str), encoding = "utf-8")
    sys.path.insert(0, str(Path(a.root).resolve()))
    try:
        if a.cold and sys.platform.startswith("linux"):
            files = repo_files(a.model)
            hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
            every = [str(p) for p in hub.glob("models--*/blobs/*") if p.is_file()]  # companion repos too (TE prequant)
            rec["cache_files"], rec["cache_gb"] = len(files), round(sum(os.path.getsize(f) for f in files) / 1e9, 2)
            rec["hub_gb"] = round(sum(os.path.getsize(f) for f in every) / 1e9, 2)
            rec["resident_before"] = drop_cache(every)
            rec["cached_before_gib"] = meminfo("Cached")
        import numpy as np
        import torch

        rec["torch"], rec["hip"] = torch.__version__, getattr(torch.version, "hip", None)
        from core.inference import diffusion_native_w8a8_rdna as R
        from core.inference import safetensors_pread as SP
        rec["module_file"] = R.__file__
        mm = {"small": 0, "large": 0}
        orig_launch = R._launch_mm

        def launch(xq, xs, wq, ws, bias, out_, m, n, k):
            mm["small" if m < 1024 else "large"] += 1
            return orig_launch(xq, xs, wq, ws, bias, out_, m, n, k)
        R._launch_mm = launch
        pread_calls = {"n": 0}
        orig_backend_for = SP._backend_for

        def backend_for(filename, *x, **k):
            r = orig_backend_for(filename, *x, **k)
            if r == "pread":
                pread_calls["n"] += 1
            return r
        SP._backend_for = backend_for
        trace: list = []
        rec["trace"] = trace
        if a.trace:
            install_trace(trace)
        rec["pread_wanted"] = SP.pread_wanted()
        rec["rocm_integrated"] = getattr(SP, "rocm_integrated", lambda: None)()
        from core.inference import diffusion_adaln_precompute as AD
        ad_calls: list = []
        rec["adaln_install_calls"] = ad_calls
        rec["adaln_env"] = {k: os.environ.get(k) for k in (AD.ENV, AD.VERIFY_ENV)}
        rec["adaln_mode"] = AD.mode()
        ad_install0 = AD.install

        def ad_install(model, pipe = None, **k):
            why = None
            try:
                why = AD.decline_reason(model, offloaded = k.get("offloaded", False), requested = k.get("requested"),
                                        split = k.get("split", False))
            except Exception as exc:  # noqa: BLE001
                why = f"decline_reason raised {exc!r}"
            r = ad_install0(model, pipe, **k)
            ad_calls.append({"cls": type(model).__name__, "offloaded": k.get("offloaded"), "split": k.get("split"),
                             "decline_reason": why, "returned": r,
                             "adapter": type(AD.adapter_for(model)).__name__ if model is not None else None})
            return r
        AD.install = ad_install
        from core.inference.diffusion import DiffusionBackend
        eng = DiffusionBackend()
        t0 = time.perf_counter()
        eng.begin_load(a.model, **json.loads(a.load))
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
        rec["load_vmhwm_gib"] = vm("VmHWM")
        rec["cached_after_load_gib"] = meminfo("Cached")
        rec["load_rss_gib"] = vm("VmRSS")
        rec["pread_shards"] = pread_calls["n"]
        rec["fadvise_calls"] = getattr(SP, "fadvise_calls", None)
        if rec.get("cached_before_gib", -1) >= 0 and rec["cached_after_load_gib"] >= 0:
            rec["cache_growth_gib"] = round(rec["cached_after_load_gib"] - rec["cached_before_gib"], 3)
        st = eng.status() or {}
        rec["status"] = {k: (v if isinstance(v, (str, int, float, bool, type(None), list)) else str(v))
                         for k, v in st.items() if not isinstance(v, dict)}
        rec["rdna_state"] = {k: str(v)[:200] for k, v in R._STATE.items()}
        pipe = eng._state.pipe
        rec["pipeline_class"] = type(pipe).__name__
        enc_t: list = []
        if callable(getattr(pipe, "encode_prompt", None)):
            ep0 = pipe.encode_prompt

            def encode_prompt(*x, **k):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                r = ep0(*x, **k)
                torch.cuda.synchronize()
                enc_t.append(time.perf_counter() - t1)
                return r
            pipe.encode_prompt = encode_prompt
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
        try:
            vdec = pipe.vae.decode

            def decode(*x, **k):
                if x and hasattr(x[0], "shape"):
                    vae_in["z"] = x[0].detach()
                    vae_in["k"] = dict(k)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                r = vdec(*x, **k)
                torch.cuda.synchronize()
                vae_t.append(time.perf_counter() - t1)
                return r
            pipe.vae.decode = decode
        except Exception:  # noqa: BLE001
            pass
        if a.checksum:
            acc = torch.zeros((), dtype = torch.float64, device = "cuda")
            n = 0
            for i, (name, prm) in enumerate(pipe.transformer.named_parameters()):
                acc += prm.detach().float().abs().sum().double() * (1 + (i % 97) / 97.0)
                n += 1
            rec["transformer_checksum"], rec["transformer_params"] = float(acc), n
            for comp in ("text_encoder", "vae"):
                mod = getattr(pipe, comp, None)
                if mod is None or not hasattr(mod, "named_parameters"):
                    continue
                acc = torch.zeros((), dtype = torch.float64)
                n = 0
                for i, (name, prm) in enumerate(mod.named_parameters()):
                    acc += prm.detach().float().abs().sum().double().cpu() * (1 + (i % 97) / 97.0)
                    n += 1
                for i, (name, buf) in enumerate(mod.named_buffers()):
                    if buf.is_floating_point() or buf.dtype in (torch.int8, torch.uint8):
                        acc += buf.detach().float().abs().sum().double().cpu() * (1 + (i % 89) / 89.0)
                        n += 1
                rec[f"{comp}_checksum"], rec[f"{comp}_params"] = float(acc), n
        gen = json.loads(a.gen)
        if "steps" not in gen and a.renders:
            from core.inference.diffusion_families import default_generation_params
            steps, guidance = default_generation_params(a.model)
            gen = dict(gen, steps = steps, guidance = guidance)
        rec["params"] = gen
        save()

        def gemm_tflops():
            x = torch.randn(4096, 4096, device = "cuda", dtype = torch.bfloat16)
            for _ in range(3):
                x @ x
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(20):
                x @ x
            torch.cuda.synchronize()
            return round(20 * 2 * 4096 ** 3 / (time.perf_counter() - t) / 1e12, 2)
        if a.gemm_probe:
            rec["gemm_tflops_before"] = gemm_tflops()
        for r in range(a.renders):
            enc_t.clear()
            step_t.clear()
            vae_t.clear()
            m0 = dict(mm)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            marks = {}
            if a.mark_times:
                marks["t0_ns_boot"] = time.clock_gettime_ns(getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC))
                marks["t0_ns_mono"] = time.monotonic_ns()
            prof = None
            if r == a.torch_profile:
                from torch.profiler import ProfilerActivity, profile
                prof = profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA])
                prof.__enter__()
            t = time.perf_counter()
            res = eng.generate(prompt = PROMPT, seed = SEED, **gen)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            if a.mark_times:
                marks["t1_ns_boot"] = time.clock_gettime_ns(getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC))
                marks["t1_ns_mono"] = time.monotonic_ns()
            if prof is not None:
                prof.__exit__(None, None, None)
                ktab = []
                for ev in prof.key_averages():
                    dev_us = float(getattr(ev, "self_device_time_total", None) or getattr(ev, "self_cuda_time_total", 0) or 0)
                    if dev_us > 0:
                        ktab.append({"name": ev.key, "count": int(ev.count), "device_us": dev_us})
                ktab.sort(key = lambda x: -x["device_us"])
                (out / f"{a.tag}_kernels.json").write_text(json.dumps({"render": r, "wall_s": wall, "kernels": ktab}),
                                                          encoding = "utf-8")
                del prof
            img = res["images"][0].convert("RGB")
            arr = np.asarray(img, dtype = np.uint8)
            row = {"r": r, "wall_s": round(wall, 3), "md5": hashlib.md5(arr.tobytes()).hexdigest(),
                   "encode_s": [round(x, 4) for x in enc_t], "n_steps": len(step_t),
                   "s_per_step": round((step_t[-1] - step_t[0]) / (len(step_t) - 1), 4) if len(step_t) >= 2 else None,
                   "first_step_s": round(step_t[0] - t, 3) if step_t else None,
                   "vae_s": [round(x, 4) for x in vae_t],
                   "rdna_mm": {k: mm[k] - m0[k] for k in mm},
                   "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                   "pix_mean": float(arr.mean())}
            row.update(marks)
            st = eng.status() or {}
            row["speed_mode"], row["speed_optims"] = st.get("speed_mode"), st.get("speed_optims")
            row["transformer_quant"], row["bg_compile"] = st.get("transformer_quant"), str(st.get("bg_compile"))[:300]
            row["t_since_start_s"] = round(time.time() - T_START, 1)
            if a.gemm_probe:
                row["gemm_tflops_after"] = gemm_tflops()
            if r == a.renders - 1:
                img.save(out / f"{a.tag}.png")
            rec["renders"].append(row)
            print("RENDER", json.dumps(row), flush = True)
            save()
        tr = getattr(pipe, "transformer", None)
        rec["adaln_counts"] = dict(AD.COUNTS)
        rec["adaln_installed"] = bool(tr is not None and AD.installed(tr))
        try:
            rec["adaln_summary"] = {k: str(v)[:200] for k, v in (AD.summary(tr) or {}).items()} if tr is not None else None
        except Exception as exc:  # noqa: BLE001
            rec["adaln_summary"] = repr(exc)[:200]
        rec["transformer_class"] = type(tr).__name__ if tr is not None else None
        try:
            rec["vae_class"], rec["vae_dtype"] = type(pipe.vae).__name__, str(next(pipe.vae.parameters()).dtype)
        except Exception as exc:  # noqa: BLE001
            rec["vae_dtype"] = repr(exc)[:200]
        rec["proc_env"] = {k: os.environ.get(k) for k in ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "MIOPEN_SEARCH_CUTOFF",
                                                          "MIOPEN_USER_DB_PATH")}
        if a.mod_probe and tr is not None:
            # One extra, untimed render with counting pre-hooks on the per-call modulation Linears; then each is timed
            # alone on its recorded input: (calls per render) x (ms per call) bounds what an AdaLN table could save.
            names = [n for n, _m in tr.named_modules() if n in ("modulation.1", "norm_out.linear")
                     or n.endswith(("img_mod.1", "txt_mod.1", "norm1.linear", "norm1_context.linear"))]
            cnt, last, hs = {}, {}, []
            for n in names:
                mod = tr.get_submodule(n)

                def pre(_m, args, _n = n):
                    cnt[_n] = cnt.get(_n, 0) + 1
                    if args and hasattr(args[0], "shape"):
                        last[_n] = args[0].detach()
                hs.append(mod.register_forward_pre_hook(pre))
            try:
                eng.generate(prompt = PROMPT, seed = SEED, **gen)
                torch.cuda.synchronize()
            finally:
                for h in hs:
                    h.remove()
            mp = {}
            with torch.no_grad():
                for n, x in last.items():
                    mod = tr.get_submodule(n)
                    for _ in range(3):
                        mod(x)
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                    e0.record()
                    for _ in range(30):
                        mod(x)
                    e1.record()
                    e1.synchronize()
                    ms = e0.elapsed_time(e1) / 30
                    w = getattr(mod, "weight", None)
                    mp[n] = {"calls_per_render": cnt.get(n), "ms_per_call": round(ms, 4), "in_shape": list(x.shape),
                             "module": type(mod).__name__, "weight_dtype": str(getattr(w, "dtype", None)),
                             "weight_shape": list(w.shape) if w is not None else None,
                             "s_per_render": round(ms * cnt.get(n, 0) / 1000, 4)}
            rec["mod_probe"] = {"sites": mp, "site_names": names,
                                "total_s_per_render": round(sum(v["s_per_render"] for v in mp.values()), 4)}
            save()
        if a.vae_swap and vae_in.get("z") is not None:
            # Real weights, Studio's loaded VAE and its last latent: every nn.Conv2d MIOpen (M) vs the vendored aiter
            # Triton conv2d (T), M T M T M T M T (first pair = warm-up / Triton compile), PSNR of T vs M.
            import math
            import statistics as stt
            import torch.nn as nn
            sys.path.insert(0, str(Path(__file__).resolve().parent / "aiter_pkg"))
            from aiter.ops.triton.conv.conv2d import conv2d as aconv
            route = {"on": False, "n": 0, "err": []}
            o2 = nn.Conv2d._conv_forward

            def fwd(self, input, weight, bias):
                if route["on"] and self.groups == 1 and self.padding_mode == "zeros" and \
                        input.dtype in (torch.bfloat16, torch.float16) and not isinstance(self.padding, str):
                    try:
                        y = aconv(input, weight, bias, self.stride, self.padding, self.dilation, layout = "nchw")
                        route["n"] += 1
                        return y
                    except Exception as exc:  # noqa: BLE001
                        route["err"].append(repr(exc)[:200])
                return o2(self, input, weight, bias)
            nn.Conv2d._conv_forward = fwd
            sw = {"s": {"M": [], "T": []}, "routed": {}, "z_shape": list(vae_in["z"].shape), "z_dtype": str(vae_in["z"].dtype)}
            ims = {}
            try:
                with torch.no_grad():
                    for arm in "MTMTMTMT":
                        route["on"] = arm == "T"
                        n0 = route["n"]
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        y = vdec(vae_in["z"], **vae_in["k"])
                        torch.cuda.synchronize()
                        sw["s"][arm].append(round(time.perf_counter() - t1, 4))
                        sw["routed"][arm] = route["n"] - n0
                        y = y[0] if isinstance(y, (tuple, list)) else getattr(y, "sample", y)
                        ims[arm] = y.float().clamp(-1, 1)
                route["on"] = False
                mse = float(((ims["T"] - ims["M"]) ** 2).mean())
                sw["psnr_T_vs_M_db"] = round(10 * math.log10(4.0 / max(mse, 1e-20)), 2)
                sw["med"] = {k_: stt.median(v[1:]) for k_, v in sw["s"].items()}
                sw["speedup"] = round(sw["med"]["M"] / sw["med"]["T"], 4)
                sw["errors"] = route["err"][:5]
            except Exception as exc:  # noqa: BLE001
                sw["error"] = repr(exc)[:500]
            nn.Conv2d._conv_forward = o2
            rec["vae_swap"] = sw
            save()
        rec["end_vmhwm_gib"] = vm("VmHWM")
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
