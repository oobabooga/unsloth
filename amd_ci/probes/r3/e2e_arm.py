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
import contextlib
import ctypes
import functools
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
                trace.append({"t": round(time.perf_counter() - T_PERF, 3), "fn": label,
                              "file": "/".join(path.replace("\\", "/").split("/")[-2:]),
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
T_PERF = time.perf_counter()


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

        backend_calls: list = []
        rec["backend_calls"] = backend_calls

        def backend_for(*x, **k):
            r = orig_backend_for(*x, **k)
            try:
                f = str(x[0] if x else k.get("filename")).replace("\\", "/").split("/")
                backend_calls.append({"file": "/".join(f[-2:]), "diffusers": bool(k.get("diffusers", x[1] if len(x) > 1 else False)),
                                      "backend": r})
            except Exception:  # noqa: BLE001
                pass
            if r == "pread":
                pread_calls["n"] += 1
            return r
        SP._backend_for = backend_for
        gate_decisions: list = []
        rec["gate_decisions"] = gate_decisions
        if hasattr(SP, "_PressureGate"):
            orig_decide = SP._PressureGate.decide

            def decide(self, shard):
                first = self.pread is None
                r = orig_decide(self, shard)
                if first:
                    gate_decisions.append({"shard": "/".join(str(shard).replace("\\", "/").split("/")[-2:]),
                                           "pread": bool(r), "decision": getattr(SP, "last_reader_decision", None)})
                return r
            SP._PressureGate.decide = decide
        # pread time split: total inside Studio's diffusers reader, and inside its page drops
        split = {"load_file_s": 0.0, "load_file_n": 0, "drop_s": 0.0, "willneed_s": 0.0}
        rec["pread_split"] = split
        if hasattr(SP, "_with_pread_load_file"):
            orig_wplf = SP._with_pread_load_file

            def wplf(load_file, safe_open):
                inner = orig_wplf(load_file, safe_open)

                @functools.wraps(inner)
                def timed(*x, **k):
                    t = time.perf_counter()
                    try:
                        return inner(*x, **k)
                    finally:
                        split["load_file_s"] += time.perf_counter() - t
                        split["load_file_n"] += 1
                timed._unsloth_pread = True
                timed.__wrapped__ = getattr(inner, "__wrapped__", load_file)  # keeps Studio's restore check
                return timed
            SP._with_pread_load_file = wplf
        if hasattr(SP, "_page_dropper"):
            orig_pd = SP._page_dropper
            willneed = os.environ.get("R3_FADV_WILLNEED") == "1"

            @contextlib.contextmanager
            def pd(filename):
                if willneed:
                    t = time.perf_counter()
                    try:
                        fd = os.open(filename, os.O_RDONLY)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
                        os.close(fd)
                    except OSError:
                        pass
                    split["willneed_s"] += time.perf_counter() - t
                with orig_pd(filename) as drop:
                    def timed_drop(key):
                        t = time.perf_counter()
                        drop(key)
                        split["drop_s"] += time.perf_counter() - t
                    yield timed_drop
            SP._page_dropper = pd
        rec["fadv_willneed"] = os.environ.get("R3_FADV_WILLNEED") == "1"
        trace: list = []
        rec["trace"] = trace
        if a.trace:
            install_trace(trace)
        rec["pread_wanted"] = SP.pread_wanted()
        rec["rocm_integrated"] = getattr(SP, "rocm_integrated", lambda: None)()
        from core.inference.diffusion import DiffusionBackend
        eng = DiffusionBackend()
        rec["mem_available_before_load_gib"] = meminfo("MemAvailable")
        rec["mem_total_gib"] = meminfo("MemTotal")
        rec["swap_free_gib"] = meminfo("SwapFree")
        rec["reader_env"] = os.environ.get("UNSLOTH_SAFETENSORS_PREAD")
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
        rec["load_t0_rel"] = round(t0 - T_PERF, 3)
        rec["load_started_at"] = time.time() - (time.perf_counter() - t0)
        rec["load_vmhwm_gib"] = vm("VmHWM")
        rec["cached_after_load_gib"] = meminfo("Cached")
        rec["load_rss_gib"] = vm("VmRSS")
        rec["pread_shards"] = pread_calls["n"]
        rec["reader_decision"] = getattr(SP, "last_reader_decision", "absent")
        rec["mem_available_after_load_gib"] = meminfo("MemAvailable")
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
        try:
            vdec = pipe.vae.decode

            def decode(*x, **k):
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
            t = time.perf_counter()
            res = eng.generate(prompt = PROMPT, seed = SEED, **gen)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
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
