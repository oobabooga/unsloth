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
            rec["cache_files"], rec["cache_gb"] = len(files), round(sum(os.path.getsize(f) for f in files) / 1e9, 2)
            rec["resident_before"] = drop_cache(files)
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

        def backend_for(filename):
            r = orig_backend_for(filename)
            if r == "pread":
                pread_calls["n"] += 1
            return r
        SP._backend_for = backend_for
        rec["pread_wanted"] = SP.pread_wanted()
        rec["rocm_integrated"] = getattr(SP, "rocm_integrated", lambda: None)()
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
        rec["load_rss_gib"] = vm("VmRSS")
        rec["pread_shards"] = pread_calls["n"]
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
        gen = json.loads(a.gen)
        rec["params"] = gen
        save()
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
