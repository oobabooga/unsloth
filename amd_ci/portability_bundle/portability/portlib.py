"""Shared helpers for the portability harness: device facts, timing, error stats, engagement probes.

A case is ``fn(ctx) -> list[dict]``; build each dict with ``ctx.row(...)`` so every device reports the same columns.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
import time
import traceback

RUNS, FALLBACK, FAILS, SKIPPED, REFUSES, WRONG = "runs", "fallback", "fails", "skipped", "refuses", "wrong"


def device_facts() -> dict:
    import torch

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "os": sys.platform,
        "torch": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "hip": getattr(torch.version, "hip", None),
        "gpu_available": bool(torch.cuda.is_available()),
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        info["triton"] = None
        info["triton_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    try:
        import torchao

        info["torchao"] = torchao.__version__
    except Exception as exc:  # noqa: BLE001
        info["torchao"] = None
        info["torchao_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["arch"] = getattr(p, "gcnArchName", None) or "sm_%d%d" % tuple(info["capability"])
        info["total_mem_gb"] = round(p.total_memory / 2**30, 1)
        info["sm_count"] = p.multi_processor_count
        try:
            info["cudnn"] = torch.backends.cudnn.version()
        except Exception:  # noqa: BLE001
            info["cudnn"] = None
    return info


def device_tag(info: dict) -> str:
    name = (info.get("name") or "").lower()
    if info.get("hip"):
        arch = (info.get("arch") or "rocm").split(":")[0]
        return f"{arch}-{'windows' if info.get('os') == 'win32' else 'linux'}"
    for key, tag in (("t4", "t4"), ("l4", "l4"), ("a100", "a100"), ("h100", "h100"), ("b200", "b200"),
                     ("rtx pro 6000", "g4"), ("6000 blackwell", "g4"), ("p100", "p100"), ("l40", "l40")):
        if key in name:
            return tag
    if not info.get("gpu_available"):
        return "cpu"
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_") or "gpu"


def base_dtype_for(info: dict):
    """What Studio computes in on this device: float16 below Ampere on NVIDIA (T4), bfloat16 elsewhere."""
    import torch

    if info.get("hip"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    cap = tuple(info.get("capability") or (0, 0))
    return torch.bfloat16 if cap >= (8, 0) else torch.float16


def dtype_name(dt) -> str:
    return {"torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32", "torch.float64": "fp64",
            "torch.int8": "int8"}.get(str(dt), str(dt))


def err_stats(out, ref) -> dict:
    """max abs, mean abs and relative RMS of ``out`` against ``ref`` (both cast to float64 in chunks)."""
    import torch

    a = out.reshape(-1)
    r = ref.reshape(-1)
    n = a.numel()
    mx = 0.0
    sabs = 0.0
    sse = 0.0
    sref = 0.0
    bad = 0
    step = 1 << 24
    for s in range(0, n, step):
        x = a[s: s + step].double()
        y = r[s: s + step].double()
        d = x - y
        nf = ~torch.isfinite(d)
        if nf.any():
            bad += int(nf.sum())
            d = torch.where(nf, torch.zeros_like(d), d)
        mx = max(mx, float(d.abs().max())) if d.numel() else mx
        sabs += float(d.abs().sum())
        sse += float((d * d).sum())
        sref += float((y * y).sum())
    res = {"max_abs": mx, "mean_abs": sabs / max(n, 1), "rel_rms": math.sqrt(sse / max(sref, 1e-300))}
    if bad:
        res["nonfinite"] = bad
    return res


def bench(fn, iters: int = 20, warmup: int = 3) -> float:
    """Median milliseconds of ``fn()`` over ``iters`` CUDA-event-timed runs."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing = True)
        e = torch.cuda.Event(enable_timing = True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def profile_names(fn) -> tuple[set, set]:
    """(aten op names, device kernel names) seen while running ``fn()`` once."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    with profile(activities = acts) as prof:
        fn()
        torch.cuda.synchronize()
    ops, kernels = set(), set()
    for ev in prof.events():
        name = ev.name
        dev = str(getattr(ev, "device_type", ""))
        if "CUDA" in dev or "HIP" in dev:
            kernels.add(name)
        else:
            ops.add(name)
    return ops, kernels


def compiled_code(fn, *args):
    """(output, generated source) of an already ``torch.compile``d fn on its first call."""
    from torch._inductor.utils import run_and_get_code

    out, codes = run_and_get_code(fn, *args)
    return out, "\n\n# ==== next graph ====\n\n".join(codes)


class Ctx:
    def __init__(self, case: str, out_dir: str, quick: bool, iters: int, bundle: str | None, info: dict | None = None):
        import torch

        self.case = case
        self.out_dir = out_dir
        self.quick = quick
        self.iters = iters
        self.bundle = bundle
        self.info = info or device_facts()
        self.tag = device_tag(self.info)
        self.is_rocm = bool(self.info.get("hip"))
        self.cap = tuple(self.info.get("capability") or (0, 0))
        self.base_dtype = base_dtype_for(self.info) if torch.cuda.is_available() else torch.float32
        self.base = dtype_name(self.base_dtype)
        self.dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        self.rows: list[dict] = []
        self.t0 = time.time()
        os.makedirs(self.log_dir, exist_ok = True)

    @property
    def log_dir(self) -> str:
        return os.path.join(self.out_dir, "logs", self.case)

    def log(self, *a):
        print(f"[{self.case} {time.time() - self.t0:7.1f}s]", *a, flush = True)

    def bench(self, fn, iters: int | None = None, warmup: int = 3) -> float:
        return bench(fn, iters or self.iters, warmup)

    def err(self, out, ref) -> dict:
        return err_stats(out, ref)

    def bundle_path(self, *parts) -> str | None:
        if not self.bundle:
            return None
        p = os.path.join(self.bundle, *parts)
        return p if os.path.exists(p) else None

    def row(self, variant: str, *, shape: str = "", dtype: str | None = None, status: str = RUNS,
            engaged=None, err: dict | None = None, ms: float | None = None, eager_ms: float | None = None,
            compiled_ms: float | None = None, note: str = "", **extra) -> dict:
        r = {"case": self.case, "variant": variant, "shape": shape, "dtype": dtype or self.base, "status": status,
             "engaged": engaged, "err": err, "ms": ms, "base_eager_ms": eager_ms, "base_compiled_ms": compiled_ms,
             "speedup_vs_eager": (eager_ms / ms) if (ms and eager_ms) else None,
             "speedup_vs_compiled": (compiled_ms / ms) if (ms and compiled_ms) else None, "note": note}
        r.update(extra)
        self.rows.append(r)
        e = err or {}
        self.log(f"{variant:<22} {shape:<34} {r['dtype']:<5} {status:<8} eng={engaged} "
                 f"max={_f(e.get('max_abs'))} rel={_f(e.get('rel_rms'))} ms={_f(ms)} "
                 f"x_eager={_f(r['speedup_vs_eager'])} x_comp={_f(r['speedup_vs_compiled'])} {note[:140]}")
        return r

    def fail(self, variant: str, exc: BaseException, **kw) -> dict:
        tb = traceback.format_exc()
        with open(os.path.join(self.log_dir, f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', variant)}.err.txt"), "a") as f:
            f.write(tb + "\n")
        return self.row(variant, status = FAILS, note = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}", **kw)

    def attempt(self, variant: str, fn, **kw):
        """Run ``fn()``; on exception record a FAILS row and return None."""
        import torch

        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            self.fail(variant, exc, **kw)
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            return None


def _f(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v == 0:
            return "0"
        if abs(v) >= 1000 or abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.3g}"
    return str(v)


def dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(obj, f, indent = 1, default = str)
