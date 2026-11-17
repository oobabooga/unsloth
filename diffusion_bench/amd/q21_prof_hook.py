"""Per-call timing + torch.profiler capture of chosen Qwen-Image-2.1 denoiser calls, for Studio and ComfyUI alike.

Env:
  Q21_PROF_OUT    directory for calls.jsonl, prof_<i>.json, prof_<i>.trace.json.gz (required to engage)
  Q21_PROF_CALLS  comma list of 0-based call indexes to profile (default none)
  Q21_PROF_TAG    label written into every record

``wrap(fn)`` returns a function that times every call with device events (no host sync added) and runs the listed
calls under torch.profiler (synchronised before and after, so the capture holds exactly that call's kernels).
"""

from __future__ import annotations

import collections
import json
import os
import re
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_STATE = {"n": 0, "pending": []}


def enabled() -> bool:
    return bool(os.environ.get("Q21_PROF_OUT"))


def _out() -> Path:
    p = Path(os.environ["Q21_PROF_OUT"])
    p.mkdir(parents = True, exist_ok = True)
    return p


def _targets() -> set:
    return {int(x) for x in (os.environ.get("Q21_PROF_CALLS") or "").split(",") if x.strip().isdigit()}


def _write_line(rec: dict) -> None:
    with open(_out() / "calls.jsonl", "a", encoding = "utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _drain(force: bool = False) -> None:
    import torch

    keep = []
    for rec, start, end in _STATE["pending"]:
        if force or end.query():
            end.synchronize()
            rec["gpu_ms"] = round(start.elapsed_time(end), 3)
            _write_line(rec)
        else:
            keep.append((rec, start, end))
    _STATE["pending"] = keep
    del torch


CATS = [
    ("attention", re.compile(r"attn|fmha|flash|aotriton|attention|softmax", re.I)),
    ("gemm", re.compile(r"Cijk_|gemm|gemv|hipblaslt|rocblas|matmul|_mm_|wmma|tensile|bmm|addmm|\bmm\b|triton_tem|_mm\b|mm_plus", re.I)),
    ("triton_fused", re.compile(r"^triton_", re.I)),
    ("norm", re.compile(r"norm|rms", re.I)),
    ("copy_cast", re.compile(r"copy|memcpy|memset|cat_|CatArray|to_copy|contiguous|transpose", re.I)),
    ("reduce", re.compile(r"reduce", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled|pointwise", re.I)),
]


def category(name: str) -> str:
    for cat, rx in CATS:
        if rx.search(name):
            return cat
    return "other"


def _intervals_union(iv: list) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _summarise(prof, index: int, wall_ms: float) -> dict:
    from torch.autograd import DeviceType

    kernels = collections.defaultdict(lambda: [0, 0.0])
    cats = collections.defaultdict(float)
    iv = []
    host = collections.Counter()
    host_us = collections.defaultdict(float)
    for e in prof.events():
        dt = getattr(e, "device_type", None)
        tr = e.time_range
        if dt == DeviceType.CUDA:
            us = tr.end - tr.start
            k = kernels[e.name]
            k[0] += 1
            k[1] += us
            cats[category(e.name)] += us
            iv.append((tr.start, tr.end))
        else:
            n = e.name
            if re.search(r"Synchronize|Memcpy|_local_scalar_dense|aten::item|nonzero|EventSynchronize|StreamWait", n):
                host[n] += 1
                host_us[n] += tr.end - tr.start
    span = (max(e for _, e in iv) - min(s for s, _ in iv)) if iv else 0.0
    busy = _intervals_union(iv)
    ops = []
    try:
        for a in prof.key_averages(group_by_input_shape = True):
            dev = getattr(a, "self_device_time_total", None)
            if dev is None:
                dev = getattr(a, "self_cuda_time_total", 0.0)
            if dev and dev > 0:
                ops.append({"op": a.key, "shapes": str(getattr(a, "input_shapes", ""))[:300], "count": a.count,
                            "self_dev_us": round(float(dev), 1)})
    except Exception as exc:  # noqa: BLE001
        ops.append({"error": str(exc)[:300]})
    ops.sort(key = lambda r: -r.get("self_dev_us", 0))
    top = sorted(({"name": n[:240], "count": c, "us": round(us, 1), "cat": category(n)} for n, (c, us) in
                  kernels.items()), key = lambda r: -r["us"])
    return {"tag": os.environ.get("Q21_PROF_TAG"), "call": index, "wall_ms": round(wall_ms, 3),
            "kernel_sum_us": round(sum(r["us"] for r in top), 1), "gpu_span_us": round(span, 1),
            "gpu_busy_us": round(busy, 1), "gpu_idle_us": round(span - busy, 1), "n_kernels": sum(r["count"] for r in top),
            "categories_us": {k: round(v, 1) for k, v in sorted(cats.items(), key = lambda kv: -kv[1])},
            "host_sync_like": {k: [host[k], round(host_us[k], 1)] for k in host}, "kernels": top[:400], "ops": ops[:300]}


def wrap(fn, label: str = ""):
    import torch

    def wrapped(*args, **kwargs):
        if not enabled():
            return fn(*args, **kwargs)
        with _LOCK:
            index = _STATE["n"]
            _STATE["n"] += 1
        rec = {"tag": os.environ.get("Q21_PROF_TAG"), "label": label, "call": index, "t": round(time.time(), 3)}
        try:
            x = next(a for a in list(args) + list(kwargs.values()) if hasattr(a, "shape"))
            rec["x_shape"] = list(x.shape)
        except Exception:  # noqa: BLE001
            pass
        if index in _targets():
            from torch.profiler import ProfilerActivity, profile

            _drain(force = True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes = True) as prof:
                out = fn(*args, **kwargs)
                torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000.0
            try:
                summary = _summarise(prof, index, wall)
                (_out() / f"prof_{index}.json").write_text(json.dumps(summary, indent = 1), encoding = "utf-8")
                prof.export_chrome_trace(str(_out() / f"prof_{index}.trace.json"))
            except Exception as exc:  # noqa: BLE001
                (_out() / f"prof_{index}.error.txt").write_text(repr(exc), encoding = "utf-8")
            rec["profiled"] = True
            rec["wall_ms"] = round(wall, 3)
            _write_line(rec)
            return out
        start = torch.cuda.Event(enable_timing = True)
        end = torch.cuda.Event(enable_timing = True)
        t0 = time.perf_counter()
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        rec["host_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        _STATE["pending"].append((rec, start, end))
        _drain()
        return out

    wrapped.__q21_prof__ = True
    return wrapped


def attach_instance(module, label: str = "") -> bool:
    """Wrap one module instance's forward (instance attribute; the class and its source digest stay untouched)."""
    if getattr(module.__dict__.get("forward"), "__q21_prof__", False):
        return False

    def current(*args, **kwargs):
        # resolved per call: a class-level forward patch installed later (deferred compile) still runs
        return type(module).forward(module, *args, **kwargs)

    module.forward = wrap(current, label)
    return True


def flush() -> None:
    try:
        _drain(force = True)
    except Exception:  # noqa: BLE001
        pass
