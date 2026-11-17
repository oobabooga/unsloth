"""Per-call device timing + torch.profiler capture of the denoiser, text encoder and VAE decode, for Studio and the
baseline alike (generalised from q21_prof_hook).

Env:
  PROF_OUT     directory for calls.jsonl, prof_<label>_<render>_<i>.json (+ .trace.json) (required to engage)
  PROF_STEP    0-based denoiser call within the profiled render to capture (default 5)
  PROF_RENDER  render label whose calls are profiled (default "prof": the extra render prof_cell adds after the
               timed protocol, so profiling never inflates a timed number)
  PROF_TAG     label written into every record

The client (prof_cell.py) writes the current render label into PROF_OUT/current_render before every render; the
hook reads it per call (the baseline runs the hook inside its server process, so a file is the shared channel).
Every call is timed with device events (no host sync added) and tagged with that render label. In the profiled
render, denoiser call PROF_STEP and the first text-encoder / VAE call run under torch.profiler.
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
_STATE = {"pending": [], "count": collections.Counter(), "render": None, "in_render": collections.Counter(),
          "depth": 0}


def enabled() -> bool:
    return bool(os.environ.get("PROF_OUT"))


def _out() -> Path:
    p = Path(os.environ["PROF_OUT"])
    p.mkdir(parents = True, exist_ok = True)
    return p


def _current_render() -> str:
    try:
        return (_out() / "current_render").read_text(encoding = "utf-8").strip()
    except OSError:
        return ""


def _prof_step() -> int:
    """PROF_OUT/prof_step (written by the client per cell: clamped to the cell's steps) wins over $PROF_STEP."""
    try:
        return int((_out() / "prof_step").read_text(encoding = "utf-8").strip())
    except (OSError, ValueError):
        return int(os.environ.get("PROF_STEP", "5"))


def _write_line(rec: dict) -> None:
    with open(_out() / "calls.jsonl", "a", encoding = "utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _drain(force: bool = False) -> None:
    keep = []
    for rec, start, end in _STATE["pending"]:
        if force or end.query():
            end.synchronize()
            rec["gpu_ms"] = round(start.elapsed_time(end), 3)
            _write_line(rec)
        else:
            keep.append((rec, start, end))
    _STATE["pending"] = keep


CATS = [
    ("attention", re.compile(r"attn|fmha|flash|aotriton|attention|softmax", re.I)),
    ("gemm", re.compile(r"Cijk_|gemm|gemv|hipblaslt|rocblas|matmul|_mm_|wmma|tensile|bmm|addmm|\bmm\b|triton_tem|_mm\b|"
                        r"mm_plus|int_mm|scaled_mm|igemm|i8", re.I)),
    ("dequant", re.compile(r"dequant|gguf|q4_k|q8_0|unpack|nvfp4|fp4|fp8|e4m3", re.I)),
    ("conv", re.compile(r"conv|igemm|winograd|im2col|naive_conv|MIOpen", re.I)),
    ("triton_fused", re.compile(r"^triton_", re.I)),
    ("norm", re.compile(r"norm|rms|adaln", re.I)),
    ("copy_cast", re.compile(r"copy|memcpy|memset|cat_|CatArray|to_copy|contiguous|transpose", re.I)),
    ("reduce", re.compile(r"reduce", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled|pointwise", re.I)),
]


def category(name: str) -> str:
    if name.startswith("## Call CompiledFxGraph"):
        return "wrapper"  # an Inductor graph's host range; its kernels are counted on their own
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


def _summarise(prof, label: str, index: int, wall_ms: float) -> dict:
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
            if e.name.startswith("## Call CompiledFxGraph"):
                continue
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
    gaps = []
    if iv:
        srt = sorted(iv)
        cur_e = srt[0][1]
        for s, e in srt[1:]:
            if s - cur_e > 200.0:  # > 0.2 ms with the device idle: a host-side stall
                gaps.append(round(s - cur_e, 1))
            cur_e = max(cur_e, e)
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
    gaps.sort(reverse = True)
    return {"tag": os.environ.get("PROF_TAG"), "label": label, "call": index, "wall_ms": round(wall_ms, 3),
            "kernel_sum_us": round(sum(r["us"] for r in top), 1), "gpu_span_us": round(span, 1),
            "gpu_busy_us": round(busy, 1), "gpu_idle_us": round(span - busy, 1), "n_kernels": sum(r["count"] for r in top),
            "idle_gaps_over_0p2ms": {"n": len(gaps), "sum_us": round(sum(gaps), 1), "largest_us": gaps[:10]},
            "categories_us": {k: round(v, 1) for k, v in sorted(cats.items(), key = lambda kv: -kv[1])},
            "host_sync_like": {k: [host[k], round(host_us[k], 1)] for k in host}, "kernels": top[:300], "ops": ops[:200]}


def wrap(fn, label: str):
    import torch

    def wrapped(*args, **kwargs):
        if not enabled() or _STATE["depth"]:
            return fn(*args, **kwargs)
        render = _current_render()
        with _LOCK:
            if render != _STATE["render"]:
                _STATE["render"] = render
                _STATE["in_render"] = collections.Counter()
            index = _STATE["count"][label]
            _STATE["count"][label] += 1
            step = _STATE["in_render"][label]
            _STATE["in_render"][label] += 1
        rec = {"tag": os.environ.get("PROF_TAG"), "label": label, "call": index, "render": render, "i": step,
               "t": round(time.time(), 3)}
        try:  # this process's allocator before the call (phase attribution of the device peak)
            rec["alloc_mib"] = int(torch.cuda.memory_allocated()) >> 20
            rec["reserved_mib"] = int(torch.cuda.memory_reserved()) >> 20
        except Exception:  # noqa: BLE001
            pass
        try:
            x = next(a for a in list(args) + list(kwargs.values()) if hasattr(a, "shape"))
            rec["x_shape"] = list(x.shape)
        except Exception:  # noqa: BLE001
            pass
        want = os.environ.get("PROF_RENDER", "prof")
        is_den = label.endswith("denoiser")
        target = render == want and step == (_prof_step() if is_den else 0)
        _STATE["depth"] += 1
        try:
            if target:
                from torch.profiler import ProfilerActivity, profile

                _drain(force = True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes = True) as prof:
                    out = fn(*args, **kwargs)
                    torch.cuda.synchronize()
                wall = (time.perf_counter() - t0) * 1000.0
                name = f"prof_{label}_{render}_{step}"
                try:
                    summary = _summarise(prof, label, index, wall)
                    (_out() / f"{name}.json").write_text(json.dumps(summary, indent = 1), encoding = "utf-8")
                    prof.export_chrome_trace(str(_out() / f"{name}.trace.json"))
                except Exception as exc:  # noqa: BLE001
                    (_out() / f"{name}.error.txt").write_text(repr(exc), encoding = "utf-8")
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
            if not is_den or step == 0:  # peak inside the TE / VAE call and the first step (needs a sync: off the timed path)
                try:
                    rec["t_end"] = round(time.time(), 3)
                    rec["alloc_after_mib"] = int(torch.cuda.memory_allocated()) >> 20
                    rec["reserved_after_mib"] = int(torch.cuda.memory_reserved()) >> 20
                except Exception:  # noqa: BLE001
                    pass
            _STATE["pending"].append((rec, start, end))
            _drain()
            return out
        finally:
            _STATE["depth"] -= 1

    wrapped.__prof_hook__ = True
    _keep_signature(wrapped, fn)
    return wrapped


def _keep_signature(wrapped, src) -> None:
    """Pipelines read inspect.signature(transformer.forward) to pick the kwargs they pass (txt_ids, ...): the wrapper
    must report the wrapped callable's signature, never (*args, **kwargs)."""
    import inspect

    try:
        wrapped.__signature__ = inspect.signature(src)
    except (TypeError, ValueError):
        pass
    try:
        wrapped.__wrapped__ = src
        wrapped.__doc__ = getattr(src, "__doc__", None)
        wrapped.__name__ = getattr(src, "__name__", "wrapped")
    except (AttributeError, TypeError):
        pass


def attach_instance(module, attr: str, label: str) -> bool:
    """Wrap one instance's method (instance attribute). An instance-level forward already set (an offload hook) is
    wrapped as it is; otherwise the class method is resolved per call, so a later class-level patch still runs."""
    prev = module.__dict__.get(attr)
    if getattr(prev, "__prof_hook__", False):
        return False
    if prev is not None:
        fn = prev
        sig_src = prev
    else:
        sig_src = getattr(module, attr)  # the bound class method: its signature without self

        def fn(*args, **kwargs):
            return getattr(type(module), attr)(module, *args, **kwargs)
    w = wrap(fn, label)
    _keep_signature(w, sig_src)
    setattr(module, attr, w)
    return True


def flush() -> None:
    try:
        _drain(force = True)
    except Exception:  # noqa: BLE001
        pass
