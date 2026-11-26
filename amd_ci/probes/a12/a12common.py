"""Shared helpers for the A1/A2 gfx1151 child scripts (one arm per process; JSON to --out)."""
from __future__ import annotations

import hashlib
import json
import os
import statistics as st


def write(path, obj):
    with open(path, "w", encoding = "utf-8") as fh:
        fh.write(json.dumps(obj, default = str))


def env_info():
    import torch
    info = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None)}
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        info["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    except Exception as exc:  # noqa: BLE001
        info["arch_error"] = repr(exc)
    info["aotriton_experimental"] = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
    try:
        import triton
        amd = getattr(getattr(triton, "knobs", None), "amd", None)
        info["triton_amd_use_buffer_ops"] = getattr(amd, "use_buffer_ops", "n/a") if amd is not None else "n/a"
    except Exception as exc:  # noqa: BLE001
        info["triton_amd_use_buffer_ops"] = repr(exc)[:100]
    return info


def sha(t):
    return hashlib.sha256(t.contiguous().view(-1).view(__import__("torch").uint8).cpu().numpy().tobytes()).hexdigest()[:20]


def time_calls(fn, reps, flush = None, warm = 2):
    """Median ms of ``reps`` single calls (CUDA events), optional L2/LLC flush before each."""
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ms.append(a.elapsed_time(b))
    return {"ms": ms, "med_ms": st.median(ms)}


def time_round(fn, n_calls):
    import torch
    a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    a.record()
    for _ in range(n_calls):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / n_calls


def interleave(fns: dict, rounds: int, target_ms = 30.0):
    """name -> fn; ABBA-interleaved rounds of n calls each; returns name -> {ms: [...], med: x}."""
    names = list(fns)
    for f in fns.values():
        f()
    import torch
    torch.cuda.synchronize()
    est = time_round(fns[names[0]], 3)
    n = int(max(3, min(50, target_ms / max(est, 1e-3))))
    res = {k: {"ms": []} for k in names}
    for r in range(rounds):
        for k in (names if r % 2 == 0 else names[::-1]):
            res[k]["ms"].append(time_round(fns[k], n))
    for k in names:
        res[k]["med"] = st.median(res[k]["ms"])
    res["_calls_per_round"] = n
    return res
