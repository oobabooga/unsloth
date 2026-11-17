"""Per-call accuracy and per-shape race of Studio's INT8 attention on the real activations of a real render (eager
loads only: arm.py installs it when I8AUTO_PROBE is set, with compile off so every call reaches it).

Wraps ``diffusion_int8_attention.int8_attention_call``: every call still returns the kernel output. On a query
subsample of each call it records the rel L2 vs fp32 attention of the kernel and of the two native paths Studio runs
on ROCm (torch SDPA, and flash_attn's flash_attn_func where it imports), plus the share of heads whose V has an
attention sink (a few keys far from the rest in most channels, the case the CUDA kernel's sink path handles and the
Triton kernel does not). The first call at each distinct shape is also raced on its own inputs: int8 vs SDPA vs flash,
interleaved, CUDA events, median. One JSON line per call.

  I8AUTO_PROBE=/path/out.jsonl  I8AUTO_PROBE_ROWS=256  I8AUTO_RACE_ROUNDS=10
"""
from __future__ import annotations

import json
import os
import time

STATE = {"name": None, "call": 0, "dit_call": 0}
_RACED: set = set()


def _flash():
    try:
        from flash_attn import flash_attn_func  # noqa: PLC0415

        return flash_attn_func
    except Exception:  # noqa: BLE001
        return None


def _sdpa(q, k, v):
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def _rel(a, b) -> float:
    return float((a.float() - b).norm() / b.norm().clamp_min(1e-30))


def _sink_heads(vf) -> float:
    """vf [H, S, D] fp32: share of heads where >= half the channels have a key over 8 std from the channel mean
    and the range is over 2x what the middle 98% of keys need (a far-out chunk of keys sets V's range)."""
    import torch

    mu = vf.mean(1, keepdim = True)
    sd = vf.std(1, keepdim = True).clamp_min(1e-12)
    far = ((vf - mu).abs() / sd).amax(1) > 8.0  # [H, D]
    sub = vf[:, :: max(1, vf.shape[1] // 1024)]  # torch.quantile takes at most 2**24 elements
    lo = torch.quantile(sub, 0.01, dim = 1)
    hi = torch.quantile(sub, 0.99, dim = 1)
    ratio = (vf.amax(1) - vf.amin(1)) / (hi - lo).clamp_min(1e-12)
    clip = ((far & (ratio > 2.0)).float().mean(-1) >= 0.5)
    return float(clip.float().mean())


def _race(fns: dict, rounds: int) -> dict:
    import torch

    for fn in fns.values():
        fn()
        fn()
    torch.cuda.synchronize()
    times = {n: [] for n in fns}
    names = list(fns)
    for r in range(rounds):
        order = names[r % len(names):] + names[: r % len(names)]
        for n in order:
            a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            a.record()
            fns[n]()
            b.record()
            b.synchronize()
            times[n].append(a.elapsed_time(b))
    return {n: round(sorted(t)[len(t) // 2], 4) for n, t in times.items()}


def install(I8, path: str):
    import torch

    orig = I8.int8_attention_call
    rows_max = int(os.environ.get("I8AUTO_PROBE_ROWS", "256"))
    rounds = int(os.environ.get("I8AUTO_RACE_ROUNDS", "10"))
    flash = _flash()
    f = open(path, "a", buffering = 1, encoding = "utf-8")

    def probe(q, k, v, *rest):
        first = False
        key = (tuple(q.shape), tuple(k.shape), str(q.dtype))
        if not torch.compiler.is_compiling() and key not in _RACED:
            first = True
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        out = orig(q, k, v, *rest)
        if torch.compiler.is_compiling():
            return out
        STATE["call"] += 1
        rec = {"call": STATE["call"], "dit_call": STATE["dit_call"], "name": STATE["name"], "q": list(q.shape),
               "k": list(k.shape), "dtype": str(q.dtype), "int8_total": I8.STATS.get("int8", 0),
               "native_total": I8.STATS.get("native", 0)}
        try:
            if first:
                torch.cuda.synchronize()
                rec["first_call_ms"] = round(1000 * (time.perf_counter() - t0), 2)
            B, Sq, H, D = q.shape
            rows = torch.linspace(0, Sq - 1, min(rows_max, Sq), device = q.device).round().long()
            scale = D ** -0.5
            ker, nat, fl, sink = [], [], [], []
            for b in range(B):
                kf = k[b].float().transpose(0, 1)
                vf = v[b].float().transpose(0, 1)
                qr = q[b][rows].float().transpose(0, 1)
                ref = torch.softmax((qr @ kf.transpose(1, 2)) * scale, -1) @ vf
                ker.append(_rel(out[b][rows].transpose(0, 1), ref))
                qs = q[b : b + 1, rows]
                nat.append(_rel(_sdpa(qs, k[b : b + 1], v[b : b + 1])[0].transpose(0, 1), ref))
                if flash is not None:
                    try:
                        fl.append(_rel(flash(qs, k[b : b + 1], v[b : b + 1])[0].transpose(0, 1), ref))
                    except Exception as exc:  # noqa: BLE001
                        rec["flash_error"] = repr(exc)[:200]
                sink.append(_sink_heads(vf))
            rec["rel_kernel"] = round(max(ker), 6)
            rec["rel_sdpa"] = round(max(nat), 6)
            if fl:
                rec["rel_flash"] = round(max(fl), 6)
            rec["sink_heads"] = round(max(sink), 4)
            if first:
                _RACED.add(key)
                fns = {"int8": lambda: orig(q, k, v, *rest), "sdpa": lambda: _sdpa(q, k, v)}
                if flash is not None and "flash_error" not in rec:
                    fns["flash"] = lambda: flash(q, k, v)
                rec["race_ms"] = _race(fns, rounds)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = repr(exc)[:300]
        f.write(json.dumps(rec) + "\n")
        return out

    I8.int8_attention_call = probe
    return probe


def name_hooks(dits):
    handles = []
    for d in dits:
        def pre_dit(module, args, kwargs):
            STATE["dit_call"] += 1
        handles.append(d.register_forward_pre_hook(pre_dit, with_kwargs = True))
        for name, m in d.named_modules():
            if hasattr(m, "processor") and hasattr(m, "heads"):
                def pre(module, args, kwargs, _n = name):
                    STATE["name"] = _n
                handles.append(m.register_forward_pre_hook(pre, with_kwargs = True))
    return handles
