#!/usr/bin/env python3
"""Arm-process hooks for the nov28 AMD revisit (imported by rv28/arm.py; each is a no-op unless its env var is set).

ARM_KITCHEN_RACE=<pkg root>  C9-3: Studio's W8A8 GEMM launch (diffusion_native_w8a8_rdna._launch_mm) races
    comfy-kitchen #338 ``_C.int8_gemm`` per (N, K, M bucket) on the first call of each key: both kernels run on the real
    operands, kitchen is kept only when its output is torch.equal to Studio's AND its median time is lower; every other
    key keeps Studio's kernel. The race runs inside the warm-up renders (keys are M-bucketed so text-length changes do
    not re-race inside timed renders; a key first seen in a timed render is raced there and counted in late_races).
ARM_FP32_DENOISER=1  reference arm: every denoiser is cast to float32 after load (pipelines cast their inputs to the
    transformer's dtype), so every GEMM / norm / attention of the DiT runs in fp32.
"""
from __future__ import annotations

import os
import statistics as st
import sys

RACE = {"keys": {}, "kitchen_calls": 0, "studio_calls": 0, "errors": [], "late_races": 0, "phase": "warm",
        "aligned_clones": 0}


def _bucket(m: int) -> str:
    return "s" if m < 1024 else "l"


def install_kitchen_race(rec: dict) -> None:
    root = os.environ.get("ARM_KITCHEN_RACE") or ""
    if not root:
        return
    sys.path.insert(0, root)
    import torch
    from comfy_kitchen.backends import hip as KH

    from core.inference import diffusion_native_w8a8_rdna as R
    codes = getattr(KH, "DTYPE_TO_CODE", None)
    if codes is None:
        from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE as codes
    C = KH._C
    rec["kitchen_race"] = RACE
    RACE["kitchen_file"] = KH.__file__
    RACE["kitchen_has_wmma"] = bool(KH.has_wmma())
    studio = R._launch_mm

    def kit(xq, xs, wq, ws, bias, out, m, n, k):
        w = KH._aligned(wq)
        if w is not wq:
            RACE["aligned_clones"] += 1
        xs1 = xs.reshape(-1)
        ws1 = ws.reshape(-1)
        if xs1.dtype != torch.float32:
            xs1 = xs1.float()
        if ws1.dtype != torch.float32:
            ws1 = ws1.float()
        b = None if bias is None else KH._bias_operand(bias, n, xq.device)
        C.int8_gemm(KH._dl(xq), KH._dl(w), KH._dl(out), KH._dl(xs1.contiguous()), KH._dl(ws1.contiguous()),
                    0 if ws1.numel() == 1 else 1, None if b is None else KH._dl(b), m, n, k, codes[out.dtype],
                    KH._stream(xq))

    def timed(fn, reps = 5):
        ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            e0.record()
            fn()
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        return st.median(ts)

    def race(key, xq, xs, wq, ws, bias, out, m, n, k):
        row = {"m": m, "n": n, "k": k, "bias": bias is not None, "dtype": str(out.dtype), "phase": RACE["phase"]}
        if RACE["phase"] != "warm":
            RACE["late_races"] += 1
        o_s = torch.empty_like(out)
        o_k = torch.empty_like(out)
        try:
            studio(xq, xs, wq, ws, bias, o_s, m, n, k)
            kit(xq, xs, wq, ws, bias, o_k, m, n, k)
            torch.cuda.synchronize()
            row["equal"] = bool(torch.equal(o_s, o_k))
            ms_s, ms_k = [], []
            for r in range(3):
                pair = ((lambda: studio(xq, xs, wq, ws, bias, o_s, m, n, k)), (lambda: kit(xq, xs, wq, ws, bias, o_k, m, n, k)))
                if r % 2:
                    ms_k.append(timed(pair[1]))
                    ms_s.append(timed(pair[0]))
                else:
                    ms_s.append(timed(pair[0]))
                    ms_k.append(timed(pair[1]))
            row["studio_ms"], row["kitchen_ms"] = round(st.median(ms_s), 4), round(st.median(ms_k), 4)
            row["pick"] = "kitchen" if row["equal"] and row["kitchen_ms"] < row["studio_ms"] else "studio"
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)[:300]
            row["pick"] = "studio"
        RACE["keys"][key] = row
        return row["pick"]

    def launch(xq, xs, wq, ws, bias, out, m, n, k):
        key = f"{n}x{k}:{_bucket(m)}:{bias is not None}:{out.dtype}"
        row = RACE["keys"].get(key)
        pick = row["pick"] if row else race(key, xq, xs, wq, ws, bias, out, m, n, k)
        if pick == "kitchen":
            try:
                kit(xq, xs, wq, ws, bias, out, m, n, k)
                RACE["kitchen_calls"] += 1
                return
            except Exception as exc:  # noqa: BLE001
                RACE["errors"].append(repr(exc)[:200])
                RACE["keys"][key]["pick"] = "studio"
        RACE["studio_calls"] += 1
        studio(xq, xs, wq, ws, bias, out, m, n, k)

    R._launch_mm = launch
    RACE["installed"] = True


def fp32_denoiser(dits, rec: dict) -> None:
    if os.environ.get("ARM_FP32_DENOISER") != "1":
        return
    import torch
    done = []
    for d in dits:
        d.to(torch.float32)
        done.append({"cls": type(d).__name__,
                     "dtypes": sorted({str(p.dtype) for p in d.parameters()})})
    rec["fp32_denoiser"] = done


def end_counts(rec: dict) -> None:
    m = sys.modules.get("core.inference.diffusion_flux3_qk_rope")
    if m is not None:
        rec["flux3_qk"] = {"stats": dict(getattr(m, "STATS", {}) or {}),
                           "lift_diff": dict(getattr(m, "LIFT_DIFF", {}) or {}),
                           "verdicts": {str(k): v for k, v in (getattr(m, "_VERDICTS", {}) or {}).items()},
                           "failed": list(getattr(m, "_FAILED", []) or [])}
    m = sys.modules.get("core.inference.diffusion_native_w8a8_rdna")
    if m is not None:
        try:
            rec["w8a8_state"] = {"state": m.state(), "broken": m._STATE.get("broken")}
        except Exception as exc:  # noqa: BLE001
            rec["w8a8_state"] = repr(exc)[:200]
    m = sys.modules.get("core.inference.video_minimax_h3_vae")
    if m is not None:
        rec["h3_vae_module"] = True
