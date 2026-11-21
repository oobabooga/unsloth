#!/usr/bin/env python3
"""Probe (gfx1151, MIOpen): VAE upsample fold, nearest 2x + conv3x3(p1) vs one ConvTranspose2d(k4, s2, p1) with the taps
summed in fp32, on the FLUX.1 / Z-Image VAE upsample shapes (512 x 128^2, 512 x 256^2, 256 x 512^2) in bf16 and fp16.

Per state: per-shape ms (median of 5 interleaved rounds of 10 calls, CUDA/HIP events) for the 2x path and the fold, max
abs vs an fp64 reference for both, and, where the checkout has the fold (head), whether its default gate arms on this
device (it must not: MIOpen is unmeasured) and what ``UNSLOTH_VAE_UPSAMPLE_FOLD=1`` gives through ``_upsample``.
Observes only (criteria/vae_fold.py judges). JSON to --out, never stdout. Needs the GPU alone (timing)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

SHAPES = ((512, 128), (512, 256), (256, 512))
TAPS = {0: (2,), 1: (1, 2), 2: (0, 1), 3: (0,)}


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def _fold(w, torch):
    wf = w.float()
    cout, cin = wf.shape[:2]
    wt = wf.new_zeros(cin, cout, 4, 4)
    for a in range(4):
        for b in range(4):
            acc = wf.new_zeros(cout, cin)
            for i in TAPS[a]:
                for j in TAPS[b]:
                    acc += wf[:, :, i, j]
            wt[:, :, a, b] = acc.t()
    return wt.to(w.dtype).contiguous(memory_format = torch.channels_last)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F

    res: dict = {"state": args.state, "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                 "platform": sys.platform, "cuda": torch.cuda.is_available(), "shapes": {}}
    if torch.cuda.is_available():
        res["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    _write(args.out, res)
    if not torch.cuda.is_available():
        return 0
    dev = torch.device("cuda", 0)
    ev = lambda: torch.cuda.Event(enable_timing = True)

    def t_ms(fn):
        fn(); torch.cuda.synchronize()
        s, e = ev(), ev(); s.record()
        for _ in range(10):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / 10

    try:
        for dt in (torch.bfloat16, torch.float16):
            for c, hw in SHAPES:
                torch.manual_seed(0)
                conv = torch.nn.Conv2d(c, c, 3, padding = 1).to(dev, dt).to(memory_format = torch.channels_last)
                x = torch.randn(1, c, hw, hw, device = dev).to(dt).contiguous(memory_format = torch.channels_last)
                wt = _fold(conv.weight.detach(), torch)
                a = lambda: F.conv2d(F.interpolate(x, scale_factor = 2.0, mode = "nearest"), conv.weight, conv.bias, padding = 1)
                b = lambda: F.conv_transpose2d(x, wt, conv.bias, stride = 2, padding = 1)
                with torch.no_grad():
                    ya, yb = a().double(), b().double()
                    y64 = F.conv2d(F.interpolate(x.double(), scale_factor = 2.0, mode = "nearest"), conv.weight.double(), conv.bias.double(), padding = 1)
                    ta, tb = [], []
                    for _ in range(5):
                        ta.append(t_ms(a)); tb.append(t_ms(b))
                ta.sort(); tb.sort()
                res["shapes"][f"{str(dt)[6:]}_{c}x{hw}"] = {
                    "chain_ms": round(ta[2], 4), "fold_ms": round(tb[2], 4), "x": round(ta[2] / tb[2], 3),
                    "chain_spread": [round(ta[0], 4), round(ta[-1], 4)], "fold_spread": [round(tb[0], 4), round(tb[-1], 4)],
                    "chain_err": float((ya - y64).abs().max()), "fold_err": float((yb - y64).abs().max()),
                    "finite": bool(torch.isfinite(yb).all())}
                _write(args.out, res)
    except Exception as exc:  # noqa: BLE001
        res["error"] = repr(exc)[:1500]
        res["tb"] = traceback.format_exc()[-3000:]
    try:
        sys.path.insert(0, str(Path(args.checkout) / "studio" / "backend"))
        from core.inference import diffusion_vae_strips as st
        res["has_fold"] = hasattr(st, "_fold_on")
        if res["has_fold"]:
            x = torch.randn(1, 64, 16, 16, device = dev).to(torch.bfloat16).contiguous(memory_format = torch.channels_last)
            os.environ.pop(st.UPSAMPLE_FOLD_ENV, None)
            res["fold_default_on"] = bool(st._fold_on(x))
            conv = torch.nn.Conv2d(64, 32, 3, padding = 1).to(dev, torch.bfloat16)
            up = type("Upsample2D", (), {"conv": conv})()
            with torch.no_grad():
                base, _ = st._upsample(up, x, False)
                os.environ[st.UPSAMPLE_FOLD_ENV] = "1"
                forced, _ = st._upsample(up, x, False)
                os.environ.pop(st.UPSAMPLE_FOLD_ENV, None)
            res["forced_max_abs_vs_2x"] = float((forced.float() - base.float()).abs().max())
    except Exception as exc:  # noqa: BLE001
        res["tree_error"] = repr(exc)[:1500]
    _write(args.out, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
