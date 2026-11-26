#!/usr/bin/env python3
"""Smoke on the real card (minutes): (1) weight pitch K + 64: F.linear with the padded weight is torch.equal to the
contiguous one for DiT shapes (Wan 5B, Krea-2, Qwen-2.1, klein), plus Studio's pad_weights engagement on a toy DiT;
(2) the vendored Triton conv3d router: a few Wan decoder conv shapes, routed vs stock conv3d vs fp32, and the
counters (routed > 0, no error fallback). Writes JSON to --out only.

  python smoke_gpu.py --backend <studio/backend> --out smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

PITCH_SHAPES = [  # (N, K, M)
    (3072, 3072, 4096), (14336, 3072, 4096), (3072, 14336, 4096),  # Wan 5B
    (6144, 6144, 4608), (16384, 6144, 4608), (6144, 16384, 4608),  # Krea-2
    (3072, 3072, 4176), (12288, 3072, 4176), (9216, 3072, 4096),  # klein / Qwen-2.1
]
CONV_SHAPES = [  # (C_in, C_out, k, T, H, W) at a Wan 2.2 decoder scale, bf16
    (256, 256, 3, 4, 64, 64), (128, 128, 3, 4, 128, 128), (512, 512, 1, 2, 32, 32), (96, 96, 3, 3, 96, 96),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--out", required = True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.backend).resolve()))
    obs: dict = {}
    out = Path(a.out)

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    import torch
    import torch.nn.functional as F
    obs["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
    dev = "cuda"
    g = torch.Generator(device = dev).manual_seed(0)
    # 1. pitch exactness per GEMM
    rows = []
    for n, k, m in PITCH_SHAPES:
        w = torch.randn(n, k, device = dev, dtype = torch.bfloat16, generator = g) * 0.02
        x = torch.randn(m, k, device = dev, dtype = torch.bfloat16, generator = g)
        buf = torch.empty(n, k + 64, device = dev, dtype = torch.bfloat16)
        buf[:, :k].copy_(w)
        wp = buf[:, :k]
        y0, y1 = F.linear(x, w), F.linear(x, wp)
        times = {}
        for name, ww in (("contig", w), ("pitch", wp), ("contig2", w), ("pitch2", wp)):
            for _ in range(3):
                F.linear(x, ww)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                F.linear(x, ww)
            torch.cuda.synchronize()
            times[name] = round((time.perf_counter() - t) / 10 * 1e3, 3)
        rows.append({"n": n, "k": k, "m": m, "equal": bool(torch.equal(y0, y1)), "ms": times})
        del w, x, buf, wp, y0, y1
    obs["pitch"] = rows
    obs["pitch_all_equal"] = all(r["equal"] for r in rows)
    try:
        from core.inference import diffusion_rocm_weight_pitch as wp_mod

        dit = torch.nn.Sequential(torch.nn.Linear(3072, 3072), torch.nn.Linear(3072, 14336)).to(dev, torch.bfloat16)
        x = torch.randn(256, 3072, device = dev, dtype = torch.bfloat16)
        ref = dit(x)
        import os
        os.environ[wp_mod.PITCH_ENV] = "64"
        armed = wp_mod.install(dit, family = "wan2.2")
        y = dit(x)
        obs["pitch_install"] = {"armed": armed, "stats": dict(wp_mod.STATS), "equal": bool(torch.equal(y, ref)),
                                "strides": [tuple(m.weight.stride()) for m in dit]}
    except Exception:  # noqa: BLE001
        obs["pitch_install_error"] = traceback.format_exc()[-1500:]
    save()
    # 2. vendored conv3d router
    try:
        import os
        os.environ["UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D"] = "1"
        from core.inference import diffusion_rocm_triton_conv3d as tc

        crow = []
        for cin, cout, ks, t, h, wd in CONV_SHAPES:
            conv = torch.nn.Conv3d(cin, cout, ks, padding = ks // 2).to(dev, torch.bfloat16)
            x = torch.randn(1, cin, t, h, wd, device = dev, dtype = torch.bfloat16, generator = g)
            with torch.no_grad():
                ref32 = F.conv3d(x.float(), conv.weight.float(), conv.bias.float(), padding = ks // 2)
                stock = conv(x)
                holder = torch.nn.Module()
                holder.conv = conv
                n_routed = tc.install(holder)
                before = dict(tc.COUNTS)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                y = conv(x)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                after = dict(tc.COUNTS)

            def rel(a_):
                return float((a_.float() - ref32).norm() / ref32.norm())
            crow.append({"shape": [cin, cout, ks, t, h, wd], "installed": n_routed,
                         "delta": {k_: after.get(k_, 0) - before.get(k_, 0) for k_ in after},
                         "rel_l2_triton": rel(y), "rel_l2_stock": rel(stock), "first_call_s": round(t1 - t0, 2)})
            save()
        obs["conv"] = crow
        routed_key = next((k_ for k_ in ("routed", "triton", "aiter") if k_ in tc.COUNTS), None)
        obs["conv_ok"] = bool(crow) and all(
            (r["delta"].get(routed_key, 0) > 0) and r["rel_l2_triton"] <= max(2 * r["rel_l2_stock"], 1e-2)
            for r in crow)
        obs["conv_counts"] = dict(tc.COUNTS)
    except Exception:  # noqa: BLE001
        obs["conv_error"] = traceback.format_exc()[-2000:]
        obs["conv_ok"] = False
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
