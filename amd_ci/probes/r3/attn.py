#!/usr/bin/env python3
"""R3-07: Studio's Triton INT8 attention (gfx1151 table) vs candidate configs, one fresh process per (shape, arm).

  attn.py --backend B --shape wan5b_720p --arm studio|c_aiter|c_bn64|c_wpe2 --out F [--reps 5]

The config is Studio's ``rocm_kernel_config(arch, 128)`` after ``select_config`` (its long-key variant at these
lengths), with the arm's overrides on top. Times ``reps`` calls (CUDA events, 256 MB flush between), median; accuracy
of the same output vs an fp64 softmax oracle on 256 sampled query rows x 3 heads (rel L2, rel L1, worst cosine)."""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

os.environ.setdefault("UNSLOTH_DIFFUSION_INT8_ATTENTION_IMPL", "triton")
SHAPES = {"wan5b_720p": (1, 27280, 24, 128), "h3_35490": (1, 35490, 56, 128), "h3_91728": (1, 91728, 56, 128)}
ARMS = {"studio": {}, "c_aiter": {"bn": 64, "stages": 2, "waves_per_eu": 2}, "c_bn64": {"bn": 64, "stages": 2},
        "c_wpe2": {"waves_per_eu": 2}}


def inputs(B, S, H, D, seed = 0):
    """Packed fused-QKV views [B, S, 3*H*D], DiT-like per-channel statistics (as amd_ci/probes/i8ts/bench.py)."""
    import torch
    dtype = torch.bfloat16
    g = torch.Generator(device = "cuda").manual_seed(1234 + seed)

    def rn(*s):
        return torch.randn(*s, generator = g, device = "cuda", dtype = torch.float32)
    wq, wk = torch.exp(0.4 * rn(1, 1, H, D)), torch.exp(0.4 * rn(1, 1, H, D))
    kofs, vofs, vsc = 0.6 * rn(1, 1, H, D), 0.5 * rn(1, 1, H, D), 0.5 + rn(1, 1, H, D).abs()
    buf = torch.empty(B, S, 3, H, D, device = "cuda", dtype = dtype)
    for i in range(0, S, 8192):
        n = min(8192, S - i)
        buf[:, i:i + n, 0] = (rn(B, n, H, D) * wq * 1.6).to(dtype)
        buf[:, i:i + n, 1] = (rn(B, n, H, D) * wk * 1.6 + kofs).to(dtype)
        buf[:, i:i + n, 2] = (rn(B, n, H, D) * vsc + vofs).to(dtype)
    buf = buf.view(B, S, 3 * H * D)
    hd = H * D
    return (buf[:, :, :hd].view(B, S, H, D), buf[:, :, hd:2 * hd].view(B, S, H, D),
            buf[:, :, 2 * hd:].view(B, S, H, D))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--shape", required = True)
    ap.add_argument("--arm", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--reps", type = int, default = 5)
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    import torch

    from core.inference import diffusion_int8_attention as M
    from core.inference import diffusion_int8_attention_triton as T
    B, S, H, D = SHAPES[a.shape]
    arch = M._gfx_arch(0)
    table = M.rocm_kernel_config(arch, D)
    obs = {"shape": a.shape, "arm": a.arm, "arch": arch, "torch": torch.__version__,
           "triton": __import__("triton").__version__, "table": table}
    if table is None:
        obs["error"] = f"no ROCm config for {arch}"
        open(a.out, "w", encoding = "utf-8").write(json.dumps(obs))
        return 0
    cfg = T.select_config(table, S)
    cfg = {k: v for k, v in cfg.items() if k not in ("long", "long_keys")}
    cfg.update(ARMS[a.arm])
    obs["cfg"] = cfg
    q, k, v = inputs(B, S, H, D)
    flush = torch.empty(int(256e6), dtype = torch.int8, device = "cuda")
    t0 = time.time()
    try:
        with torch.no_grad():
            for _ in range(2):
                out = T.int8_attention(q, k, v, cfg)
            torch.cuda.synchronize()
            ms = []
            for _ in range(a.reps):
                flush.zero_()
                e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                e0.record()
                out = T.int8_attention(q, k, v, cfg)
                e1.record()
                e1.synchronize()
                ms.append(e0.elapsed_time(e1))
        obs["ms"], obs["med_ms"] = ms, st.median(ms)
        heads = sorted({0, H // 2, H - 1})
        rows = torch.linspace(0, S - 1, 256, device = "cuda").long()
        qs = q[0, rows][:, heads].permute(1, 0, 2).double()
        ks = k[0][:, heads].permute(1, 0, 2).double()
        vs = v[0][:, heads].permute(1, 0, 2).double()
        ref = torch.matmul(torch.softmax(torch.matmul(qs, ks.transpose(1, 2)) / (D ** 0.5), -1), vs)
        got = out[0, rows][:, heads].permute(1, 0, 2).double()
        obs["rel_l2"] = float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref))
        obs["rel_l1"] = float((got - ref).abs().sum() / ref.abs().sum())
        obs["worst_cos"] = float(torch.nn.functional.cosine_similarity(got, ref, dim = -1).min())
        obs["finite"] = bool(torch.isfinite(out).all())
        obs["out_sha"] = __import__("hashlib").sha256(out.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                                                     ).hexdigest()[:16]
    except Exception as exc:  # noqa: BLE001
        obs["error"] = repr(exc)[:400]
    obs["seconds"] = round(time.time() - t0, 1)
    open(a.out, "w", encoding = "utf-8").write(json.dumps(obs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
