#!/usr/bin/env python3
"""M10: comfy-kitchen HIP INT8 attention (``comfy_kitchen.sage_attention.int8_attention``) vs Studio's Triton INT8
attention on gfx1151 (Studio's ROCm table: D128 ``pv=f16_t``, long-key variant past ``long_keys``), one process.

  kern_attn.py --backend B --out F [--kitchen PKG_ROOT] [--reps 5] [--order sk|ks]

Shapes D128, B 1, H 24, S 4096 / 16384 / 37740 (self-attention). Inputs are the r3 attn.py DiT-like statistics in
Studio's packed [B, S, H, D] layout; kitchen gets the same values as contiguous [B, H, S, D] (made outside timing).
Timing: CUDA events, 256 MB flush before every call, arms alternate per rep (median of --reps). Error of each output vs
an fp64 softmax oracle on 256 sampled query rows x 3 heads (rel L2, rel L1, worst cosine). Writes one JSON."""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

os.environ.setdefault("UNSLOTH_DIFFUSION_INT8_ATTENTION_IMPL", "triton")
SHAPES = [(1, 4096, 24, 128), (1, 16384, 24, 128), (1, 37740, 24, 128)]


def inputs(B, S, H, D, seed = 0):
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


def err(out_bshd, q, k, v, S, H, D):
    import torch
    heads = sorted({0, H // 2, H - 1})
    rows = torch.linspace(0, S - 1, 256, device = "cuda").long()
    qs = q[0, rows][:, heads].permute(1, 0, 2).double()
    ks = k[0][:, heads].permute(1, 0, 2).double()
    vs = v[0][:, heads].permute(1, 0, 2).double()
    ref = torch.matmul(torch.softmax(torch.matmul(qs, ks.transpose(1, 2)) / (D ** 0.5), -1), vs)
    got = out_bshd[0, rows][:, heads].permute(1, 0, 2).double()
    return {"rel_l2": float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)),
            "rel_l1": float((got - ref).abs().sum() / ref.abs().sum()),
            "worst_cos": float(torch.nn.functional.cosine_similarity(got, ref, dim = -1).min()),
            "finite": bool(torch.isfinite(out_bshd).all())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--kitchen", default = "")
    ap.add_argument("--reps", type = int, default = 5)
    ap.add_argument("--order", default = "sk")
    a = ap.parse_args()
    obs: dict = {"t0": time.time(), "order": a.order, "rows": []}

    def save():
        open(a.out, "w", encoding = "utf-8").write(json.dumps(obs, default = str))
    try:
        sys.path.insert(0, a.backend)
        import torch

        from core.inference import diffusion_int8_attention as M
        from core.inference import diffusion_int8_attention_triton as T
        arch = M._gfx_arch(0)
        table = M.rocm_kernel_config(arch, 128)
        obs.update(arch = arch, torch = torch.__version__, triton = __import__("triton").__version__, table = table)
        KA = None
        if a.kitchen:
            sys.path.insert(0, a.kitchen)
            try:
                import comfy_kitchen  # noqa: F401  (registers torch.ops.comfy_kitchen.*)
                from comfy_kitchen.backends import hip as KH
                from comfy_kitchen.sage_attention import int8_attention as KA
                obs["kitchen_file"] = KH.__file__
                obs["kitchen_int8_attention_available"] = bool(KH.int8_attention_is_available())
            except Exception as exc:  # noqa: BLE001
                obs["kitchen_error"] = repr(exc)[:600]
                KA = None
        if table is None:
            obs["error"] = f"no ROCm config for {arch}"
            save()
            return 0
        flush = torch.empty(int(256e6), dtype = torch.int8, device = "cuda")
        for B, S, H, D in SHAPES:
            row = {"S": S, "H": H, "D": D}
            cfg = T.select_config(table, S)
            cfg = {kk: vv for kk, vv in cfg.items() if kk not in ("long", "long_keys")}
            row["studio_cfg"] = cfg
            q, k, v = inputs(B, S, H, D)
            arms = {"studio": lambda: T.int8_attention(q, k, v, cfg)}
            if KA is not None:
                qh, kh, vh = (t.permute(0, 2, 1, 3).contiguous() for t in (q, k, v))
                arms["kitchen"] = lambda: KA(qh, kh, vh)
            names = list(arms) if a.order == "sk" else list(arms)[::-1]
            res = {n: {"ms": []} for n in names}
            outs = {}
            try:
                with torch.no_grad():
                    for n in list(names):
                        try:
                            for _ in range(2):
                                outs[n] = arms[n]()
                            torch.cuda.synchronize()
                        except Exception as exc:  # noqa: BLE001 - one arm failing keeps the other
                            row[f"{n}_error"] = repr(exc)[:500]
                            names.remove(n)
                            res.pop(n, None)
                    for rep in range(a.reps):
                        for n in (names if rep % 2 == 0 else names[::-1]):
                            flush.zero_()
                            e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                            e0.record()
                            outs[n] = arms[n]()
                            e1.record()
                            e1.synchronize()
                            res[n]["ms"].append(e0.elapsed_time(e1))
                for n in names:
                    res[n]["med_ms"] = st.median(res[n]["ms"])
                    o = outs[n]
                    o = o.permute(0, 2, 1, 3) if n == "kitchen" else o  # kitchen returns [B, H, S, D]
                    res[n].update(err(o, q, k, v, S, H, D))
            except Exception as exc:  # noqa: BLE001
                row["error"] = repr(exc)[:500]
            row["res"] = res
            obs["rows"].append(row)
            save()
            del q, k, v, outs, arms
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    obs["seconds"] = round(time.time() - obs["t0"], 1)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
