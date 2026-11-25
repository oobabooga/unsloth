#!/usr/bin/env python3
"""Items 5.1 / 5.2 at block level: one diffusers Krea2TransformerBlock (real Krea-2 Turbo config: hidden 6144, 48 / 12
heads, SwiGLU 16384; random bf16 weights) at 1024^2 (4096 image + 512 text tokens), eager, Studio's gfx1151 attention
wrapper installed (k/v heads-major). Arms interleaved in one process (harness: 3+ processes):
  base       stock module
  pad8       SwiGLU product written into a [M, 16384 + 8] buffer (torch.mul out= view: no extra copy), down reads it
  pad64      same with + 64
  col_attn   attention projections (q, k, v, gate, out) stored column-major (5.1; the shapes that won the micro)
  col_pad8   col_attn + pad8
Outputs vs base: torch.equal / max abs. Engagement: the padded buffer's pointer is checked inside the down call."""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a12common import env_info, interleave, write  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--rounds", type = int, default = 7)
    ap.add_argument("--tokens", type = int, default = 4608)
    a = ap.parse_args()
    sys.path.insert(0, a.backend)
    obs = {}
    t0 = time.time()
    try:
        import torch
        import torch.nn.functional as F
        obs["env"] = env_info()
        from core.inference import diffusion_attention as DA
        l2 = int(getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or 0)
        DA._ROCM_CONTIGUOUS_KV_L2[0] = l2
        obs["studio_kv_wrapper"] = bool(DA._install_contiguous_native())
        from diffusers.models.transformers.transformer_krea2 import Krea2TransformerBlock
        torch.manual_seed(0)
        H, I = 6144, 16384
        blk = Krea2TransformerBlock(H, I, 48, 12, 1e-5)
        with torch.no_grad():
            for p in blk.parameters():
                if p.dim() == 2 and p.shape[0] != 6:
                    p.normal_(0, p.shape[1] ** -0.5)
            blk.scale_shift_table.normal_(0, 0.02)
        blk = blk.to("cuda", torch.bfloat16).eval()
        g = torch.Generator(device = "cuda").manual_seed(1)
        x = torch.randn(1, a.tokens, H, generator = g, device = "cuda").to(torch.bfloat16)
        temb = (torch.randn(1, 1, 6 * H, generator = g, device = "cuda") * 0.1).to(torch.bfloat16)
        ff = blk.ff
        projs = [blk.attn.to_q, blk.attn.to_k, blk.attn.to_v, blk.attn.to_gate, blk.attn.to_out[0]]
        rows = {"W": [p.weight.data for p in projs]}
        rows["Wc"] = [w.t().contiguous().t() for w in rows["W"]]
        bufs = {p: torch.empty(a.tokens, I + p, device = "cuda", dtype = torch.bfloat16) for p in (8, 64)}
        hits = {"n": 0}

        def ff_pad(pad):
            def fwd(h):
                gt = F.silu(ff.gate(h))
                buf = bufs[pad][: h.shape[-2] if h.dim() == 2 else h.shape[1], :I]
                prod = torch.mul(gt.view(-1, I), ff.up(h).view(-1, I), out = buf)
                hits["n"] += int(prod.stride(0) == I + pad)
                return ff.down(prod).view(*h.shape[:-1], H)
            return fwd
        stock_ff = ff.forward

        def setup(col, pad):
            for p, w, wc in zip(projs, rows["W"], rows["Wc"]):
                p.weight.data = wc if col else w
            ff.forward = ff_pad(pad) if pad else stock_ff

        arms = {"base": (False, 0), "pad8": (False, 8), "pad64": (False, 64), "col_attn": (True, 0),
                "col_pad8": (True, 8)}

        def mk(col, pad):
            def fn():
                setup(col, pad)
                return blk(x, temb, None)
            return fn
        fns = {k: mk(*v) for k, v in arms.items()}
        with torch.no_grad():
            ref = fns["base"]()
            obs["exact"], obs["max_abs"] = {}, {}
            for k, fn in fns.items():
                y = fn()
                obs["exact"][k] = bool(torch.equal(y, ref))
                obs["max_abs"][k] = float((y.float() - ref.float()).abs().max())
            hits["n"] = 0
            res = interleave(fns, a.rounds, target_ms = 200.0)
        obs["pad_hits"] = hits["n"]
        obs["calls_per_round"] = res.pop("_calls_per_round")
        obs["med_ms"] = {k: v["med"] for k, v in res.items()}
        obs["ms"] = {k: v["ms"] for k, v in res.items()}
    except Exception:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    obs["seconds"] = round(time.time() - t0, 1)
    write(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
