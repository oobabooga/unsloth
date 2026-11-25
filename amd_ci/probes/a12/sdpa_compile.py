#!/usr/bin/env python3
"""Item 5.7b (thenoise utils/attention.py:167-177): a compiled DiT block on gfx1151 with the high-token SDPA call
in the graph (``ingraph``) vs wrapped in ``torch._dynamo.disable`` (``disable``). One arm per process, both shapes.
Times the whole compiled block forward (median of reps, CUDA events); compares to the eager block (max abs) and
records the graph count (the disable arm must show a break: proof the new path ran)."""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a12common import env_info, sha, time_calls, write  # noqa: E402

SHAPES = {"h3_37720": (37720, 5376, 56, 14336), "wan5b_27280": (27280, 3072, 24, 14336)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required = True, choices = ("ingraph", "disable"))
    ap.add_argument("--out", required = True)
    ap.add_argument("--shapes", default = ",".join(SHAPES))
    ap.add_argument("--reps", type = int, default = 5)
    a = ap.parse_args()
    obs = {"arm": a.arm, "rows": []}
    t0 = time.time()
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        obs["env"] = env_info()

        def attn(q, k, v):
            return F.scaled_dot_product_attention(q, k, v)
        attn_fn = torch._dynamo.disable(attn) if a.arm == "disable" else attn

        class Block(nn.Module):
            def __init__(self, dim, heads, ffn):
                super().__init__()
                self.h, self.d = heads, dim // heads
                self.n1 = nn.LayerNorm(dim, elementwise_affine = False, eps = 1e-6)
                self.qkv = nn.Linear(dim, 3 * dim)
                self.nq = nn.RMSNorm(self.d, eps = 1e-6)
                self.nk = nn.RMSNorm(self.d, eps = 1e-6)
                self.o = nn.Linear(dim, dim)
                self.n2 = nn.LayerNorm(dim, elementwise_affine = False, eps = 1e-6)
                self.up = nn.Linear(dim, ffn)
                self.down = nn.Linear(ffn, dim)

            def forward(self, x, shift, scale, gate):
                B, S, C = x.shape
                hN = self.n1(x) * (1 + scale) + shift
                q, k, v = self.qkv(hN).view(B, S, 3, self.h, self.d).unbind(2)
                q, k = self.nq(q), self.nk(k)
                y = attn_fn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
                x = x + gate * self.o(y.transpose(1, 2).reshape(B, S, C))
                return x + gate * self.down(F.gelu(self.up(self.n2(x)), approximate = "tanh"))
        for shape in a.shapes.split(","):
            S, dim, heads, ffn = SHAPES[shape]
            torch.manual_seed(0)
            blk = Block(dim, heads, ffn).to("cuda", torch.bfloat16).eval()
            g = torch.Generator(device = "cuda").manual_seed(1)
            x = torch.randn(1, S, dim, generator = g, device = "cuda").to(torch.bfloat16)
            mods = [(torch.randn(1, 1, dim, generator = g, device = "cuda") * 0.1).to(torch.bfloat16) for _ in range(3)]
            torch._dynamo.reset()
            from torch._dynamo.utils import counters
            counters.clear()
            cblk = torch.compile(blk, dynamic = False)
            row = {"shape": shape}
            with torch.no_grad():
                ref = blk(x, *mods)
                tc = time.time()
                y = cblk(x, *mods)
                torch.cuda.synchronize()
                row["compile_s"] = round(time.time() - tc, 1)
                row["unique_graphs"] = int(counters["stats"].get("unique_graphs", 0))
                row["graph_breaks"] = sum(counters["graph_break"].values())
                row["max_abs_vs_eager"] = float((y.float() - ref.float()).abs().max())
                row["out_sha"] = sha(y)
                del ref, y
                row.update(time_calls(lambda: cblk(x, *mods), a.reps, warm = 2))
            obs["rows"].append(row)
            del blk, cblk, x
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    obs["seconds"] = round(time.time() - t0, 1)
    write(a.out, obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
