#!/usr/bin/env python3
"""PR 18825 mechanism micro-probe: does a checkpointed block give the same LoRA gradients, bit for bit, as a kept
block, on this GPU in eager mode? Uses Studio's own checkpoint wrapper (_Checkpoint, plain and save-attention) and
fast-LoRA autograd function from --checkout. Also checks whether the same SDPA / GEMM / bnb nf4 call gives the same
bits on the main thread and on the autograd engine's thread (where a checkpoint recompute runs).

Writes JSON to --out only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--dim", type = int, default = 3072)
    ap.add_argument("--heads", type = int, default = 24)
    ap.add_argument("--seq", type = int, default = 1536)
    ap.add_argument("--blocks", type = int, default = 4)
    ap.add_argument("--device", default = "cuda")
    args = ap.parse_args()
    backend = Path(args.checkout) / "studio" / "backend"
    sys.path.insert(0, str(backend))
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    res: dict = {}
    try:
        res = run(args)
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-4000:]
    Path(args.out).write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")
    return 0


def run(args) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from core.training.diffusion_gc_budget import _Checkpoint
    from core.training.diffusion_lora_fast import _LoraProduct, _SplitCast

    dev = args.device
    res: dict = {"torch": torch.__version__, "hip": torch.version.hip, "platform": sys.platform,
                 "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"}
    D, H, S = args.dim, args.heads, args.seq

    class Lora(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.w = nn.Parameter(torch.randn(o, i, device = dev, dtype = torch.bfloat16) * i ** -0.5,
                                  requires_grad = False)
            self.a = nn.Parameter(torch.randn(16, i, device = dev) * i ** -0.5)
            self.b = nn.Parameter(torch.randn(o, 16, device = dev) * 0.01)

        def forward(self, x):
            dt = torch.get_autocast_dtype(dev)
            if x.dtype != dt:
                xb, xl = _SplitCast.apply(x, dt)
            else:
                xb = xl = x
            base = F.linear(xb, self.w)
            lora = _LoraProduct.apply(xl, self.a, self.b, self.a.detach().to(dt), self.b.detach().to(dt))
            return base + lora

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.v, self.o = Lora(D, D), Lora(D, D), Lora(D, D), Lora(D, D)
            self.fi, self.fo = Lora(D, 3 * D), Lora(3 * D, D)

        def forward(self, x, mod):
            h = F.layer_norm(x, (D,)) * (1 + mod)
            B = h.shape[0]
            q = self.q(h).view(B, S, H, D // H).transpose(1, 2)
            k = self.k(h).view(B, S, H, D // H).transpose(1, 2)
            v = self.v(h).view(B, S, H, D // H).transpose(1, 2)
            q = F.rms_norm(q, (D // H,))
            k = F.rms_norm(k, (D // H,))
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, S, D)
            x = x + self.o(a)
            h = F.layer_norm(x, (D,)) * (1 + mod)
            x = x + self.fo(F.gelu(self.fi(h), approximate = "tanh"))
            return x

    torch.manual_seed(0)
    blocks = nn.ModuleList([Block() for _ in range(args.blocks)])
    x0 = torch.randn(1, S, D, device = dev, dtype = torch.bfloat16)
    mod = torch.randn(1, 1, D, device = dev, dtype = torch.bfloat16) * 0.1
    tgt = torch.randn(1, S, D, device = dev, dtype = torch.bfloat16)
    params = [p for p in blocks.parameters() if p.requires_grad]

    def step(modes):
        for p in params:
            p.grad = None
        plain, sac = _Checkpoint(save_attention = False), _Checkpoint(save_attention = True)
        with torch.autocast(dev, dtype = torch.bfloat16):
            x = x0
            for blk, m in zip(blocks, modes):
                if m == "keep":
                    x = blk(x, mod)
                elif m == "ckpt":
                    x = plain(blk.forward, x, mod)
                else:
                    x = sac(blk.forward, x, mod)
            loss = F.mse_loss(x.float(), tgt.float())
        loss.backward()
        (torch.cuda.synchronize() if dev == "cuda" else None)
        g = torch.cat([p.grad.reshape(-1) for p in params])
        return loss.detach().clone(), g.clone(), sac.saved_attention

    n = args.blocks
    configs = {
        "all_ckpt (main plain)": ["ckpt"] * n,
        "all_ckpt_rep": ["ckpt"] * n,
        "all_keep": ["keep"] * n,
        "all_keep_rep": ["keep"] * n,
        "all_sac": ["sac"] * n,
        "all_sac_rep": ["sac"] * n,
        "half_keep_half_sac (budget)": ["keep"] * (n // 2) + ["sac"] * (n - n // 2),
        "half_keep_half_ckpt": ["keep"] * (n // 2) + ["ckpt"] * (n - n // 2),
    }
    out = {}
    ref_loss, ref_g, _ = step(configs["all_ckpt (main plain)"])
    for name, modes in configs.items():
        loss, g, saved = step(modes)
        d = (g - ref_g).abs()
        out[name] = {
            "loss_bit_identical": bool(torch.equal(loss, ref_loss)),
            "loss_delta": float((loss - ref_loss).abs()),
            "grad_bit_identical": bool(torch.equal(g, ref_g)),
            "grad_max_abs": float(d.max()),
            "grad_rel_l2": float(d.norm() / ref_g.norm()),
            "grad_frac_elems_differ": float((d != 0).float().mean()),
            "saved_attention_calls": saved,
        }
    res["block_modes"] = out

    # Same op, main thread vs autograd engine thread (a checkpoint recompute runs on the latter).
    thread: dict = {}
    q = torch.randn(1, H, S, D // H, device = dev, dtype = torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    xa = torch.randn(S, D, device = dev, dtype = torch.bfloat16)
    wa = torch.randn(D, D, device = dev, dtype = torch.bfloat16)
    box = {}

    class InBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            return t.clone()

        @staticmethod
        def backward(ctx, g):
            import threading
            box["thread"] = threading.current_thread().name
            box["sdpa"] = F.scaled_dot_product_attention(q, k, v)
            box["mm"] = torch.mm(xa, wa.t())
            box["linear"] = F.linear(xa, wa)
            with torch.enable_grad():
                qq = q.detach().requires_grad_()
                box["sdpa_grad_mode"] = F.scaled_dot_product_attention(qq, k, v).detach()
            return g

    main_sdpa = F.scaled_dot_product_attention(q, k, v)
    main_mm = torch.mm(xa, wa.t())
    main_lin = F.linear(xa, wa)
    with torch.no_grad():
        nograd_sdpa = F.scaled_dot_product_attention(q, k, v)
    qq = q.detach().requires_grad_()
    grad_sdpa = F.scaled_dot_product_attention(qq, k, v).detach()
    t = torch.ones(4, device = dev, requires_grad = True)
    InBackward.apply(t).sum().backward()
    (torch.cuda.synchronize() if dev == "cuda" else None)
    thread["engine_thread"] = box.get("thread")
    thread["sdpa_main_vs_engine_identical"] = bool(torch.equal(main_sdpa, box["sdpa"]))
    thread["sdpa_grad_main_vs_engine_identical"] = bool(torch.equal(grad_sdpa, box["sdpa_grad_mode"]))
    thread["sdpa_nograd_vs_grad_identical"] = bool(torch.equal(nograd_sdpa, grad_sdpa))
    thread["sdpa_nograd_vs_grad_max_abs"] = float((nograd_sdpa.float() - grad_sdpa.float()).abs().max())
    thread["mm_main_vs_engine_identical"] = bool(torch.equal(main_mm, box["mm"]))
    thread["linear_main_vs_engine_identical"] = bool(torch.equal(main_lin, box["linear"]))
    # Repeatability on the main thread.
    thread["sdpa_repeat_identical"] = bool(torch.equal(main_sdpa, F.scaled_dot_product_attention(q, k, v)))
    res["thread"] = thread

    # Which attention kernels ran (grad mode on), from the profiler's op names.
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities = [ProfilerActivity.CPU]) as prof:
            F.scaled_dot_product_attention(qq, k, v).sum().backward()
        res["sdpa_ops"] = sorted({e.key for e in prof.key_averages() if "attention" in e.key.lower()})
    except Exception as exc:  # noqa: BLE001
        res["sdpa_ops"] = f"{type(exc).__name__}: {exc}"

    # bnb nf4: per-call path vs a cached dequantised weight, forward and input grad.
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as bnbF

        lin = bnb.nn.Linear4bit(D, D, bias = False, compute_dtype = torch.bfloat16, quant_type = "nf4")
        lin.weight = bnb.nn.Params4bit(torch.randn(D, D) * D ** -0.5, requires_grad = False, quant_type = "nf4")
        lin = lin.to(dev)
        xx = torch.randn(S, D, device = dev, dtype = torch.bfloat16, requires_grad = True)
        with torch.autocast(dev, dtype = torch.bfloat16):
            y1 = lin(xx)
        g1, = torch.autograd.grad(y1.float().pow(2).sum(), xx)
        w = bnbF.dequantize_4bit(lin.weight.data, lin.weight.quant_state).to(torch.bfloat16)
        with torch.autocast(dev, dtype = torch.bfloat16):
            y2 = F.linear(xx.to(torch.bfloat16), w)
        g2, = torch.autograd.grad(y2.float().pow(2).sum(), xx)
        res["nf4"] = {"fwd_identical": bool(torch.equal(y1, y2)),
                      "fwd_max_abs": float((y1.float() - y2.float()).abs().max()),
                      "dgrad_identical": bool(torch.equal(g1, g2)),
                      "dgrad_rel_l2": float((g1.float() - g2.float()).norm() / g1.float().norm())}
    except Exception as exc:  # noqa: BLE001
        res["nf4"] = f"{type(exc).__name__}: {exc}"
    return res


if __name__ == "__main__":
    sys.exit(main())
