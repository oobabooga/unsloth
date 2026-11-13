"""unsloth#17548 repro: compiled SDPA backward stride asserts, with and without a backport of
pytorch#202297 (is_aligned_realized_tensor ignores provisional FlexibleLayout strides).
Usage: python sdpa_layout_repro.py OUT.json [--patch]. One case per fresh dynamo state."""

import json
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

out_path = sys.argv[1]
patch = "--patch" in sys.argv
dev = torch.device("cuda", 0)
res = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
       "device": torch.cuda.get_device_name(0), "patch": patch, "cases": []}

if patch:
    import torch._inductor.ir as ir
    import torch._inductor.lowering as lowering

    _orig = ir.is_aligned_realized_tensor

    def is_aligned_realized_tensor(x, alignment):
        if isinstance(x, ir.IRNode) and isinstance(x.maybe_get_layout(), ir.FlexibleLayout):
            return False
        return _orig(x, alignment)

    ir.is_aligned_realized_tensor = is_aligned_realized_tensor
    res["lowering_reads_ir_attr"] = not hasattr(lowering, "is_aligned_realized_tensor")


def err(e):
    return f"{type(e).__name__}: {e}"[:1500]


def run(name, fn, args, backend, backward):
    torch._dynamo.reset()
    row = {"name": name, "backend": backend, "backward": backward}
    try:
        ctx = sdpa_kernel([getattr(SDPBackend, backend)]) if backend != "default" else sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        with ctx:
            eager = fn(*args)
            out = torch.compile(fn)(*args)
            row["max_abs_diff"] = float((out.float() - eager.float()).abs().max())
            if backward:
                out.float().square().mean().backward()
        torch.cuda.synchronize()
        row["ok"] = True
    except Exception as e:
        row["ok"] = False
        row["error"] = err(e)
        row["stride_assert"] = "expected size" in str(e) and "stride" in str(e)
    res["cases"].append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "error"}), flush = True)


dtype = torch.bfloat16


def upstream_case():
    def pma_attention(query, a, b, weight):
        x = F.silu(a).transpose(1, 2) + b.transpose(1, 2)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + 1e-5) * weight
        kv = x.unsqueeze(1).contiguous()
        return F.scaled_dot_product_attention(query, kv, kv) + F.scaled_dot_product_attention(query * 2, kv, kv)

    mk = lambda *s: torch.randn(*s, device = dev, dtype = dtype, requires_grad = True)
    w = (torch.rand(64, device = dev, dtype = dtype) + 0.5).requires_grad_()
    return pma_attention, (mk(4, 1, 8, 64), mk(4, 64, 32), mk(4, 64, 32), w)


def qwen35_case(B, S, Hq, Hk, D, ckpt):
    hid = 64
    mk = lambda *s: (torch.randn(*s, device = dev, dtype = dtype) * 0.05).requires_grad_()
    x = torch.randn(B, S, hid, device = dev, dtype = dtype, requires_grad = True)
    wq, wk, wv, wo = mk(hid, 2 * Hq * D), mk(hid, Hk * D), mk(hid, Hk * D), mk(Hq * D, hid)
    nq = torch.ones(D, device = dev, dtype = dtype, requires_grad = True)
    nk = torch.ones(D, device = dev, dtype = dtype, requires_grad = True)
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device = dev, dtype = torch.float32) / D))
    fr = torch.outer(torch.arange(S, device = dev, dtype = torch.float32), inv)
    emb = torch.cat([fr, fr], -1)
    cos, sin = emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]

    def rms(t, w):
        tf = t.float()
        return (tf * torch.rsqrt(tf.pow(2).mean(-1, keepdim = True) + 1e-6)).to(t.dtype) * w

    def rot(t):
        return torch.cat([-t[..., D // 2:], t[..., : D // 2]], -1)

    def layer(x, wq, wk, wv, wo, nq, nk):
        q, gate = torch.chunk((x @ wq).view(B, S, Hq, 2 * D), 2, dim = -1)
        q = rms(q, nq).transpose(1, 2)
        k = rms((x @ wk).view(B, S, Hk, D), nk).transpose(1, 2)
        v = (x @ wv).view(B, S, Hk, D).transpose(1, 2)
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        o = F.scaled_dot_product_attention(q, k, v, is_causal = True, enable_gqa = Hq != Hk)
        return (o.transpose(1, 2).reshape(B, S, Hq * D) * torch.sigmoid(gate.reshape(B, S, Hq * D))) @ wo

    def f(x, *p):
        if ckpt:
            return torch.utils.checkpoint.checkpoint(layer, x, *p, use_reentrant = False)
        return layer(x, *p)

    return f, (x, wq, wk, wv, wo, nq, nk)


for backend in ("FLASH_ATTENTION", "default"):
    fn, args = upstream_case()
    run("upstream_202297", fn, args, backend, True)
    for shape in ((1, 93, 24, 4, 256), (2, 93, 4, 2, 256), (1, 93, 16, 16, 128)):
        for ckpt in (False, True):
            fn, args = qwen35_case(*shape, ckpt)
            run(f"qwen35{list(shape)} ckpt={ckpt}", fn, args, backend, True)

with open(out_path, "w", encoding = "utf-8") as fh:
    json.dump(res, fh)
