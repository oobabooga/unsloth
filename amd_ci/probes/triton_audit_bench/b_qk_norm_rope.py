"""Fused Q/K RMSNorm + RoPE: fast_qk_norm_rope (Fast_QK_Norm_RoPE). Qwen3 attention only, and only when
UNSLOTH_FUSED_QKNORM_ROPE=1 (default off: models/qwen3.py then runs fast_rms_layernorm x2 +
fast_rope_embedding = the 'unfused' arm here). Frozen q/k norm weights (LoRA), train + infer."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
from _dense_common import fwd_bwd, config_sweep, grads_of, once, small_vram

CASES = [
    dict(name="qwen3_8b_B2S2048_idx", B=2, S=2048, H=32, KV=8, D=128, idx=True),
    dict(name="qwen3_8b_B2S2048_noidx", B=2, S=2048, H=32, KV=8, D=128, idx=False),
    dict(name="qwen3_8b_B8S2048_idx", B=8, S=2048, H=32, KV=8, D=128, idx=True),
    dict(name="tiny_B1S16_idx", B=1, S=16, H=32, KV=8, D=128, idx=True),
]


class Norm:
    def __init__(self, W, eps):
        self.weight = W
        self.variance_epsilon = eps


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def ref(Q, K, Wq, Wk, cos, sin, pos, eps):
    def rms(x, w):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return w * xf.to(x.dtype)
    q = rms(Q, Wq).transpose(1, 2)
    k = rms(K, Wk).transpose(1, 2)
    c = cos[pos].unsqueeze(1); s = sin[pos].unsqueeze(1)
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s


def run(case, dtype, Hh):
    global torch
    import torch
    qk = import_unsloth("unsloth.kernels.qk_norm_rope")
    B, S, NH, KV, D = case["B"], case["S"], case["H"], case["KV"], case["D"]
    if small_vram() and B > 4:
        B = 4
        Hh.emit(kind="info", what="B reduced to fit VRAM", B=B, B_requested=case["B"])
    dt = Hh.dtype
    eps = 1e-6
    g = torch.Generator(device="cuda").manual_seed(0)
    inv = 1.0 / (1000000 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    t = torch.arange(max(S, 4096), device="cuda").float()
    fr = torch.outer(t, inv); emb = torch.cat((fr, fr), -1)
    cos, sin = emb.cos().to(dt), emb.sin().to(dt)
    pos = torch.arange(S, device="cuda").view(1, S).expand(B, S).contiguous()
    pos_arg = pos if case["idx"] else None
    q0 = torch.randn(B, S, NH, D, device="cuda", dtype=dt, generator=g)
    k0 = torch.randn(B, S, KV, D, device="cuda", dtype=dt, generator=g)
    Wq = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(dt)
    Wk = (1 + 0.1 * torch.randn(D, device="cuda", generator=g)).to(dt)
    qn, kn = Norm(Wq, eps), Norm(Wk, eps)
    shape = [B, S, NH, KV, D]

    def mk():
        return (q0.clone().requires_grad_(True) * 1, k0.clone().requires_grad_(True) * 1)

    ours = lambda Q, K: qk.fast_qk_norm_rope(qn, kn, Q, K, cos, sin, pos_arg)
    unfused = lambda Q, K: qk._unfused(qn, kn, Q, K, cos, sin, pos_arg)
    eager = lambda Q, K: ref(Q, K, Wq, Wk, cos, sin, pos, eps)
    comp = torch.compile(ref, dynamic=False)
    compiled = lambda Q, K: comp(Q, K, Wq, Wk, cos, sin, pos, eps)
    fwd_bwd(Hh, dict(ours=ours, unsloth_unfused=unfused, eager=eager, compiled=compiled), mk, shape, host=S <= 16)

    oo, go = grads_of(ours, mk())
    ou, gu = grads_of(unfused, mk())
    orr, gr = grads_of(eager, mk())
    q64 = q0.double().requires_grad_(True); k64 = k0.double().requires_grad_(True)
    o64 = ref(q64, k64, Wq.double(), Wk.double(), cos.double(), sin.double(), pos, eps)
    gg = torch.Generator(device="cuda").manual_seed(1234)
    gq = torch.randn(o64[0].shape, device="cuda", dtype=dt, generator=gg)
    gk = torch.randn(o64[1].shape, device="cuda", dtype=dt, generator=gg)
    g64 = torch.autograd.grad(o64, (q64, k64), (gq.double(), gk.double()))
    o64 = tuple(x.detach() for x in o64)
    Hh.parity("fwd", tuple(oo), o64, tuple(orr), shape=shape)
    Hh.parity("bwd", tuple(go), g64, tuple(gr), shape=shape)
    Hh.parity("fwd_unfused", tuple(ou), o64, tuple(orr), shape=shape)
    Hh.parity("bwd_unfused", tuple(gu), g64, tuple(gr), shape=shape)

    if S > 16:
        def f(Q, K):
            p = pos_arg[:, : Q.shape[1]].contiguous() if pos_arg is not None else None
            return sum(x.float().sum() for x in qk.fast_qk_norm_rope(qn, kn, Q, K, cos, sin, p))
        if once("compile"): Hh.compile_check("fast_qk_norm_rope", f,
                         [mk(), (q0[:, : S // 2].clone().requires_grad_(True) * 1, k0[:, : S // 2].clone().requires_grad_(True) * 1)])
        Q, K = q0.clone(), k0.clone()
        ob, ow = qk._block_h, qk._num_warps
        dflt = (ob(NH, KV, D), ow(ob(NH, KV, D), D))

        def cfg(bh, w):
            def fn():
                qk._block_h = lambda *a: bh
                qk._num_warps = lambda *a: w
                try:
                    with torch.no_grad():
                        qk.fast_qk_norm_rope(qn, kn, Q, K, cos, sin, pos_arg)
                finally:
                    qk._block_h, qk._num_warps = ob, ow
            return fn
        combos = [dflt] + [c for c in ((8, 8), (4, 4), (2, 4), (1, 4), (8, 2)) if c != dflt]
        config_sweep(Hh, "fwd_kernel(BLOCK_H,warps)", [("BH%d_w%d" % c, cfg(*c), c == dflt) for c in combos], shape)


if __name__ == "__main__":
    main(CASES, run)
