"""RoPE: fast_rope_embedding (Fast_RoPE_Embedding without position ids, Fast_RoPE_Embedding_QK with
them). Called by llama / mistral / qwen2 / qwen3 (flag off) / gemma / gemma2 / granite / cohere /
falcon_h1 attention, train + infer prefill, 4-bit and 16-bit. inplace_rope_embedding is pure torch
(Slow_RoPE_Embedding), not a Triton kernel."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
from _dense_common import fwd_bwd, config_sweep, grads_of, once

CASES = [
    dict(name="llama8b_B2S2048_noidx", B=2, S=2048, H=32, KV=8, D=128, idx=False),
    dict(name="llama8b_B2S2048_idx", B=2, S=2048, H=32, KV=8, D=128, idx=True),
    dict(name="llama8b_B8S2048_noidx", B=8, S=2048, H=32, KV=8, D=128, idx=False),
    dict(name="tiny_B1S16_idx", B=1, S=16, H=32, KV=8, D=128, idx=True),
]


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def ref_rope(Q, K, cos, sin, pos):
    # HF apply_rotary_pos_emb, Q / K [B, H, S, D], cos / sin [S_max, D]
    c = cos[pos].unsqueeze(1)
    s = sin[pos].unsqueeze(1)
    return Q * c + rotate_half(Q) * s, K * c + rotate_half(K) * s


def run(case, dtype, Hh):
    global torch
    import torch
    rp = import_unsloth("unsloth.kernels.rope_embedding")
    B, S, NH, KV, D = case["B"], case["S"], case["H"], case["KV"], case["D"]
    dt = Hh.dtype
    g = torch.Generator(device="cuda").manual_seed(0)
    inv = 1.0 / (500000 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    t = torch.arange(max(S, 4096), device="cuda").float()
    fr = torch.outer(t, inv); emb = torch.cat((fr, fr), -1)
    cos, sin = emb.cos().to(dt), emb.sin().to(dt)
    pos = torch.arange(S, device="cuda").view(1, S).expand(B, S).contiguous()
    pos_arg = pos if case["idx"] else None
    q0 = torch.randn(B, S, NH, D, device="cuda", dtype=dt, generator=g)
    k0 = torch.randn(B, S, KV, D, device="cuda", dtype=dt, generator=g)
    shape = [B, S, NH, KV, D]

    def mk():
        # non-leaf [B, H, S, D] views of [B, S, H, D] storage, as after the q/k projections
        ql = q0.clone().requires_grad_(True); kl = k0.clone().requires_grad_(True)
        return ((ql * 1).transpose(1, 2), (kl * 1).transpose(1, 2))

    ours = lambda Q, K: rp.fast_rope_embedding(Q, K, cos, sin, pos_arg)
    eager = lambda Q, K: ref_rope(Q, K, cos, sin, pos)
    comp = torch.compile(ref_rope, dynamic=False)
    compiled = lambda Q, K: comp(Q, K, cos, sin, pos)
    fwd_bwd(Hh, dict(ours=ours, eager=eager, compiled=compiled), mk, shape, host=S <= 16)

    # parity (ours rotates in place, so run it on fresh inputs)
    Qo, Ko = mk(); oo, go = grads_of(ours, (Qo, Ko))
    Qr, Kr = mk(); orr, gr = grads_of(eager, (Qr, Kr))
    q64 = q0.double().transpose(1, 2).requires_grad_(True); k64 = k0.double().transpose(1, 2).requires_grad_(True)
    o64 = ref_rope(q64, k64, cos.double(), sin.double(), pos)
    gg = torch.Generator(device="cuda").manual_seed(1234)
    gq = torch.randn(o64[0].shape, device="cuda", dtype=dt, generator=gg)
    gk = torch.randn(o64[1].shape, device="cuda", dtype=dt, generator=gg)
    g64 = torch.autograd.grad(o64, (q64, k64), (gq.double(), gk.double()))
    Hh.parity("fwd", tuple(oo), tuple(x.detach() for x in o64), tuple(orr), shape=shape)
    Hh.parity("bwd", tuple(go), g64, tuple(gr), shape=shape)

    if S > 16:
        def f(Q, K):
            return rp.fast_rope_embedding(Q, K, cos, sin, pos_arg if Q.shape[2] == S else pos_arg[:, : Q.shape[2]] if pos_arg is not None else None)
        q1, k1 = mk()
        q2 = (q0[:, : S // 2].clone().requires_grad_(True) * 1).transpose(1, 2)
        k2 = (k0[:, : S // 2].clone().requires_grad_(True) * 1).transpose(1, 2)
        if once("compile" + str(case["idx"])): Hh.compile_check("fast_rope_embedding", lambda Q, K: sum(x.float().sum() for x in f(Q, K)), [(q1, k1), (q2, k2)])

        orig = rp.calculate_settings
        qn, kn = mk()
        qn, kn = qn.detach(), kn.detach()
        default_w = orig(D // 2 if not case["idx"] else D)[1]

        def with_warps(w):
            def fn():
                rp.calculate_settings = lambda n: (orig(n)[0], w)
                try:
                    with torch.no_grad():
                        rp.fast_rope_embedding(qn, kn, cos, sin, pos_arg)
                finally:
                    rp.calculate_settings = orig
            return fn
        config_sweep(Hh, "rope_kernel_warps", [("w%d" % w, with_warps(w), w == default_w) for w in (1, 2, 4, 8)], shape)


if __name__ == "__main__":
    main(CASES, run)
