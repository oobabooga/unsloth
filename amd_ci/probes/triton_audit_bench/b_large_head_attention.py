"""head_dim 512 attention: large_head_attention (_lh_fwd / _lh_bwd_dq / _lh_bwd_dkdv, autotuned), installed by
patch_large_head_attention (models/vision.py) as the SDPA router for Gemma-4 global layers, sm80+ only
(T4 keeps SDPA), fp16 / bf16, train + prefill, 4-bit and 16-bit. Reference = what it replaces: HF's
sdpa_attention_forward (repeat_kv + SDPA is_causal) and its torch.compile."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
from _dense_common import fwd_bwd, grads_of, once

# Gemma-4 global layers: 12B 16 q / 1 kv head, 31B 32 q / 4 kv heads, head_dim 512
CASES = [
    dict(name="gemma4_12b_global_B1S2048", B=1, S=2048, H=16, KV=1, D=512, min_cap=(8, 0)),
    dict(name="gemma4_31b_global_B1S4096", B=1, S=4096, H=32, KV=4, D=512, min_cap=(8, 0), tune_s=(512, 4096)),
    dict(name="gemma4_31b_global_B2S2048", B=2, S=2048, H=32, KV=4, D=512, min_cap=(8, 0)),
    dict(name="tiny_B1S16", B=1, S=16, H=16, KV=1, D=512, min_cap=(8, 0)),
]


def sdpa_ref(q, k, v, scale):
    G = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(G, dim=1)
    v = v.repeat_interleave(G, dim=1)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    return o.transpose(1, 2).contiguous()


def math_ref(q, k, v, scale):
    G = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(G, dim=1); v = v.repeat_interleave(G, dim=1)
    S = q.shape[2]
    s = (q @ k.transpose(-1, -2)) * scale
    m = torch.ones(S, S, dtype=torch.bool, device=q.device).tril()
    s = s.masked_fill(~m, float("-inf"))
    return (s.softmax(-1) @ v).transpose(1, 2)


def run(case, dtype, H):
    global torch
    import torch
    lh = import_unsloth("unsloth.kernels.large_head_attention")
    B, S, NH, KV, D = case["B"], case["S"], case["H"], case["KV"], case["D"]
    dt = H.dtype
    scale = D ** -0.5
    g = torch.Generator(device="cuda").manual_seed(0)
    q0 = torch.randn(B, NH, S, D, device="cuda", dtype=dt, generator=g)
    k0 = torch.randn(B, KV, S, D, device="cuda", dtype=dt, generator=g)
    v0 = torch.randn(B, KV, S, D, device="cuda", dtype=dt, generator=g)
    lo, hi = lh._dense_intervals(B, S, True, q0.device)
    shape = [B, S, NH, KV, D]

    def mk():
        return tuple(x.clone().requires_grad_(True) for x in (q0, k0, v0))

    ours = lambda q, k, v: lh.large_head_attention(q, k, v, lo, hi, scale)
    sdpa = lambda q, k, v: sdpa_ref(q, k, v, scale)
    comp = torch.compile(sdpa_ref, dynamic=False)
    compiled = lambda q, k, v: comp(q, k, v, scale)
    # warm the autotuners outside the timed region
    oo, go = grads_of(ours, mk())
    fwd_bwd(H, dict(ours=ours, sdpa=sdpa, sdpa_compiled=compiled), mk, shape, host=S <= 16)
    for nm in ("_lh_fwd", "_lh_bwd_dq", "_lh_bwd_dkdv"):
        H.autotune_report(nm, getattr(lh, nm))

    orr, gr = grads_of(sdpa, mk())
    q32, k32, v32 = (x.float().requires_grad_(True) for x in (q0, k0, v0))
    o32 = math_ref(q32, k32, v32, scale)
    gg = torch.Generator(device="cuda").manual_seed(1234)
    gy = torch.randn(o32.shape, device="cuda", dtype=dt, generator=gg).float()
    g32 = torch.autograd.grad(o32, (q32, k32, v32), gy)
    H.parity("fwd", oo[0], o32.detach(), orr[0], shape=shape)
    H.parity("bwd", tuple(go), g32, tuple(gr), shape=shape)
    del q32, k32, v32, o32, g32
    if case.get("tune_s"):
        # autotune key is (HEAD_DIM, G): the config picked at the first seq length is reused for all.
        # Re-tune at each S in a clean cache and report the pick + spread per S.
        for S_t in case["tune_s"]:
            for nm in ("_lh_fwd", "_lh_bwd_dq", "_lh_bwd_dkdv"):
                getattr(lh, nm).cache.clear()
            ins = tuple(torch.randn(B, h, S_t, D, device="cuda", dtype=dt).requires_grad_(True) for h in (NH, KV, KV))
            lo_, hi_ = lh._dense_intervals(B, S_t, True, q0.device)
            out = lh.large_head_attention(*ins, lo_, hi_, scale)
            torch.autograd.grad(out, ins, torch.randn_like(out))
            for nm in ("_lh_fwd", "_lh_bwd_dq", "_lh_bwd_dkdv"):
                H.autotune_report("%s@S%d" % (nm, S_t), getattr(lh, nm))
            del ins, out
    if S > 16:
        def f(q, k, v):
            lo_, hi_ = lh._dense_intervals(q.shape[0], q.shape[2], True, q.device)
            return lh.large_head_attention(q, k, v, lo_, hi_, scale).float().sum()
        S2 = S // 2
        if once("compile"): H.compile_check("large_head_attention", f,
                        [mk(), tuple(x[:, :, :S2].clone().requires_grad_(True) for x in (q0, k0, v0))])


if __name__ == "__main__":
    main(CASES, run)
