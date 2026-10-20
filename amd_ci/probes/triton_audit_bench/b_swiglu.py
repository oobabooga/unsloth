"""SwiGLU: swiglu_fg_kernel (fwd) / swiglu_DWf_DW_dfg_kernel (bwd), used by fast_lora apply_lora_mlp_swiglu
(Llama / Qwen / Mistral dense MLP, LoRA training, 4-bit and 16-bit)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *

# (tokens, intermediate): Llama-3-8B / Qwen3-8B have I=14336 / 12288; bsz*seq = 2*2048 and 8*2048
CASES = [
    dict(name="llama8b_T4096", T=4096, I=14336),
    dict(name="qwen3_8b_T16384", T=16384, I=12288),
    dict(name="tiny_T16", T=16, I=14336),
]


def ref_fwd(e, g):
    return (e.float() * torch.sigmoid(e.float())).to(e.dtype) * g


def ref_bwd(DW, e, g):
    e_float = e.float()
    se = torch.sigmoid(e_float)
    f = (se * e_float).to(DW.dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = (dg.float() * se * (1.0 + e_float * (1.0 - se))).to(DW.dtype)
    return h, df, de


def run(case, dtype, H):
    global torch
    import torch
    sw = import_unsloth("unsloth.kernels.swiglu")
    T, I = case["T"], case["I"]
    dt = H.dtype
    e = torch.randn(1, T, I, device="cuda", dtype=dt)
    g = torch.randn(1, T, I, device="cuda", dtype=dt)
    DW = torch.randn(T, I, device="cuda", dtype=dt)
    comp_fwd = torch.compile(ref_fwd, dynamic=False)
    comp_bwd = torch.compile(ref_bwd, dynamic=False)
    shape = [T, I]
    H.interleave(dict(
        ours=lambda: sw.swiglu_fg_kernel(e, g),
        eager=lambda: ref_fwd(e, g),
        compiled=lambda: comp_fwd(e, g),
    ), "fwd", shape=shape, host=T <= 16)
    e2, g2 = e.view(T, I), g.view(T, I)
    # ours is in place on (DW, e, g): bench on clones made outside the timed region would need a copy;
    # the in-place kernel is idempotent-safe for timing (values change, cost does not).
    DWc, ec, gc = DW.clone(), e2.clone(), g2.clone()
    H.interleave(dict(
        ours=lambda: sw.swiglu_DWf_DW_dfg_kernel(DWc, ec, gc),
        eager=lambda: ref_bwd(DW, e2, g2),
        compiled=lambda: comp_bwd(DW, e2, g2),
    ), "bwd", shape=shape, host=T <= 16)
    # parity on a <=2048-token slice (fp64 refs; keeps T4 under 15 GB)
    del DWc, ec, gc
    torch.cuda.empty_cache()
    S = min(T, 2048)
    e, g, DW = e[:, :S].contiguous(), g[:, :S].contiguous(), DW[:S].contiguous()
    e2, g2 = e.view(S, I), g.view(S, I)
    o = sw.swiglu_fg_kernel(e, g)
    r32 = (e.double() * torch.sigmoid(e.double())) * g.double()
    H.parity("fwd_h", o, r32, ref_fwd(e, g), shape=shape)
    outs = sw.swiglu_DWf_DW_dfg_kernel(DW.clone(), e2.clone(), g2.clone())
    ed, gd, Dd = e2.double(), g2.double(), DW.double()
    se = torch.sigmoid(ed)
    r = (se * ed * gd, Dd * se * ed, Dd * gd * se * (1 + ed * (1 - se)))
    H.parity("bwd", tuple(outs), r, ref_bwd(DW, e2, g2), shape=shape)
    if T > 16:
        H.compile_check("fwd", lambda a, b: sw.swiglu_fg_kernel(a, b),
                        [(e, g), (e[:, : S // 2].contiguous(), g[:, : S // 2].contiguous())])
        # heuristic check (fixed BLOCK_SIZE, default num_warps) on the parity slice [S, I]
        import triton
        from _dense_common import config_sweep
        n = e.numel(); h = torch.empty_like(e)
        DWc, ec, gc = DW.clone(), e2.clone(), g2.clone()
        LI = 0 if n <= sw.INT32_SAFETY_BUFFER else 1
        B0 = sw.BLOCK_SIZE
        fcfg = lambda B, w: (lambda: sw._fg_kernel[(triton.cdiv(n, B),)](e, g, h, n, BLOCK_SIZE=B, LONG_INDEXING=LI, num_warps=w, enable_fp_fusion=True))
        bcfg = lambda B, w: (lambda: sw._DWf_DW_dfg_kernel[(triton.cdiv(n, B),)](DWc, ec, gc, n, BLOCK_SIZE=B, LONG_INDEXING=LI, num_warps=w, enable_fp_fusion=True))
        combos = [(B0, 4), (B0, 8), (2048, 8), (4096, 8), (512, 4)]
        config_sweep(H, "fwd_kernel", [("B%d_w%d" % c, fcfg(*c), c == (B0, 4)) for c in combos], [S, I])
        config_sweep(H, "bwd_kernel", [("B%d_w%d" % c, bcfg(*c), c == (B0, 4)) for c in combos], [S, I])


if __name__ == "__main__":
    main(CASES, run)
