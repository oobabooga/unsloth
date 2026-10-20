"""GeGLU: geglu_{exact,approx}_{forward,backward}_kernel, used by fast_lora apply_lora_mlp_geglu_exact /
_approx (Gemma / Gemma-2 / Gemma-3 dense MLP, LoRA training, 4-bit and 16-bit). Fixed BLOCK_SIZE=1024,
default num_warps (4)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
from _dense_common import config_sweep, once, small_vram

# Gemma-2-9B: I=14336; Gemma-7B I=24576
CASES = [
    dict(name="gemma2_9b_T4096_exact", T=4096, I=14336, approx=False),
    dict(name="gemma2_9b_T4096_approx", T=4096, I=14336, approx=True),
    dict(name="gemma7b_T16384_approx", T=16384, I=24576, approx=True),
    dict(name="tiny_T16_approx", T=16, I=14336, approx=True),
]


def gelu_f(e32, approx):
    if approx:
        return 0.5 * e32 * (1.0 + torch.tanh(0.7978845608028654 * (e32 + 0.044715 * e32 ** 3)))
    return 0.5 * e32 * (1.0 + torch.erf(e32 * 0.7071067811865476))


def ref_fwd(e, g, approx):
    return gelu_f(e.float(), approx).to(e.dtype) * g


def ref_bwd(DW, e, g, approx):
    e32 = e.float()
    if approx:
        a = 0.7978845608028654 * e32
        b = a * 0.044715 * e32 * e32
        T = 1.0 + torch.tanh(a + b)
        T2 = 0.5 * T
        df_de = T2 + (-T2 * (T - 2.0) * (a + 3.0 * b))
        f = (T2 * e32).to(DW.dtype)
    else:
        part = 0.5 * (torch.erf(e32 * 0.7071067811865476) + 1.0)
        f = (part * e32).to(DW.dtype)
        df_de = part + 0.3989422804014327 * e32 * torch.exp(-0.5 * e32 * e32)
    h = f * g
    df = DW * f
    dg = DW * g
    de = (dg.float() * df_de).to(DW.dtype)
    return h, df, de


def run(case, dtype, H):
    global torch
    import torch
    gl = import_unsloth("unsloth.kernels.geglu")
    T, I, approx = case["T"], case["I"], case["approx"]
    if small_vram() and T * I > 4096 * 24576:
        T = 4096 * 24576 // I
        H.emit(kind="info", what="T reduced to fit VRAM", T=T, T_requested=case["T"])
    dt = H.dtype
    fwdk = gl.geglu_approx_forward_kernel if approx else gl.geglu_exact_forward_kernel
    bwdk = gl.geglu_approx_backward_kernel if approx else gl.geglu_exact_backward_kernel
    rawf = gl._approx_forward_kernel if approx else gl._exact_forward_kernel
    rawb = gl._approx_backward_kernel if approx else gl._exact_backward_kernel
    e = torch.randn(1, T, I, device="cuda", dtype=dt)
    g = torch.randn(1, T, I, device="cuda", dtype=dt)
    DW = torch.randn(T, I, device="cuda", dtype=dt)
    cf = torch.compile(ref_fwd, dynamic=False)
    cb = torch.compile(ref_bwd, dynamic=False)
    shape = [T, I]
    H.interleave(dict(ours=lambda: fwdk(e, g), eager=lambda: ref_fwd(e, g, approx), compiled=lambda: cf(e, g, approx)),
                 "fwd", shape=shape, host=T <= 16)
    e2, g2 = e.view(T, I), g.view(T, I)
    DWc, ec, gc = DW.clone(), e2.clone(), g2.clone()
    H.interleave(dict(ours=lambda: bwdk(DWc, ec, gc), eager=lambda: ref_bwd(DW, e2, g2, approx),
                      compiled=lambda: cb(DW, e2, g2, approx)), "bwd", shape=shape, host=T <= 16)
    del DWc, ec, gc
    torch.cuda.empty_cache()
    # parity vs fp64 on a <=2048-token slice (keeps T4 under 15 GB); sweeps on the same slice
    S = min(T, 2048)
    e, g, DW = e[:, :S].contiguous(), g[:, :S].contiguous(), DW[:S].contiguous()
    e2, g2 = e.view(S, I), g.view(S, I)
    T = S
    ed, gd, Dd = e2.double(), g2.double(), DW.double()
    f64 = gelu_f(ed, approx)
    H.parity("fwd_h", fwdk(e, g), (f64 * gd).view(1, T, I), ref_fwd(e, g, approx), shape=shape)
    if approx:
        a = 0.7978845608028654 * ed; b = a * 0.044715 * ed * ed
        Tt = 1.0 + torch.tanh(a + b); dfde = 0.5 * Tt + (-0.5 * Tt * (Tt - 2.0) * (a + 3.0 * b))
    else:
        dfde = 0.5 * (torch.erf(ed * 0.7071067811865476) + 1.0) + 0.3989422804014327 * ed * torch.exp(-0.5 * ed * ed)
    r64 = (f64 * gd, Dd * f64, Dd * gd * dfde)
    outs = bwdk(DW.clone(), e2.clone(), g2.clone())
    H.parity("bwd", tuple(outs), r64, ref_bwd(DW, e2, g2, approx), shape=shape)
    del ed, gd, Dd, f64, r64, outs
    if T > 16:
        if once("compile"): H.compile_check("fwd", lambda a, b: fwdk(a, b), [(e, g), (e[:, : T // 2].contiguous(), g[:, : T // 2].contiguous())])
        n = e.numel()
        h = torch.empty_like(e)
        LI = 0 if n <= gl.INT32_SAFETY_BUFFER else 1

        def fcfg(B, w):
            return lambda: rawf[(triton.cdiv(n, B),)](e, g, h, n, BLOCK_SIZE=B, LONG_INDEXING=LI, num_warps=w, enable_fp_fusion=True)

        def bcfg(B, w):
            return lambda: rawb[(triton.cdiv(n, B),)](DW, e2, g2, n, BLOCK_SIZE=B, LONG_INDEXING=LI, num_warps=w, enable_fp_fusion=True)
        combos = [(1024, 4), (1024, 8), (2048, 8), (4096, 8), (512, 4)]
        config_sweep(H, "fwd_kernel", [("B%d_w%d" % c, fcfg(*c), c == (1024, 4)) for c in combos], shape)
        config_sweep(H, "bwd_kernel", [("B%d_w%d" % c, bcfg(*c), c == (1024, 4)) for c in combos], shape)


import triton  # noqa: E402

if __name__ == "__main__":
    main(CASES, run)
