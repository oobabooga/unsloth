"""RMSNorm: fast_rms_layernorm (Fast_RMS_Layernorm) used for every decoder input / post-attention norm and
the final norm in llama / mistral / qwen2 / qwen3 (+ q/k norm when QK_NORM_ROPE is off) / qwen3_moe /
gemma (gemma=True) / gemma2 / granite / glm4_moe / falcon_h1, train + infer, 4-bit and 16-bit.
Narrow rows (33..128 cols, e.g. Qwen3 head_dim 128 q/k norm) take the multi-row kernels."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
from _dense_common import fwd_bwd, config_sweep, grads_of, once

CASES = [
    dict(name="llama8b_T4096_frozenW", T=4096, N=4096, gemma=False, train_w=False),
    dict(name="llama8b_T16384_frozenW", T=16384, N=4096, gemma=False, train_w=False),
    dict(name="llama8b_T4096_trainW", T=4096, N=4096, gemma=False, train_w=True),
    dict(name="gemma2_9b_T4096_gemma", T=4096, N=3584, gemma=True, train_w=False),
    dict(name="qwen3_qknorm_T4096x32_D128", T=4096 * 32, N=128, gemma=False, train_w=False),
    dict(name="tiny_T16", T=16, N=4096, gemma=False, train_w=False),
]


class Norm:
    def __init__(self, W, eps):
        self.weight = W
        self.variance_epsilon = eps


def ref_rms(X, W, eps, gemma):
    x = X.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if gemma:
        return (x * (1.0 + W.float())).to(X.dtype)
    return W * x.to(X.dtype)


def run(case, dtype, H):
    global torch
    import torch
    rms = import_unsloth("unsloth.kernels.rms_layernorm")
    T, N, gemma, train_w = case["T"], case["N"], case["gemma"], case["train_w"]
    dt = H.dtype
    eps = 1e-6
    g = torch.Generator(device="cuda").manual_seed(0)
    x0 = torch.randn(T, N, device="cuda", dtype=dt, generator=g)
    w0 = (1 + 0.1 * torch.randn(N, device="cuda", generator=g)).to(dt)
    shape = [T, N]
    reuse = not gemma  # decoder pre-norm call sites pass reuse_grad=True

    def mk():
        X = (x0.clone().requires_grad_(True) * 1)  # non-leaf, as a residual stream
        W = w0.clone().requires_grad_(train_w)
        return (X, W) if train_w else (X,)

    Wf = w0.clone()

    def ours(X, W=None):
        return rms.fast_rms_layernorm(Norm(W if W is not None else Wf, eps), X, gemma=gemma, reuse_grad=reuse)

    def eager(X, W=None):
        return ref_rms(X, W if W is not None else Wf, eps, gemma)

    comp = torch.compile(ref_rms, dynamic=False)

    def compiled(X, W=None):
        return comp(X, W if W is not None else Wf, eps, gemma)

    fwd_bwd(H, dict(ours=ours, eager=eager, compiled=compiled), mk, shape, host=T <= 16)

    oo, go = grads_of(ours, mk())
    orr, gr = grads_of(eager, mk())
    X64 = x0.double().requires_grad_(True)
    W64 = w0.double().requires_grad_(train_w)
    o64 = ref_rms(X64, W64, eps, gemma) if True else None
    o64 = (X64 * torch.rsqrt(X64.pow(2).mean(-1, keepdim=True) + eps)) * ((1 + W64) if gemma else W64)
    gg = torch.Generator(device="cuda").manual_seed(1234)
    gy = torch.randn(o64.shape, device="cuda", dtype=dt, generator=gg).double()
    g64 = torch.autograd.grad(o64, [X64] + ([W64] if train_w else []), gy)
    H.parity("fwd", oo[0], o64.detach(), orr[0], shape=shape)
    H.parity("bwd", tuple(go), tuple(g64), tuple(gr), shape=shape)

    if T > 16:
        if once("compile"): H.compile_check("fast_rms_layernorm", lambda X: ours(X).float().sum(),
                        [(mk()[0],), ((x0[: T // 2].clone().requires_grad_(True) * 1),)])
        X = x0.clone(); W = w0.clone()
        dY = torch.randn_like(X); dX = torch.empty_like(X)
        cur = rms._settings(N)
        multirow = rms._multirow_checked(X, W, eps, gemma)
        H.emit(kind="info", what="heuristic", settings=list(cur), multirow=list(multirow) if multirow else None)

        def step(settings=None, multirow_arg=False):
            def fn():
                orig = rms._settings
                if settings is not None:
                    rms._settings = lambda n: settings
                try:
                    Y, r = rms._rms_forward(X, W, eps, gemma, rms._eager_kernel, multirow_arg)
                    rms._rms_backward(dY, dX, X, W, r, eps, gemma, rms._eager_kernel, multirow_arg, train_w=False)
                finally:
                    rms._settings = orig
            return fn
        cfgs = []
        if multirow:
            cfgs.append(("multirow_default%s" % (list(multirow),), step(None, multirow), True))
            cfgs.append(("onerow_w%d" % cur[1], step(None, False), False))
            for rows in (16, 64):
                mr = (multirow[0], rows, multirow[2], multirow[3])
                cfgs.append(("multirow_rows%d" % rows, step(None, mr), False))
        else:
            for w in (2, 4, 8, 16, 32):
                if w * 32 > cur[0] * 2 and w != cur[1]:
                    continue
                cfgs.append(("onerow_B%d_w%d" % (cur[0], w), step((cur[0], w), False), w == cur[1]))
        config_sweep(H, "rms_fwd+bwd_kernels", cfgs, shape)


if __name__ == "__main__":
    main(CASES, run)
