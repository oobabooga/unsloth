"""Cross entropy: fast_cross_entropy_loss (Fast_CrossEntropyLoss) used by llama.py / mistral.py CausalLM
forward when logits are materialised, and by zoo loss_utils' patched ForCausalLMLoss (all text models,
LoRA / full training, 4-bit and 16-bit). Vocab > 65536 takes the chunked forward."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
from _dense_common import fwd_bwd, config_sweep, once, vram_budget_bytes

CASES = [
    dict(name="llama3_V128256_T4096", T=4096, V=128256),
    dict(name="qwen3_V151936_T4096", T=4096, V=151936),
    dict(name="gemma2_V256000_T2048_softcap30", T=2048, V=256000, softcap=30.0),
    dict(name="mistral_V32768_T4096_onechunk", T=4096, V=32768),
    dict(name="tiny_V128256_T16", T=16, V=128256),
]


def ref_loss(logits, labels, n, softcap=0.0, scaling=0.0):
    x = logits.float()
    if scaling:
        x = x * scaling
    if softcap:
        x = softcap * torch.tanh(x / softcap)
    return torch.nn.functional.cross_entropy(x, labels, ignore_index=-100, reduction="sum") / n


def run(case, dtype, H):
    global torch
    import torch
    ce = import_unsloth("unsloth.kernels.cross_entropy_loss")
    T, V = case["T"], case["V"]
    # three arms' graphs + grads live at once (~22x one bf16 [T, V] logits tensor measured): shrink T to fit
    cap = vram_budget_bytes()
    while T > 16 and 22 * T * V * 2 > cap:
        T //= 2
    if T != case["T"]:
        H.emit(kind="info", what="T reduced to fit VRAM", T=T, T_requested=case["T"])
    sc = case.get("softcap", 0.0)
    dt = H.dtype
    g = torch.Generator(device="cuda").manual_seed(0)
    base = (torch.randn(T, V, device="cuda", generator=g) * 2).to(dt)
    labels = torch.randint(0, V, (T,), device="cuda", generator=g)
    labels[torch.rand(T, device="cuda", generator=g) < 0.1] = -100
    n = (labels != -100).sum()
    shape = [T, V]

    def ours(x):
        return ce.fast_cross_entropy_loss(x.view(1, T, V), labels.view(1, T), logit_softcapping=sc, n_items=n)

    def eager(x):
        return ref_loss(x, labels, n, sc)

    comp = torch.compile(ref_loss, dynamic=False)

    def compiled(x):
        return comp(x, labels, n, sc)

    mk = lambda: (base.clone().requires_grad_(True),)
    fwd_bwd(H, dict(ours=ours, eager=eager, compiled=compiled), mk, shape, host=T <= 16)

    # parity: loss + dlogits vs fp64 on a <=1024-row slice (fp64 [T, V] would not fit a T4)
    P = min(T, 1024)
    xs, ls = base[:P].contiguous(), labels[:P].contiguous()
    ns = (ls != -100).sum()
    ours_p = lambda x: ce.fast_cross_entropy_loss(x.view(1, P, V), ls.view(1, P), logit_softcapping=sc, n_items=ns)
    x64 = xs.double().requires_grad_(True)
    xx = sc * torch.tanh(x64 / sc) if sc else x64
    l64 = torch.nn.functional.cross_entropy(xx, ls, ignore_index=-100, reduction="sum") / ns
    (d64,) = torch.autograd.grad(l64, x64)
    xo = xs.clone().requires_grad_(True); lo = ours_p(xo); (do,) = torch.autograd.grad(lo, xo)
    xr = xs.clone().requires_grad_(True); lr = ref_loss(xr, ls, ns, sc); (dr,) = torch.autograd.grad(lr, xr)
    H.parity("loss", lo.detach(), l64.detach(), lr.detach(), shape=shape)
    H.parity("dlogits", do, d64, dr, shape=shape)
    del x64, d64, do, dr

    if T > 16:
        # graph breaks / recompiles when the loss is called inside a compiled region
        def f(x, lab):
            return ce.fast_cross_entropy_loss(x.view(1, -1, V), lab.view(1, -1), logit_softcapping=sc)
        if once("compile"): H.compile_check("fast_cross_entropy_loss", f,
                        [(base[: T // 2].clone().requires_grad_(True), labels[: T // 2]),
                         (base[: T // 4].clone().requires_grad_(True), labels[: T // 4])])

        # config sweep on the raw kernels
        dev = base.device
        x = base.contiguous()
        losses = torch.empty(T, dtype=torch.float32, device=dev)
        if V > 65536:
            def cfwd(BLOCK, warps):
                nch = triton.cdiv(V, BLOCK)
                lse = torch.empty((T, nch), dtype=torch.float32, device=dev)
                return lambda: ce._chunked_cross_entropy_forward[(T, nch)](
                    x, x.stride(0), losses, lse, labels, VOCAB_SIZE=V, N_CHUNKS=nch, BLOCK_SIZE=BLOCK,
                    DO_SOFTCAPPING=bool(sc), SOFTCAP=sc, DO_LOGIT_SCALING=False, LOGIT_SCALE=0.0,
                    EVICT=ce._chunked_evict_policy(dev), num_warps=warps, enable_fp_fusion=True)
            cfgs = [("B65536_w32", cfwd(65536, 32), True), ("B65536_w16", cfwd(65536, 16), False),
                    ("B32768_w16", cfwd(32768, 16), False), ("B16384_w8", cfwd(16384, 8), False),
                    ("B32768_w32", cfwd(32768, 32), False)]
            config_sweep(H, "chunked_fwd", cfgs, shape)
        else:
            BS, W = ce.calculate_settings(V)
            lse1 = torch.empty(T, dtype=torch.float32, device=dev)
            def sfwd(warps):
                return lambda: ce._cross_entropy_forward[(T,)](
                    x, x.stride(0), losses, lse1, labels, VOCAB_SIZE=V, BLOCK_SIZE=BS,
                    DO_SOFTCAPPING=bool(sc), SOFTCAP=sc, DO_LOGIT_SCALING=False, LOGIT_SCALE=0.0,
                    num_warps=warps, enable_fp_fusion=True)
            cfgs = [("w%d" % w, sfwd(w), w == W) for w in (8, 16, 32)]
            config_sweep(H, "onechunk_fwd", cfgs, shape)
        lse = torch.logsumexp(x.float(), dim=1)
        dl = torch.full((T,), 1.0 / T, device=dev)
        dlog = torch.empty_like(x)

        def cbwd(BLOCK, warps):
            return lambda: ce._cross_entropy_backward[(T, triton.cdiv(V, BLOCK))](
                x, x.stride(0), dlog, dl, dl.stride(0), lse, labels, VOCAB_SIZE=V, BLOCK_SIZE=BLOCK,
                DO_SOFTCAPPING=bool(sc), SOFTCAP=sc, DO_LOGIT_SCALING=False, LOGIT_SCALE=0.0,
                num_warps=warps, enable_fp_fusion=True)
        cfgs = [("B4096_w8", cbwd(4096, 8), True), ("B4096_w4", cbwd(4096, 4), False),
                ("B8192_w8", cbwd(8192, 8), False), ("B8192_w16", cbwd(8192, 16), False),
                ("B2048_w4", cbwd(2048, 4), False)]
        config_sweep(H, "bwd", cfgs, shape)


import triton  # noqa: E402

if __name__ == "__main__":
    main(CASES, run)
