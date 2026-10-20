"""LayerNorm: fast_layernorm (Fast_Layernorm). Exported (kernels/__init__, patch_layernorm) but no model
file in unsloth/models calls fast_layernorm or patch_layernorm on main (cohere uses its own compiled
LayerNorm), so this path is reachable only by user code."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
from _dense_common import fwd_bwd, config_sweep, grads_of, once

CASES = [
    dict(name="h4096_T4096_frozen", T=4096, N=4096, train=False),
    dict(name="h4096_T4096_trainWb", T=4096, N=4096, train=True),
    dict(name="tiny_T16", T=16, N=4096, train=False),
]


class LN:
    def __init__(self, W, b, eps):
        self.weight, self.bias, self.eps = W, b, eps
        self.elementwise_affine = True


def run(case, dtype, H):
    global torch
    import torch
    lnm = import_unsloth("unsloth.kernels.layernorm")
    T, N, train = case["T"], case["N"], case["train"]
    dt = H.dtype
    eps = 1e-5
    g = torch.Generator(device="cuda").manual_seed(0)
    x0 = torch.randn(T, N, device="cuda", dtype=dt, generator=g)
    w0 = (1 + 0.1 * torch.randn(N, device="cuda", generator=g)).to(dt)
    b0 = (0.1 * torch.randn(N, device="cuda", generator=g)).to(dt)
    shape = [T, N]

    def mk():
        X = x0.clone().requires_grad_(True)
        if train:
            return (X, w0.clone().requires_grad_(True), b0.clone().requires_grad_(True))
        return (X,)

    ours = lambda X, W=w0, b=b0: lnm.fast_layernorm(LN(W, b, eps), X)
    ref = lambda X, W=w0, b=b0: torch.nn.functional.layer_norm(X, (N,), W, b, eps)
    comp = torch.compile(lambda X, W, b: torch.nn.functional.layer_norm(X, (N,), W, b, eps), dynamic=False)
    compiled = lambda X, W=w0, b=b0: comp(X, W, b)
    fwd_bwd(H, dict(ours=ours, eager=ref, compiled=compiled), mk, shape, host=T <= 16)

    oo, go = grads_of(ours, mk())
    orr, gr = grads_of(ref, mk())
    X64 = x0.double().requires_grad_(True)
    W64 = w0.double().requires_grad_(train); b64 = b0.double().requires_grad_(train)
    o64 = torch.nn.functional.layer_norm(X64, (N,), W64, b64, eps)
    gg = torch.Generator(device="cuda").manual_seed(1234)
    gy = torch.randn(o64.shape, device="cuda", dtype=dt, generator=gg).double()
    g64 = torch.autograd.grad(o64, [X64] + ([W64, b64] if train else []), gy)
    H.parity("fwd", oo[0], o64.detach(), orr[0], shape=shape)
    H.parity("bwd", tuple(go), tuple(g64), tuple(gr), shape=shape)
    if T > 16:
        if once("compile"): H.compile_check("fast_layernorm", lambda X: ours(X).float().sum(),
                        [(mk()[0],), (x0[: T // 2].clone().requires_grad_(True),)])
        cur = lnm._settings(N)
        X = x0.clone()
        orig = lnm._settings

        def fwd_w(w):
            def fn():
                lnm._settings = lambda n: (cur[0], w)
                try:
                    with torch.no_grad():
                        lnm.Fast_Layernorm.apply(X, w0, b0, eps)
                finally:
                    lnm._settings = orig
            return fn
        config_sweep(H, "ln_fwd_warps", [("B%d_w%d" % (cur[0], w), fwd_w(w), w == cur[1]) for w in (4, 8, 16, 32)], shape)


if __name__ == "__main__":
    main(CASES, run)
