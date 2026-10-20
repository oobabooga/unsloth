"""NVFP4 W4A16 (compressed-tensors nvfp4-pack-quantized) kernels in unsloth/kernels/nvfp4.py:
  nvfp4_dequantize -> _nvfp4_dequant_kernel (fast_dequantize for NVFP4QuantState: LoRA training + prefill)
  nvfp4_linear (NVFP4Linear_matmul) fwd -> _nvfp4_matmul_fwd_kernel (+ _split_sum_kernel) when _fused_ok,
      else dequant + cuBLAS; bwd dX -> _nvfp4_matmul_dx_kernel or dequant + cuBLAS.
  The fused gate _FUSED_ROW_GATE covers sm80 / sm89 / sm120 only (T4 / H100 / B200 always dense).
W4A16, so every GPU can run it (bf16 needs sm80; T4 runs fp16)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *

SHAPES = dict(q=(4096, 4096), up=(14336, 4096), down=(4096, 14336))
CASES = (
    [dict(name="dequant_" + k, kind="dequant", N=n, K=kk) for k, (n, kk) in SHAPES.items()]
    + [dict(name="linear_M%d_%s" % (m, k), kind="linear", M=m, N=n, K=kk)
       for m in (1, 8, 32, 128, 4096) for k, (n, kk) in SHAPES.items()]
    + [dict(name="tiny_linear_M1_256x256", kind="linear", M=1, N=256, K=256, host=True)]
    + [dict(name="sweep_fused_M32_up", kind="sweep", M=32, N=14336, K=4096),
       dict(name="sweep_fused_M128_q", kind="sweep", M=128, N=4096, K=4096)]
)

_M = {}


def mods():
    if not _M:
        import_zoo("unsloth_zoo.utils", stub=("unsloth_zoo", "unsloth_zoo.temporary_patches"))
        _M["nv"] = import_unsloth("unsloth.kernels.nvfp4")
    return _M["nv"]


def make(N, K, dtype):
    import torch
    g = torch.Generator(device="cuda").manual_seed(N * 7 + K)
    packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8, generator=g)
    scale = (torch.rand(N, K // 16, device="cuda", generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)
    gs = torch.tensor([200.0], device="cuda")
    return packed, scale, gs


def set_mode(nv, mode):
    nv._FUSED_ENV = (mode, None)


def run(case, dtype, H):
    import torch
    nv = mods()
    dt = H.dtype
    N, K = case["N"], case["K"]
    packed, scale, gs = make(N, K, dt)
    W32 = nv._nvfp4_dequantize_torch(packed, scale, gs, torch.float32)
    Wd = W32.to(dt)
    shp = [N, K]
    cap = torch.cuda.get_device_capability(0)
    if case["kind"] == "dequant":
        tq = lambda: nv._nvfp4_dequantize_torch(packed, scale, gs, dt)
        cq = torch.compile(tq, dynamic=False)
        H.interleave(dict(ours=lambda: nv.nvfp4_dequantize(packed, scale, gs, dt), compiled=cq, eager=tq), "fwd", shape=shp, budget_ms=150)
        o = nv.nvfp4_dequantize(packed, scale, gs, dt)
        H.emit(kind="exact", what="dequant_vs_torch_ref", mismatches=int((o != Wd).sum().item()), numel=o.numel())
        return
    M = case["M"]
    X = torch.randn(M, K, device="cuda", dtype=dt)
    dY = torch.randn(M, N, device="cuda", dtype=dt)
    gate_f = nv._fused_max_rows(cap, N, K, False)
    gate_b = nv._fused_max_rows(cap, N, K, True)
    set_mode(nv, "auto")
    auto_f = nv._fused_ok(X, packed, scale, gs, K)
    auto_b = nv._fused_ok(dY, packed, scale, gs, N, dx=True)
    H.emit(kind="info", fused_rows_fwd=gate_f, fused_rows_dx=gate_b, auto_fused_fwd=auto_f, auto_fused_dx=auto_b)
    if case["kind"] == "sweep":
        sweep(H, nv, X, packed, scale, gs, shp)
        return

    ref_comp = torch.compile(lambda x: torch.matmul(x, nv._nvfp4_dequantize_torch(packed, scale, gs, dt).t()), dynamic=False)

    def arm(mode, fn):
        def f():
            set_mode(nv, mode)
            return fn()
        return f

    fwd = lambda: nv._nvfp4_matmul_fwd(X, packed, scale, gs)
    arms = dict(ours=arm("auto", fwd), ours_dense=arm("0", fwd),
                compiled=lambda: ref_comp(X), cublas_dense=lambda: torch.matmul(X, Wd.t()))
    if dt == torch.bfloat16 and cap < (8, 0):
        arms.pop("ours_fused", None)
    else:
        arms["ours_fused"] = arm("1", fwd)
    H.interleave(arms, "fwd", shape=[M] + shp, host=case.get("host", False), budget_ms=150)
    bwd = lambda: nv._nvfp4_matmul_bwd(dY, packed, scale, gs)
    barms = dict(ours=arm("auto", bwd), ours_dense=arm("0", bwd), ours_fused=arm("1", bwd),
                 cublas_dense=lambda: torch.matmul(dY, Wd))
    H.interleave(barms, "bwd", shape=[M] + shp, budget_ms=150)
    set_mode(nv, "1")
    of = nv._nvfp4_matmul_fwd(X, packed, scale, gs)
    ob = nv._nvfp4_matmul_bwd(dY, packed, scale, gs)
    set_mode(nv, "0")
    od = nv._nvfp4_matmul_fwd(X, packed, scale, gs)
    set_mode(nv, "auto")
    r = X.double() @ W32.double().t()
    H.parity("fwd_fused", of, r, od, shape=[M] + shp)
    H.parity("dx_fused", ob, dY.double() @ W32.double(), torch.matmul(dY, Wd), shape=[M] + shp)
    if not case.get("host"):
        Xg = X.clone().requires_grad_()
        H.compile_check("nvfp4_linear", lambda x: nv.nvfp4_linear(x, packed, scale, gs),
                        [(Xg,), (torch.randn(M + 8, K, device="cuda", dtype=dt, requires_grad=True),)])


def sweep(H, nv, X, packed, scale, gs, shp):
    """Fused fwd tile sweep around the _fused_config heuristic (ABA vs default)."""
    import torch
    set_mode(nv, "1")
    f = lambda: nv._nvfp4_matmul_fwd(X, packed, scale, gs)
    base = nv._fused_config
    M, K = X.shape
    N = packed.shape[0]
    d = base(M, N, K, X.device, False)
    # 16 tiles at the default split + split variants of the default tile (T4 compiles each one: keep it small)
    cands = [(bm, bo, br, w, d[4], d[5]) for bm in (16, 32, 64, 128) for (bo, br) in ((64, 128), (128, 64)) for w in (4, 8)]
    cands += [d[:5] + (sp,) for sp in (1, 2, 4, 8) if sp != d[5]]
    d_prev = H.measure(f, budget_ms=30)[0]
    res = []
    for c in cands:
        nv._CONFIG_HOOK = lambda *a, c=c: c
        try:
            t = H.measure(f, budget_ms=20)[0]
        except Exception as ex:
            res.append(dict(cfg=c, err=repr(ex)[:80])); nv._CONFIG_HOOK = base; continue
        nv._CONFIG_HOOK = base
        dd = H.measure(f, budget_ms=20)[0]
        res.append(dict(cfg=c, ms=t, ratio=t / ((dd + d_prev) / 2)))
        d_prev = dd
    nv._CONFIG_HOOK = base
    set_mode(nv, "auto")
    ok = sorted((r for r in res if "ratio" in r), key=lambda r: r["ratio"])
    H.emit(kind="config_sweep", what="nvfp4_fused_fwd", shape=[M] + shp, default_cfg=str(d), default_ms=d_prev,
           best=ok[:5], worst=ok[-1:], n=len(res), best_over_default=ok[0]["ratio"] if ok else None)


if __name__ == "__main__":
    main(CASES, run)
