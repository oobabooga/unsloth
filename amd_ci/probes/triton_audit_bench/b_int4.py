"""compressed-tensors pack-quantized INT4 (W4A16, e.g. RedHat / neuralmagic *-quantized.w4a16 checkpoints) kernels in
unsloth/kernels/int4_packed.py, used via models/compressed_tensors_int4.py + utils.fast_dequantize / matmul_lora /
fast_linear_forward (Int4QuantState):
  _dequant_kernel           <- int4_dequantize (training: every LoRA fwd/bwd; prefill M > 1)
  _gemv_kernel              <- int4_matmul M == 1 (checkpoint layout, no repack: pre-sm80, fp16 w/o marlin, act-order)
  _dequant_repacked_kernel  <- int4_dequantize after int4_repack_ (tinygemm / marlin layout, sm80+)
  fast call (tinygemm _weight_int4pack_mm / vLLM marlin) for M <= _fast_rows after repack (not Triton)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *

SHAPES = dict(q=(4096, 4096), up=(14336, 4096), down=(4096, 14336))
CASES = (
    [dict(name="dequant_" + k, kind="dequant", N=n, K=kk) for k, (n, kk) in SHAPES.items()]
    + [dict(name="gemv_M1_" + k, kind="gemv", N=n, K=kk) for k, (n, kk) in SHAPES.items()]
    + [dict(name="repacked_M%d_%s" % (m, k), kind="repacked", M=m, N=n, K=kk, min_cap=(8, 0))
       for m in (1, 8, 32) for k, (n, kk) in (("q", SHAPES["q"]), ("up", SHAPES["up"]))]
    + [dict(name="prefill_M4096_" + k, kind="prefill", M=4096, N=n, K=kk) for k, (n, kk) in SHAPES.items()]
    + [dict(name="tiny_gemv_M1_256", kind="gemv", N=256, K=256, host=True)]
)
GROUP = 128
_M = {}


def mods():
    if not _M:
        import_zoo("unsloth_zoo.utils", stub=("unsloth_zoo", "unsloth_zoo.temporary_patches"))
        _M["i"] = import_unsloth("unsloth.kernels.int4_packed")
    return _M["i"]


def make(i4, N, K, dt):
    import torch
    q = torch.randint(0, 16, (N, K), device="cuda", dtype=torch.int32)
    packed = torch.zeros(N, K // 8, device="cuda", dtype=torch.int32)
    qv = q.view(N, K // 8, 8)
    for j in range(8):
        packed |= qv[:, :, j] << (4 * j)
    scale = (torch.rand(N, K // GROUP, device="cuda") * 0.01 + 0.001).to(dt)
    qs = i4.Int4QuantState(scale, None, None, (N, K), 4, GROUP, dt)
    return packed, qs


def torch_deq(packed, scale, N, K, dt):
    import torch
    sh = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    q = ((packed[:, :, None] >> sh) & 15).view(N, K)
    w = (q - 8).float().view(N, K // GROUP, GROUP) * scale.float()[:, :, None]
    return w.to(scale.dtype).to(dt).view(N, K)


def run(case, dtype, H):
    import torch
    i4 = mods()
    dt = H.dtype
    N, K = case["N"], case["K"]
    shp = [N, K]
    packed, qs = make(i4, N, K, dt)
    Wd = torch_deq(packed, qs.scale, N, K, dt)
    W64 = ((((packed[:, :, None] >> torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)) & 15).view(N, K) - 8).double()
           * qs.scale.double().repeat_interleave(GROUP, 1))
    kind = case["kind"]
    if kind == "dequant":
        e = lambda: torch_deq(packed, qs.scale, N, K, dt)
        c = torch.compile(e, dynamic=False)
        H.interleave(dict(ours=lambda: i4.int4_dequantize(packed, qs, dt), compiled=c, eager=e), "fwd", shape=shp, budget_ms=150)
        o = i4.int4_dequantize(packed, qs, dt)
        H.emit(kind="exact", what="dequant_vs_torch", mismatches=int((o != Wd).sum().item()), numel=o.numel())
        H.compile_check("int4_dequantize_matmul", lambda x: torch.matmul(x, i4.int4_dequantize(packed, qs, dt).t()),
                        [(torch.randn(64, K, device="cuda", dtype=dt),), (torch.randn(128, K, device="cuda", dtype=dt),)])
    elif kind == "gemv":
        X = torch.randn(1, 1, K, device="cuda", dtype=dt)
        H.interleave(dict(ours=lambda: i4.int4_matmul(X, packed, qs),
                          dequant_cublas=lambda: torch.matmul(X, i4.int4_dequantize(packed, qs, dt).t()),
                          cublas_dense=lambda: torch.matmul(X, Wd.t())), "fwd", shape=[1] + shp, budget_ms=150,
                     host=case.get("host", False))
        o = i4.int4_matmul(X, packed, qs)
        H.parity("gemv_vs_fp64", o, X.double() @ W64.t(), torch.matmul(X, Wd.t()), shape=[1] + shp)
        if not case.get("host"):
            sweep_gemv(H, i4, X, packed, qs, shp)
    elif kind == "repacked":
        M = case["M"]
        X = torch.randn(M, K, device="cuda", dtype=dt)
        p2 = packed.clone()
        layout = i4.int4_repack_(p2, qs)
        H.emit(kind="info", layout=layout, fast_rows=i4._fast_rows(layout, 0) if layout else None)
        if layout is None:
            H.emit(kind="skip", why="no repack layout for dtype %s" % dtype)
            return
        with torch.no_grad():
            H.interleave(dict(ours=lambda: i4.int4_matmul(X, p2, qs),
                              dequant_cublas=lambda: torch.matmul(X, i4.int4_dequantize(p2, qs, dt).t()),
                              ours_dequant_only=lambda: i4.int4_dequantize(p2, qs, dt),
                              cublas_dense=lambda: torch.matmul(X, Wd.t())), "fwd", shape=[M] + shp, budget_ms=150)
            o = i4.int4_matmul(X, p2, qs)
        H.parity("repacked_vs_fp64", o, X.double() @ W64.t(), torch.matmul(X, Wd.t()), shape=[M] + shp)
        qs.layout = None; qs._launch = qs._launchers = qs._fast = None
    elif kind == "prefill":
        M = case["M"]
        X = torch.randn(M, K, device="cuda", dtype=dt)
        dY = torch.randn(M, N, device="cuda", dtype=dt)
        H.interleave(dict(ours=lambda: i4.int4_matmul(X, packed, qs, fast=False),
                          cublas_dense=lambda: torch.matmul(X, Wd.t())), "fwd", shape=[M] + shp, budget_ms=150)
        H.interleave(dict(ours=lambda: i4.int4_matmul_t(dY, packed, qs),
                          cublas_dense=lambda: torch.matmul(dY, Wd)), "bwd", shape=[M] + shp, budget_ms=150)


def sweep_gemv(H, i4, X, packed, qs, shp):
    """_gemv_kernel tile sweep (BLOCK_N, BLOCK_KP, warps, split) vs the fixed (8, 64, 4, auto split) launch."""
    import torch, triton
    (P, S, Z, G), (N, K, KP), (sp, ss, sz), meta = i4._launch_args(packed, qs)
    x2 = X.view(1, K)
    sms = i4._sm_count(0)

    def launch(bn, bkp, w, split):
        y = torch.zeros((1, N), dtype=torch.float32 if split > 1 else X.dtype, device="cuda")
        def g():
            if split > 1:
                y.zero_()
            i4._gemv_kernel[(triton.cdiv(N, bn), split, 1)](x2, P, S, Z, G, y, 1, N, K, KP, x2.stride(0), sp, ss, sz,
                                                            y.stride(0), BLOCK_N=bn, BLOCK_KP=bkp, SPLIT_K=split,
                                                            num_warps=w, enable_fp_fusion=True, **meta)
        return g
    nb = triton.cdiv(N, 8)
    dsplit = 1 if nb >= sms else min(4, triton.cdiv(sms, nb))
    d = launch(8, 64, 4, dsplit)
    d_prev = H.measure(d, budget_ms=20)[0]
    res = []
    for bn in (4, 8, 16, 32):
        for bkp in (32, 64, 128):
            for w in (2, 4, 8):
                for split in sorted({1, dsplit}):
                    try:
                        t = H.measure(launch(bn, bkp, w, split), budget_ms=15)[0]
                    except Exception as ex:
                        res.append(dict(cfg=(bn, bkp, w, split), err=repr(ex)[:60])); continue
                    dd = H.measure(d, budget_ms=15)[0]
                    res.append(dict(cfg=(bn, bkp, w, split), ms=t, ratio=t / ((dd + d_prev) / 2)))
                    d_prev = dd
    ok = sorted((r for r in res if "ratio" in r), key=lambda r: r["ratio"])
    H.emit(kind="config_sweep", what="int4_gemv", shape=shp, default_cfg=str((8, 64, 4, dsplit)), default_ms=d_prev,
           best=ok[:5], worst=ok[-1:], n=len(res), best_over_default=ok[0]["ratio"] if ok else None)


if __name__ == "__main__":
    main(CASES, run)
