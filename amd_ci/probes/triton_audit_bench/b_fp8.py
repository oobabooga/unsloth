"""FP8 kernels in unsloth/kernels/fp8.py (FP8 checkpoints: Qwen3-*-FP8, DeepSeek-style 128x128 block FP8,
fbgemm/compressed-tensors per-row FP8; LoRA training + inference):
  weight_dequant_kernel   <- weight_dequant / _blockwise_weight_dequant_any_shape (fast_dequantize, bwd dX, <sm89 fwd)
  act_quant_kernel        <- FP8BlockQuantLinear.forward (fp8_linear with 2-D block scale), sm89+
  _w8a8_block_fp8_matmul  <- fp8_block_matmul when torchao blockwise_fp8_gemm is missing (or > 2**31 elems)
  _fp8_rowwise_gemv_kernel<- fast_linear_forward decode (bsz=1,q=1) with per-row scale, sm89+
References: dequant + cuBLAS (the <sm89 path), torchao blockwise_fp8_gemm (Triton), torch._scaled_mm rowwise
(tensor-core FP8 speed of light, different scaling), dense 16-bit cuBLAS."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *

SHAPES = dict(q=(4096, 4096), up=(14336, 4096), down=(4096, 14336))
CASES = (
    [dict(name="dequant_block_" + k, kind="dequant", N=n, K=kk) for k, (n, kk) in SHAPES.items()]
    + [dict(name="block_linear_M%d_%s" % (m, k), kind="block", M=m, N=n, K=kk, min_cap=(8, 9))
       for m in (1, 16, 4096) for k, (n, kk) in SHAPES.items()]
    + [dict(name="rowwise_gemv_M1_" + k, kind="gemv", N=n, K=kk, min_cap=(8, 9)) for k, (n, kk) in SHAPES.items()]
    + [dict(name="tiny_rowwise_gemv_256", kind="gemv", N=256, K=256, min_cap=(8, 9), host=True)]
)

_M = {}


def mods():
    if not _M:
        import_zoo("unsloth_zoo.utils", stub=("unsloth_zoo", "unsloth_zoo.temporary_patches"))
        _M["f"] = import_unsloth("unsloth.kernels.fp8")
    return _M["f"]


def block_weight(N, K, dt):
    import torch
    W = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.02
    Wb = W.view(N // 128, 128, K // 128, 128)
    s = Wb.abs().amax(dim=(1, 3)) / 448.0
    q = (Wb / s[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).view(N, K)
    return q, s.contiguous()


def run(case, dtype, H):
    import torch
    f = mods()
    dt = H.dtype
    N, K = case["N"], case["K"]
    shp = [N, K]
    kind = case["kind"]
    H.emit(kind="info", torchao_blockwise=f.torchao_blockwise_gemm is not None,
           block_impl=getattr(f.fp8_block_quant_linear, "__name__", str(f.fp8_block_quant_linear)),
           fp8_block_matmul=getattr(f.fp8_block_matmul, "__name__", "?"))
    if kind == "dequant":
        q, s = block_weight(N, K, dt)
        ref = lambda: f._torch_blockwise_dequant(q, s, [128, 128], dt)
        plain = lambda: (q.view(N // 128, 128, K // 128, 128).float() * s[:, None, :, None]).to(dt).view(N, K)
        comp = torch.compile(plain, dynamic=False)
        H.interleave(dict(ours=lambda: f.weight_dequant(q, s, dt), triton_kernel=lambda: f.weight_dequant_block(q, s, 128, dt),
                          torch_chunked=ref, eager=plain, compiled=comp), "fwd", shape=shp, budget_ms=150)
        o = f.weight_dequant(q, s, dt)
        H.emit(kind="exact", what="dequant_vs_torch", mismatches=int((o != plain()).sum().item()), numel=o.numel(),
               path_triton=not f._fp8_kernel_unsupported(q))
        return
    if kind == "block":
        if dt == torch.float16:
            # act_quant/ block matmul are bf16 models in practice; fp16 still runs, keep it
            pass
        M = case["M"]
        q, s = block_weight(N, K, dt)
        X = torch.randn(1, M, K, device="cuda", dtype=dt)
        Wd = f.weight_dequant(q, s, dt)
        xq, xs = f.act_quant(X, 128)
        arms = dict(
            ours=lambda: f.fp8_linear(X, q, s),
            act_quant=lambda: f.act_quant(X, 128),
            triton_block_mm=lambda: f.w8a8_block_fp8_matmul_triton(xq, q, xs, s, [128, 128], output_dtype=dt),
            dequant_cublas=lambda: torch.matmul(X, f.weight_dequant(q, s, dt).t()),
            cublas_dense=lambda: torch.matmul(X, Wd.t()),
        )
        if f.torchao_blockwise_gemm is not None:
            arms["torchao_block_mm"] = lambda: f.torchao_block_matmul(xq, q, xs, s, [128, 128], output_dtype=dt)
        # tensor-core FP8 speed of light: rowwise _scaled_mm on the same fp8 weight
        try:
            X2 = X.view(M, K)
            xr, xsr = f._quantize_fp8_per_row(X2)
            sw = torch.ones(1, N, device="cuda")
            Mp = max(16, M)
            if M % 16:
                xr = torch.nn.functional.pad(xr.view(torch.uint8), (0, 0, 0, Mp - M)).view(torch.float8_e4m3fn)
                xsr = torch.nn.functional.pad(xsr, (0, 0, 0, Mp - M), value=1.0)
            torch._scaled_mm(xr, q.t(), scale_a=xsr, scale_b=sw, out_dtype=torch.bfloat16)
            arms["scaled_mm_rowwise"] = lambda: torch._scaled_mm(xr, q.t(), scale_a=xsr, scale_b=sw, out_dtype=torch.bfloat16)
        except Exception as ex:
            H.emit(kind="info", scaled_mm_err=repr(ex)[:200])
        H.interleave(arms, "fwd", shape=[M] + shp, budget_ms=150)
        dY = torch.randn(1, M, N, device="cuda", dtype=dt)
        H.interleave(dict(
            ours=lambda: torch.matmul(dY, f._blockwise_weight_dequant_any_shape(q, s, [128, 128], dt)),  # FP8BlockQuantLinear.backward
            cublas_dense=lambda: torch.matmul(dY, Wd)), "bwd", shape=[M] + shp, budget_ms=150)
        o = f.fp8_linear(X, q, s)
        W32 = q.view(N // 128, 128, K // 128, 128).double() * s.double()[:, None, :, None]
        r = X.double() @ W32.view(N, K).t()
        H.parity("fp8_linear_vs_fp64_deqW", o, r, torch.matmul(X, Wd.t()), shape=[M] + shp)
        if M > 1:
            H.compile_check("fp8_linear", lambda x: f.fp8_linear(x, q, s),
                            [(X,), (torch.randn(1, M + 128, K, device="cuda", dtype=dt),)])
        return
    if kind == "gemv":
        W = torch.randn(N, K, device="cuda") * 0.02
        sc = (W.abs().amax(1, keepdim=True) / 448.0)
        q = (W / sc).to(torch.float8_e4m3fn)
        X = torch.randn(1, 1, K, device="cuda", dtype=dt)
        Wd = (q.float() * sc).to(dt)
        ok = f.can_use_fp8_rowwise_gemv(X, q, sc)
        H.emit(kind="info", can_use_rowwise_gemv=ok, rowwise_backend=str(f._probe_fp8_rowwise_backend(X.device)))
        arms = dict(ours=lambda: f.fp8_rowwise_gemv(X, q, sc),
                    dequant_cublas=lambda: torch.matmul(X, (q.to(dt) * sc.to(dt)).t()),
                    cublas_dense=lambda: torch.matmul(X, Wd.t()),
                    fp8_linear_rowwise=lambda: f.fp8_linear(X, q, sc))
        H.interleave(arms, "fwd", shape=[1] + shp, budget_ms=150, host=case.get("host", False))
        o = f.fp8_rowwise_gemv(X, q, sc)
        H.parity("rowwise_gemv_vs_fp64", o, X.double() @ (q.double() * sc.double()).t(), torch.matmul(X, Wd.t()), shape=[1] + shp)
        # config sweep of the fixed (BLOCK_N, BLOCK_K, warps) heuristic
        if not case.get("host"):
            sweep_gemv(H, f, X, q, sc, shp)
            H.compile_check("fp8_rowwise_gemv", lambda x: f.fp8_rowwise_gemv(x, q, sc), [(X,), (torch.randn(1, 1, K, device="cuda", dtype=dt),)])


def sweep_gemv(H, f, X, q, sc, shp):
    import torch, triton
    N, K = q.shape
    out = torch.empty((1, 1, N), device="cuda", dtype=X.dtype)

    def launch(bn, bk, w):
        def g():
            f._fp8_rowwise_gemv_kernel[(triton.cdiv(N, bn),)](X, q, sc, out, N, K, q.stride(0), BLOCK_N=bn, BLOCK_K=bk,
                                                              LONG_INDEXING=False, num_warps=w, enable_fp_fusion=True)
        return g
    d = launch(1 if N <= 4096 else (4 if N <= 32768 else 8), 512, 2)
    d_prev = H.measure(d, budget_ms=20)[0]
    res = []
    for bn in (1, 2, 4, 8):
        for bk in (256, 512, 1024):
            for w in (1, 2, 4):
                try:
                    t = H.measure(launch(bn, bk, w), budget_ms=15)[0]
                except Exception as ex:
                    res.append(dict(cfg=(bn, bk, w), err=repr(ex)[:60])); continue
                dd = H.measure(d, budget_ms=15)[0]
                res.append(dict(cfg=(bn, bk, w), ms=t, ratio=t / ((dd + d_prev) / 2)))
                d_prev = dd
    ok = sorted((r for r in res if "ratio" in r), key=lambda r: r["ratio"])
    H.emit(kind="config_sweep", what="fp8_rowwise_gemv", shape=shp,
           default_cfg=str((1 if N <= 4096 else (4 if N <= 32768 else 8), 512, 2)), default_ms=d_prev,
           best=ok[:5], worst=ok[-1:], n=len(res), best_over_default=ok[0]["ratio"] if ok else None)


if __name__ == "__main__":
    main(CASES, run)
