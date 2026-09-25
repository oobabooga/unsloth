"""Portable Triton W8A8: per-token int8 activations x per-output-channel int8 weights, int32 accumulate, dequant epilogue.

Two variants:
  two-kernel  ``quant_rows`` (one read of x, int8 + fp32 scale out) then ``int8_mm`` (dot + dequant + bias epilogue)
  fused       ``int8_mm(FUSED_QUANT=True)`` reads the bf16/fp16 x tile, takes the row absmax in a first K loop and
              quantises on the fly in the second: no int8 activation tensor, no extra launch.
The two-kernel variant matches the torch reference exactly (IEEE divide, round half to even, same epilogue order), so
it is bit-identical to ``torch._int_mm`` on the same operands. The fused one multiplies by the reciprocal scale, which
can move an exact tie by one int8 step.
Weights are ``[N, K]`` int8 row-major (nn.Linear layout).
"""
from __future__ import annotations

import torch

_KERNELS = None


def kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def _rne(y):
        # round half to even from floor: HIP's libdevice lacks rint on some Triton builds (3.6 ROCm)
        f = tl.floor(y)
        d = y - f
        odd = f - 2.0 * tl.floor(f * 0.5)
        return tl.where(d > 0.5, f + 1.0, tl.where(d < 0.5, f, f + odd))

    hip = bool(getattr(torch.version, "hip", None))
    stages = [2] if hip else [3, 4]
    cfgs = []
    for bm, bn, bk, w in ((128, 128, 64, 8), (128, 64, 64, 4), (64, 128, 64, 4), (64, 64, 128, 4), (128, 128, 128, 8),
                          (128, 256, 64, 8), (256, 128, 64, 8)):
        for s in stages:
            cfgs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8}, num_warps = w,
                                      num_stages = s))

    @triton.jit
    def quant_rows(x_ptr, q_ptr, s_ptr, K, stride_xm, BLOCK_K: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        amax = tl.zeros([BLOCK_K], dtype = tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + row * stride_xm + offs, mask = offs < K, other = 0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.abs(x))
        s = tl.maximum(tl.max(amax, 0), 1e-12) / 127.0
        tl.store(s_ptr + row, s)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + row * stride_xm + offs, mask = offs < K, other = 0.0).to(tl.float32)
            q = _rne(tl.div_rn(x, s))
            q = tl.minimum(tl.maximum(q, -127.0), 127.0)
            tl.store(q_ptr + row * K + offs, q.to(tl.int8), mask = offs < K)

    @triton.autotune(configs = cfgs, key = ["M", "N", "K", "FUSED_QUANT"])
    @triton.jit
    def int8_mm(a_ptr, b_ptr, c_ptr, xs_ptr, ws_ptr, bias_ptr, M, N, K, stride_am, stride_ak, stride_bn, stride_bk,
                stride_cm, stride_cn, HAS_BIAS: tl.constexpr, FUSED_QUANT: tl.constexpr, BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * num_pid_n
        group_id = pid // width
        first_m = group_id * GROUP_M
        gsize = tl.minimum(num_pid_m - first_m, GROUP_M)
        pid_m = first_m + (pid % gsize)
        pid_n = (pid % width) // gsize
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        rm64 = rm.to(tl.int64)
        rn64 = rn.to(tl.int64)
        if FUSED_QUANT:
            amax = tl.zeros([BLOCK_M], dtype = tl.float32)
            for k0 in range(0, K, BLOCK_K):
                ka = k0 + rk
                a = tl.load(a_ptr + rm64[:, None] * stride_am + ka[None, :] * stride_ak,
                            mask = (rm[:, None] < M) & (ka[None, :] < K), other = 0.0).to(tl.float32)
                amax = tl.maximum(amax, tl.max(tl.abs(a), 1))
            xs = tl.maximum(amax, 1e-12) / 127.0
            inv = tl.div_rn(tl.full([BLOCK_M], 1.0, tl.float32), xs)
        else:
            xs = tl.load(xs_ptr + rm, mask = rm < M, other = 0.0)
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype = tl.int32)
        for k0 in range(0, K, BLOCK_K):
            ka = k0 + rk
            a = tl.load(a_ptr + rm64[:, None] * stride_am + ka[None, :] * stride_ak,
                        mask = (rm[:, None] < M) & (ka[None, :] < K), other = 0)
            if FUSED_QUANT:
                # multiply by the reciprocal: an IEEE divide per element per N tile costs ~15x the whole GEMM
                q = _rne(a.to(tl.float32) * inv[:, None])
                a = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
            b = tl.load(b_ptr + rn64[None, :] * stride_bn + ka[:, None] * stride_bk,
                        mask = (rn[None, :] < N) & (ka[:, None] < K), other = 0)
            acc += tl.dot(a, b, out_dtype = tl.int32)
        ws = tl.load(ws_ptr + rn, mask = rn < N, other = 0.0)
        out = acc.to(tl.float32) * xs[:, None] * ws[None, :]
        if HAS_BIAS:
            out = out + tl.load(bias_ptr + rn, mask = rn < N, other = 0.0).to(tl.float32)[None, :]
        tl.store(c_ptr + rm64[:, None] * stride_cm + rn64[None, :] * stride_cn, out.to(c_ptr.dtype.element_ty),
                 mask = (rm[:, None] < M) & (rn[None, :] < N))

    import types

    _KERNELS = types.SimpleNamespace(quant_rows = quant_rows, int8_mm = int8_mm, triton = triton)
    return _KERNELS


def quantize_weight(w: torch.Tensor):
    """Per-output-channel symmetric int8 of ``w [N, K]`` in float32: (int8 [N, K], fp32 scale [N])."""
    wf = w.float()
    ws = wf.abs().amax(dim = 1).clamp(min = 1e-12) / 127.0
    wq = (wf / ws[:, None]).round().clamp(-127, 127).to(torch.int8)
    return wq.contiguous(), ws.contiguous()


def w8a8_torch(x2, wq, ws, bias, out_dtype):
    """The torch-op reference W8A8 (Studio's torch fallback): per-token quant, torch._int_mm, float epilogue."""
    xf = x2.float()
    xs = xf.abs().amax(dim = 1).clamp(min = 1e-12) / 127.0
    xq = (xf / xs[:, None]).round().clamp(-127, 127).to(torch.int8)
    y = torch._int_mm(xq, wq.t()).float() * xs[:, None] * ws[None, :]
    if bias is not None:
        y = y + bias.float()
    return y.to(out_dtype)


def w8a8_triton(x2, wq, ws, bias, out_dtype, fused: bool):
    k = kernels()
    M, K = x2.shape
    N = wq.shape[0]
    out = torch.empty((M, N), dtype = out_dtype, device = x2.device)
    if fused:
        a, xs = x2, ws  # xs unused
    else:
        a = torch.empty((M, K), dtype = torch.int8, device = x2.device)
        xs = torch.empty(M, dtype = torch.float32, device = x2.device)
        k.quant_rows[(M,)](x2, a, xs, K, x2.stride(0), BLOCK_K = 1024, num_warps = 4)
    grid = lambda meta: (k.triton.cdiv(M, meta["BLOCK_M"]) * k.triton.cdiv(N, meta["BLOCK_N"]),)  # noqa: E731
    k.int8_mm[grid](a, wq, out, xs, ws, bias if bias is not None else ws, M, N, K, a.stride(0), a.stride(1),
                    wq.stride(0), wq.stride(1), out.stride(0), out.stride(1), HAS_BIAS = bias is not None,
                    FUSED_QUANT = fused)
    return out
