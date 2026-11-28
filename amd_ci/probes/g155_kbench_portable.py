"""G155: rotated int8 LoRA GEMM variants on sm75 / sm8x (standalone: torch + triton only).

AMD CI copy (gfx1151 / ROCm) of scripts/g155/kbench_portable.py. Changes versus the original:
  * records torch.version.hip and the gcnArchName;
  * on HIP the "skip prefetch" variants are dropped (the patched pass is NVIDIA ttgpuir; on AMD the
    variant would only re-time the stock kernel and double the sweep);
  * the reference fwd / dgrad outputs come from the first tile that launches (128x128 tiles with
    3-4 stages can exceed RDNA's 64 KB LDS and would otherwise abort the whole run);
  * torch._int_mm-only timings and each (shape, M) cell are guarded, so one failure is recorded
    in the JSON instead of killing the sweep.

Variants per (N, K, M): bf16 matmul; Triton int8 fwd / dgrad over tiles, stock and with Triton's sm8x prefetch pass
skipped for these kernels only; torch._int_mm + one fused Triton epilogue (fwd: dequant + delta, dgrad: dequant +
128-block un-rotation + delta); activation quant two-pass (current) vs one-pass.

python kbench_sm8x.py --out out.json [--models qwen3_4b,qwen3_8b] [--ms 4096,8192]
"""
import argparse, json, math, os, sys, threading, time, traceback
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

ap = argparse.ArgumentParser()
ap.add_argument("--out", required = True)
ap.add_argument("--models", default = "qwen3_1.7b,qwen3_4b,qwen3_8b")
ap.add_argument("--ms", default = "4096,8192")
ap.add_argument("--quick", action = "store_true")
args = ap.parse_args()

dev = torch.device("cuda")
cap = torch.cuda.get_device_capability()
DT = torch.bfloat16 if torch.cuda.is_bf16_supported() and cap >= (8, 0) else torch.float16
G = 128

# ---- prefetch skip, scoped to kernels launched under _SKIP.on (and keyed by the PF constexpr) ----
_SKIP = threading.local()
PREFETCH_PATCH = "absent"
try:
    from triton._C.libtriton import passes as _passes
    _orig_pf = _passes.ttgpuir.add_prefetch

    def _pf(*a, **k):
        if getattr(_SKIP, "on", False):
            return None
        return _orig_pf(*a, **k)

    _passes.ttgpuir.add_prefetch = _pf
    PREFETCH_PATCH = "installed"
except Exception as e:  # noqa
    PREFETCH_PATCH = f"failed: {e}"


@triton.jit
def _rint(x):
    # HIP libdevice has no rint in every Triton build: round half away from zero (timing probe only;
    # codes may differ from rint at exact ties)
    return tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit(do_not_specialize = ["M"])
def mm_kernel(a_ptr, b_ptr, as_ptr, bs_ptr, d_ptr, c_ptr, M, N, K, stride_d,
              HAS_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              GROUP_M: tl.constexpr, PF: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + rn[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype = tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    xs = tl.load(as_ptr + rm)
    ws = tl.load(bs_ptr + rn)
    y = acc.to(tl.float32) * xs[:, None] * ws[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if HAS_D:
        y += tl.load(d_ptr + offs_m[:, None] * stride_d + offs_n[None, :], mask = mask, other = 0.0).to(tl.float32)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], y.to(c_ptr.dtype.element_ty), mask = mask)


@triton.jit(do_not_specialize = ["M"])
def dgrad_kernel(a_ptr, b_ptr, as_ptr, h_ptr, d_ptr, c_ptr, M, N, K, stride_d,
                 HAS_D: tl.constexpr, INV_SQRT_G: tl.constexpr, G: tl.constexpr, BLOCK_M: tl.constexpr,
                 BLOCK_R: tl.constexpr, GROUP_M: tl.constexpr, PF: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(K, G)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_n.to(tl.int64) * G + tl.arange(0, G)
    rm = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_r = tl.arange(0, BLOCK_R)
    acc = tl.zeros((BLOCK_M, G), dtype = tl.int32)
    for r0 in range(0, N, BLOCK_R):
        rmask = r0 + offs_r < N
        a = tl.load(a_ptr + rm[:, None] * N + (r0 + offs_r)[None, :], mask = rmask[None, :], other = 0)
        b = tl.load(b_ptr + offs_k[None, :] * N + (r0 + offs_r)[:, None], mask = rmask[:, None], other = 0)
        acc = tl.dot(a, b, acc, out_dtype = tl.int32)
    y = acc.to(tl.float32) * tl.load(as_ptr + rm)[:, None]
    g = tl.arange(0, G)
    H = tl.load(h_ptr + g[:, None] * G + g[None, :])
    hi = y.to(tl.bfloat16)
    lo = (y - hi.to(tl.float32)).to(tl.bfloat16)
    z = (tl.dot(hi, H) + tl.dot(lo, H)) * INV_SQRT_G
    mask = offs_m[:, None] < M
    if HAS_D:
        z += tl.load(d_ptr + offs_m[:, None] * stride_d + offs_k[None, :], mask = mask, other = 0.0).to(tl.float32)
    tl.store(c_ptr + offs_m[:, None] * K + offs_k[None, :], z.to(c_ptr.dtype.element_ty), mask = mask)


@triton.jit(do_not_specialize = ["M"])
def epi_fwd_kernel(acc_ptr, as_ptr, bs_ptr, d_ptr, c_ptr, M, N, stride_d,
                   HAS_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pm = tl.program_id(0).to(tl.int64)
    pn = tl.program_id(1).to(tl.int64)
    offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.load(acc_ptr + offs_m[:, None] * N + offs_n[None, :], mask = mask, other = 0)
    y = acc.to(tl.float32) * tl.load(as_ptr + offs_m, mask = offs_m < M, other = 0.0)[:, None] \
        * tl.load(bs_ptr + offs_n, mask = offs_n < N, other = 0.0)[None, :]
    if HAS_D:
        y += tl.load(d_ptr + offs_m[:, None] * stride_d + offs_n[None, :], mask = mask, other = 0.0).to(tl.float32)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], y.to(c_ptr.dtype.element_ty), mask = mask)


@triton.jit(do_not_specialize = ["M"])
def epi_dgrad_kernel(acc_ptr, as_ptr, h_ptr, d_ptr, c_ptr, M, K, stride_d,
                     HAS_D: tl.constexpr, INV_SQRT_G: tl.constexpr, G: tl.constexpr, BLOCK_M: tl.constexpr,
                     IEEE: tl.constexpr):
    pm = tl.program_id(0).to(tl.int64)
    pk = tl.program_id(1).to(tl.int64)
    offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pk * G + tl.arange(0, G)
    mask = offs_m[:, None] < M
    acc = tl.load(acc_ptr + offs_m[:, None] * K + offs_k[None, :], mask = mask, other = 0)
    y = acc.to(tl.float32) * tl.load(as_ptr + offs_m, mask = offs_m < M, other = 0.0)[:, None]
    g = tl.arange(0, G)
    if IEEE:
        H = tl.load(h_ptr + g[:, None] * G + g[None, :]).to(tl.float32)
        z = tl.dot(y, H, input_precision = "ieee") * INV_SQRT_G
    else:
        H = tl.load(h_ptr + g[:, None] * G + g[None, :])
        hi = y.to(tl.bfloat16)
        lo = (y - hi.to(tl.float32)).to(tl.bfloat16)
        z = (tl.dot(hi, H) + tl.dot(lo, H)) * INV_SQRT_G
    if HAS_D:
        z += tl.load(d_ptr + offs_m[:, None] * stride_d + offs_k[None, :], mask = mask, other = 0.0).to(tl.float32)
    tl.store(c_ptr + offs_m[:, None] * K + offs_k[None, :], z.to(c_ptr.dtype.element_ty), mask = mask)


@triton.jit(do_not_specialize = ["M"])
def rotq2_kernel(x_ptr, h_ptr, q_ptr, s_ptr, M, K, stride_x, G: tl.constexpr, INV_SCALE: tl.constexpr,
                 BLOCK_M: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    g = tl.arange(0, G)
    H = tl.load(h_ptr + g[:, None] * G + g[None, :])
    amax = tl.zeros((BLOCK_M,), dtype = tl.float32)
    nans = tl.zeros((BLOCK_M,), dtype = tl.int32)
    for k0 in range(0, K, G):
        x = tl.load(x_ptr + rows[:, None] * stride_x + (k0 + g)[None, :], mask = rmask[:, None], other = 0.0)
        xr = tl.dot(x, H)
        amax = tl.maximum(amax, tl.max(tl.abs(xr), axis = 1))
        nans += tl.sum((xr != xr).to(tl.int32), axis = 1)
    amax = tl.maximum(amax, 1e-30)
    mul = 127.0 / amax
    for k0 in range(0, K, G):
        x = tl.load(x_ptr + rows[:, None] * stride_x + (k0 + g)[None, :], mask = rmask[:, None], other = 0.0)
        xr = tl.dot(x, H)
        y = _rint(xr * mul[:, None])
        y = tl.minimum(tl.maximum(y, -127.0), 127.0)
        tl.store(q_ptr + rows[:, None] * K + (k0 + g)[None, :], y.to(tl.int8), mask = rmask[:, None])
    tl.store(s_ptr + rows, tl.where(nans > 0, float("nan"), amax * INV_SCALE), mask = rmask)


@triton.jit(do_not_specialize = ["M"])
def rotq1_kernel(x_ptr, h_ptr, q_ptr, s_ptr, M, K, stride_x, G: tl.constexpr, INV_SCALE: tl.constexpr,
                 ROWS: tl.constexpr, NB: tl.constexpr):
    # one read of x: ROWS rows per program held as [ROWS * NB, G] (NB = next pow2 of K / G blocks)
    r = tl.arange(0, ROWS)
    b = tl.arange(0, NB)
    g = tl.arange(0, G)
    rows = tl.program_id(0).to(tl.int64) * ROWS + r
    nb = K // G
    m3 = (rows < M)[:, None, None] & (b < nb)[None, :, None]
    offs = rows[:, None, None] * stride_x + (b[None, :, None] * G + g[None, None, :])
    x = tl.load(x_ptr + offs, mask = m3, other = 0.0)
    H = tl.load(h_ptr + g[:, None] * G + g[None, :])
    xr = tl.reshape(tl.dot(tl.reshape(x, (ROWS * NB, G)), H), (ROWS, NB * G))
    amax = tl.maximum(tl.max(tl.abs(xr), axis = 1), 1e-30)
    nans = tl.sum((xr != xr).to(tl.int32), axis = 1)
    y = _rint(xr * (127.0 / amax)[:, None])
    y = tl.minimum(tl.maximum(y, -127.0), 127.0).to(tl.int8)
    y3 = tl.reshape(y, (ROWS, NB, G))
    qoffs = rows[:, None, None] * K + (b[None, :, None] * G + g[None, None, :])
    tl.store(q_ptr + qoffs, y3, mask = m3)
    tl.store(s_ptr + rows, tl.where(nans > 0, float("nan"), amax * INV_SCALE), mask = rows < M)


@triton.jit(do_not_specialize = ["N"])
def rowq2_kernel(x_ptr, c_ptr, q_ptr, s_ptr, N, stride_x, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    amax = tl.zeros((BLOCK,), dtype = tl.float32)
    nans = tl.zeros((BLOCK,), dtype = tl.int32)
    for n0 in range(0, N, BLOCK):
        mask = n0 + offs < N
        x = tl.load(x_ptr + row * stride_x + n0 + offs, mask = mask, other = 0.0).to(tl.float32)
        c = tl.load(c_ptr + n0 + offs, mask = mask, other = 0.0)
        xc = x * c
        amax = tl.maximum(amax, tl.abs(xc))
        nans += (xc != xc).to(tl.int32)
    rmax = tl.maximum(tl.max(amax, axis = 0), 1e-30)
    mul = 127.0 / rmax
    for n0 in range(0, N, BLOCK):
        mask = n0 + offs < N
        x = tl.load(x_ptr + row * stride_x + n0 + offs, mask = mask, other = 0.0).to(tl.float32)
        c = tl.load(c_ptr + n0 + offs, mask = mask, other = 0.0)
        y = _rint(x * c * mul)
        y = tl.minimum(tl.maximum(y, -127.0), 127.0)
        tl.store(q_ptr + row * N + n0 + offs, y.to(tl.int8), mask = mask)
    tl.store(s_ptr + row, tl.where(tl.sum(nans, axis = 0) > 0, float("nan"), rmax / 127.0))


@triton.jit(do_not_specialize = ["M"])
def rowq1_kernel(x_ptr, c_ptr, q_ptr, s_ptr, M, N, stride_x, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    offs = tl.arange(0, BLOCK)
    mask = (rows < M)[:, None] & (offs < N)[None, :]
    x = tl.load(x_ptr + rows[:, None] * stride_x + offs[None, :], mask = mask, other = 0.0).to(tl.float32)
    c = tl.load(c_ptr + offs, mask = offs < N, other = 0.0)
    xc = x * c[None, :]
    rmax = tl.maximum(tl.max(tl.abs(xc), axis = 1), 1e-30)
    nans = tl.sum((xc != xc).to(tl.int32), axis = 1)
    y = _rint(xc * (127.0 / rmax)[:, None])
    y = tl.minimum(tl.maximum(y, -127.0), 127.0)
    tl.store(q_ptr + rows[:, None] * N + offs[None, :], y.to(tl.int8), mask = mask)
    tl.store(s_ptr + rows, tl.where(nans > 0, float("nan"), rmax / 127.0), mask = rows < M)


def hadamard(dtype):
    m = torch.ones((1, 1))
    while m.shape[0] < G:
        m = torch.cat([torch.cat([m, m], 1), torch.cat([m, -m], 1)], 0)
    return m.to(dev, dtype).contiguous()


H_X = hadamard(DT)
H_BF = hadamard(torch.bfloat16)
INV_SCALE = 1.0 / (127.0 * math.sqrt(G))


def quant_weight(w):
    N, K = w.shape
    h = hadamard(torch.float32) / math.sqrt(G)
    wr = (w.float().view(N, K // G, G) @ h).view(N, K)
    amax = wr.abs().amax(1).clamp_min(1e-30)
    return torch.round(wr * (127.0 / amax)[:, None]).clamp_(-127, 127).to(torch.int8), amax / 127.0


def run_launch(fn, skip):
    _SKIP.on = bool(skip)
    try:
        return fn()
    finally:
        _SKIP.on = False


def i8_fwd(xq, xs, wq, ws, d, tile, skip):
    M, K = xq.shape
    N = wq.shape[0]
    bm, bn, bk, w, s = tile
    out = torch.empty((M, N), device = dev, dtype = DT)
    dd = d if d is not None else out
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    return run_launch(lambda: (mm_kernel[grid](xq, wq, xs, ws, dd, out, M, N, K, dd.stride(0), HAS_D = d is not None,
                                               BLOCK_M = bm, BLOCK_N = bn, BLOCK_K = bk, GROUP_M = 8, PF = int(skip),
                                               num_warps = w, num_stages = s), out)[1], skip)


def i8_dgrad(dq, ds, wqt, d, tile, skip):
    M, N = dq.shape
    K = wqt.shape[0]
    bm, br, w, s = tile
    out = torch.empty((M, K), device = dev, dtype = DT)
    dd = d if d is not None else out
    grid = (triton.cdiv(M, bm) * (K // G),)
    return run_launch(lambda: (dgrad_kernel[grid](dq, wqt, ds, H_BF, dd, out, M, N, K, dd.stride(0),
                                                  HAS_D = d is not None, INV_SQRT_G = 1 / math.sqrt(G), G = G,
                                                  BLOCK_M = bm, BLOCK_R = br, GROUP_M = 8, PF = int(skip),
                                                  num_warps = w, num_stages = s), out)[1], skip)


def intmm_fwd(xq, xs, wq, ws, d, epi = (64, 128, 4)):
    acc = torch._int_mm(xq, wq.t())
    M, N = acc.shape
    out = torch.empty((M, N), device = dev, dtype = DT)
    dd = d if d is not None else out
    bm, bn, w = epi
    epi_fwd_kernel[(triton.cdiv(M, bm), triton.cdiv(N, bn))](acc, xs, ws, dd, out, M, N, dd.stride(0),
                                                             HAS_D = d is not None, BLOCK_M = bm, BLOCK_N = bn,
                                                             num_warps = w)
    return out


def intmm_dgrad(dq, ds, wqt, d, ieee = False, bm = 64, w = 4):
    acc = torch._int_mm(dq, wqt.t())  # [M, K]; wqt [K, N] contiguous -> column-major [N, K] operand
    M, K = acc.shape
    out = torch.empty((M, K), device = dev, dtype = DT)
    dd = d if d is not None else out
    epi_dgrad_kernel[(triton.cdiv(M, bm), K // G)](acc, ds, H_BF, dd, out, M, K, dd.stride(0), HAS_D = d is not None,
                                                   INV_SQRT_G = 1 / math.sqrt(G), G = G, BLOCK_M = bm, IEEE = ieee,
                                                   num_warps = w)
    return out


def rotq2(x, rows = 16):
    M, K = x.shape
    q = torch.empty((M, K), device = dev, dtype = torch.int8)
    s = torch.empty((M,), device = dev, dtype = torch.float32)
    rotq2_kernel[(triton.cdiv(M, rows),)](x, H_X, q, s, M, K, x.stride(0), G = G, INV_SCALE = INV_SCALE,
                                         BLOCK_M = rows, num_warps = 4)
    return q, s


def rotq1(x, budget = 256, warps = 8):
    M, K = x.shape
    nb = triton.next_power_of_2(max(K // G, 16))
    rows = max(1, budget // nb)
    q = torch.empty((M, K), device = dev, dtype = torch.int8)
    s = torch.empty((M,), device = dev, dtype = torch.float32)
    rotq1_kernel[(triton.cdiv(M, rows),)](x, H_X, q, s, M, K, x.stride(0), G = G, INV_SCALE = INV_SCALE,
                                         ROWS = rows, NB = nb, num_warps = warps)
    return q, s


def rowq2(x, c):
    M, N = x.shape
    q = torch.empty((M, N), device = dev, dtype = torch.int8)
    s = torch.empty((M,), device = dev, dtype = torch.float32)
    rowq2_kernel[(M,)](x, c, q, s, N, x.stride(0), BLOCK = 2048, num_warps = 8)
    return q, s


def rowq1(x, c, budget = 16384, warps = 8):
    M, N = x.shape
    blk = triton.next_power_of_2(N)
    rows = max(1, budget // blk)
    q = torch.empty((M, N), device = dev, dtype = torch.int8)
    s = torch.empty((M,), device = dev, dtype = torch.float32)
    rowq1_kernel[(triton.cdiv(M, rows),)](x, c, q, s, M, N, x.stride(0), ROWS = rows, BLOCK = blk, num_warps = warps)
    return q, s


def bench(f):
    return triton.testing.do_bench(f, warmup = 20, rep = 80, return_mode = "median") * 1e3  # us


def relerr(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


MODELS = {
    # (N, K) per decoder layer: q, k, v, o, gate, up, down
    "qwen3_1.7b": [(2048, 2048), (1024, 2048), (1024, 2048), (2048, 2048), (6144, 2048), (6144, 2048), (2048, 6144)],
    "qwen3_4b": [(4096, 2560), (1024, 2560), (1024, 2560), (2560, 4096), (9728, 2560), (9728, 2560), (2560, 9728)],
    "qwen3_8b": [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (12288, 4096), (12288, 4096), (4096, 12288)],
}
SHARED = {1, 2, 5}
FWD_TILES = [(128, 128, 64, 4, 4), (128, 128, 64, 8, 3), (128, 128, 64, 8, 4), (128, 128, 128, 8, 3),
             (128, 256, 64, 8, 3), (256, 128, 64, 8, 3), (64, 128, 64, 4, 4), (128, 64, 64, 4, 4),
             (128, 128, 32, 4, 4), (64, 256, 64, 8, 3)]
DG_TILES = [(128, 128, 8, 3), (128, 64, 8, 3), (128, 128, 8, 4), (64, 128, 4, 3), (64, 64, 4, 4), (128, 64, 4, 4),
            (128, 128, 4, 3), (64, 128, 8, 3)]
if args.quick:
    FWD_TILES = FWD_TILES[:3]
    DG_TILES = DG_TILES[:2]


def try_(f):
    try:
        return f()
    except Exception as e:  # noqa
        return f"ERR {type(e).__name__}: {str(e)[:200]}"


def first_ok(fn, tiles, r, key):
    errs = []
    for t in tiles:
        try:
            out = fn(t)
            torch.cuda.synchronize()
            r[key] = list(t)
            return out
        except Exception as e:  # noqa
            errs.append(f"{t}: {type(e).__name__}: {str(e)[:120]}")
    r[key] = "none launched: " + " | ".join(errs)[:1500]
    return None


res = {"device": torch.cuda.get_device_name(), "cap": list(cap), "torch": torch.__version__, "triton": triton.__version__,
       "cuda": torch.version.cuda, "hip": getattr(torch.version, "hip", None),
       "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None), "dtype": str(DT), "prefetch_patch": PREFETCH_PATCH, "probes": {}, "shapes": {},
       "started": time.strftime("%Y-%m-%d %H:%M:%S")}
IS_HIP = bool(res["hip"])
SKIPS = (False,) if IS_HIP else (False, True)
res["skip_variants"] = "dropped on HIP" if IS_HIP else "stock and skip"
print(json.dumps({k: res[k] for k in ("device", "cap", "torch", "triton", "cuda", "hip", "arch", "dtype", "prefetch_patch")}), flush = True)

# probes
a8 = torch.randint(-127, 128, (64, 256), device = dev, dtype = torch.int8)
b8 = torch.randint(-127, 128, (128, 256), device = dev, dtype = torch.int8)
ref = (a8.double() @ b8.double().t())
res["probes"]["int_mm"] = try_(lambda: "ok exact" if torch.equal(torch._int_mm(a8, b8.t()).double(), ref) else "MISMATCH")
one = torch.ones(64, device = dev)
res["probes"]["triton_int8_dot"] = try_(
    lambda: "ok exact" if torch.equal(i8_fwd(a8, one, b8, torch.ones(128, device = dev), None, (64, 64, 64, 4, 2), False).double(),
                                      ref.to(DT).double()) else "MISMATCH")
print("probes", json.dumps(res["probes"]), flush = True)
triton_ok = str(res["probes"]["triton_int8_dot"]).startswith("ok")
intmm_ok = str(res["probes"]["int_mm"]).startswith("ok")

ms = [int(m) for m in args.ms.split(",")]
models = args.models.split(",")
shapes = sorted({s for m in models for s in MODELS[m]})
torch.manual_seed(0)
for (N, K) in shapes:
    w = (torch.randn(N, K, device = dev) * 0.02).to(DT)
    wq, ws = quant_weight(w)
    wqt = wq.t().contiguous()
    wdense = (wq.float() * ws[:, None]).view(N, K // G, G) @ (hadamard(torch.float32) / math.sqrt(G)).t()
    wdense = wdense.view(N, K)
    for M in ms:
        key = f"{N}x{K}@{M}"
        r = {}
        try:
            x = torch.randn(M, K, device = dev).to(DT)
            dy = (torch.randn(M, N, device = dev) * 1e-3).to(DT)
            dfw = (torch.randn(M, N, device = dev) * 0.1).to(DT)
            dbw = (torch.randn(M, K, device = dev) * 1e-4).to(DT)
            r = {"bf16_fwd": [bench(lambda: x @ w.t()) for _ in range(3)], "bf16_dgrad": [bench(lambda: dy @ w) for _ in range(3)]}
            # quant
            q2, s2 = rotq2(x)
            r["rotq2"] = bench(lambda: rotq2(x))
            for budget, warps in [(128, 4), (128, 8), (256, 8), (256, 4)]:
                k_ = f"rotq1_b{budget}_w{warps}"
                out = try_(lambda: rotq1(x, budget, warps))
                if isinstance(out, str):
                    r[k_] = out
                    continue
                q1, s1 = out
                diff = (q1.int() - q2.int()).abs()
                r[k_] = bench(lambda: rotq1(x, budget, warps))
                r[k_ + "_check"] = {"max_code_diff": diff.max().item(), "n_diff": int((diff > 0).sum().item()),
                                    "scale_rel": ((s1 - s2).abs() / s2).max().item()}
            dq2, ds2 = rowq2(dy, ws)
            r["rowq2"] = bench(lambda: rowq2(dy, ws))
            for budget, warps in [(8192, 8), (16384, 8), (16384, 16)]:
                k_ = f"rowq1_b{budget}_w{warps}"
                out = try_(lambda: rowq1(dy, ws, budget, warps))
                if isinstance(out, str):
                    r[k_] = out
                    continue
                dq1, ds1 = out
                r[k_] = bench(lambda: rowq1(dy, ws, budget, warps))
                r[k_ + "_check"] = {"max_code_diff": (dq1.int() - dq2.int()).abs().max().item(),
                                    "scale_rel": ((ds1 - ds2).abs() / ds2).max().item()}
            xq, xs = q2, s2
            dq, ds = dq2, ds2
            ref_f = None
            ref_d = None
            if triton_ok:
                ref_f = first_ok(lambda t: i8_fwd(xq, xs, wq, ws, dfw, t, False), FWD_TILES, r, "ref_fwd_tile")
            if ref_f is not None:
                # sanity vs dense dequantized math
                r["fwd_relerr_vs_bf16"] = relerr(ref_f, (x.float() @ w.float().t() + dfw.float()))
                for t in FWD_TILES:
                    for skip in SKIPS:
                        k_ = f"tri_fwd_{'_'.join(map(str, t))}_{'skip' if skip else 'stock'}"
                        out = try_(lambda: i8_fwd(xq, xs, wq, ws, dfw, t, skip))
                        if isinstance(out, str):
                            r[k_] = out
                            continue
                        r[k_] = bench(lambda: i8_fwd(xq, xs, wq, ws, dfw, t, skip))
                        if not torch.equal(out, ref_f):
                            r[k_ + "_mismatch"] = relerr(out, ref_f)
            if triton_ok:
                ref_d = first_ok(lambda t: i8_dgrad(dq, ds, wqt, dbw, t, False), DG_TILES, r, "ref_dgrad_tile")
            if ref_d is not None:
                r["dgrad_relerr_vs_bf16"] = relerr(ref_d, dy.float() @ w.float() + dbw.float())
                for t in DG_TILES:
                    for skip in SKIPS:
                        k_ = f"tri_dgrad_{'_'.join(map(str, t))}_{'skip' if skip else 'stock'}"
                        out = try_(lambda: i8_dgrad(dq, ds, wqt, dbw, t, skip))
                        if isinstance(out, str):
                            r[k_] = out
                            continue
                        r[k_] = bench(lambda: i8_dgrad(dq, ds, wqt, dbw, t, skip))
                        r[k_ + "_relerr"] = relerr(out, ref_d)
            if intmm_ok:
                r["intmm_only_fwd"] = try_(lambda: bench(lambda: torch._int_mm(xq, wq.t())))
                r["intmm_only_dgrad"] = try_(lambda: bench(lambda: torch._int_mm(dq, wqt.t())))
                for epi in [(64, 128, 4), (32, 256, 4), (64, 256, 8)]:
                    k_ = f"intmm_fwd_epi_{'_'.join(map(str, epi))}"
                    out = try_(lambda: intmm_fwd(xq, xs, wq, ws, dfw, epi))
                    if isinstance(out, str):
                        r[k_] = out
                        continue
                    r[k_] = bench(lambda: intmm_fwd(xq, xs, wq, ws, dfw, epi))
                    r[k_ + "_relerr"] = relerr(out, ref_f) if ref_f is not None else relerr(out, x.float() @ w.float().t() + dfw.float())
                for ieee, bm, wr in [(False, 64, 4), (False, 32, 4), (True, 32, 4), (True, 64, 8)]:
                    k_ = f"intmm_dgrad_epi_{'ieee' if ieee else 'hilo'}_{bm}_{wr}"
                    out = try_(lambda: intmm_dgrad(dq, ds, wqt, dbw, ieee, bm, wr))
                    if isinstance(out, str):
                        r[k_] = out
                        continue
                    r[k_] = bench(lambda: intmm_dgrad(dq, ds, wqt, dbw, ieee, bm, wr))
                    r[k_ + "_relerr"] = relerr(out, ref_d) if ref_d is not None else relerr(out, dy.float() @ w.float() + dbw.float())
            r["transpose_w"] = bench(lambda: wq.t().contiguous())
            res["shapes"][key] = r
            best_f = min([v for k2, v in r.items() if k2.startswith(("tri_fwd_", "intmm_fwd_epi")) and isinstance(v, float) and not k2.endswith(("_relerr", "_mismatch"))] or [float("nan")])
            best_d = min([v for k2, v in r.items() if k2.startswith(("tri_dgrad_", "intmm_dgrad_epi")) and isinstance(v, float) and not k2.endswith(("_relerr", "_mismatch"))] or [float("nan")])
            print(f"{key}: bf16 fwd {min(r['bf16_fwd']):.0f} dgrad {min(r['bf16_dgrad']):.0f} | best int8 fwd {best_f:.0f} dgrad {best_d:.0f} | "
                  f"rotq2 {r['rotq2']:.0f} rotq1 {r.get('rotq1_b256_w8')} rowq2 {r['rowq2']:.0f}", flush = True)
            json.dump(res, open(args.out, "w"), indent = 1)
            del x, dy, dfw, dbw
        except Exception as e:  # noqa
            r["cell_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            r["cell_traceback"] = traceback.format_exc()[-3000:]
            res["shapes"][key] = r
            print(f"{key}: CELL ERROR {r['cell_error']}", flush = True)
            json.dump(res, open(args.out, "w"), indent = 1)
            torch.cuda.empty_cache()
    del w, wq, wqt
    torch.cuda.empty_cache()

res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(res, open(args.out, "w"), indent = 1)
print("DONE", args.out, flush = True)
