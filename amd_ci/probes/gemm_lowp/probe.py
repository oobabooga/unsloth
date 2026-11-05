"""Low-precision GEMM probe for gfx1151 (observes only; every result goes to --out JSON, never stdout).

Sections (one per process, so a crash in one cannot take the others down):
  env       torch / HIP / device facts
  dense     bf16 and fp16 F.linear TFLOP/s
  intmm     torch._int_mm TOP/s and Studio's eager W8A8 (UNSLOTH_NATIVE_INT8_ACT=1) end to end + breakdown
  triton    Triton fused W8A8 (int8 tl.dot, scale epilogue), Triton bf16 same skeleton, Triton W4A16, row quant
  compile   torch.compile: generic MLP block, Studio's _w8a8_rows (dynamic=True, fullgraph=True)
  hip       hand-written WMMA kernels (f16, bf16, iu8, iu4, w4a16) from build_wmma.py: peak + GEMM + correctness
  tunable   hipBLASLt TunableOp off vs on (same process), bf16
  accuracy  int4 / int8 error proxies on real FLUX.2-klein-4B weights (HTTP range reads) + synthetic activations
  block     DiT block time and GEMM share (Qwen-Image-like double block, FLUX.2-like single block)
  inline    torch.utils.cpp_extension.load_inline smoke (tiny HIP kernel)
"""
import argparse
import json
import math
import os
import platform
import sys
import time
import traceback

OUT = {}
OUT_PATH = None


def save():
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(OUT, f, indent=1, default=str)
    os.replace(tmp, OUT_PATH)


def err(exc=None):
    return {"error": repr(exc) if exc is not None else "", "traceback": traceback.format_exc()[-5000:]}


import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = os.environ.get("GEMM_PROBE_DEV", "cuda")
KN = [(3072, 3072), (3072, 12288), (12288, 3072)]
MS = [64, 256, 4096, 8192]
SHAPES = [(m, k, n) for m in MS for (k, n) in KN] + [(4096, 4096, 4096)]
_FLUSH = None


def key(m, k, n):
    return f"M{m}_K{k}_N{n}"


def flush():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(32 * 1024 * 1024, dtype=torch.float32, device=DEV)  # 128 MB > 32 MB MALL
    _FLUSH.zero_()


def bench(fn, warmup=3, reps=15, cold=True):
    """Median ms over reps, each rep timed with its own event pair; caches flushed between reps (cold weights)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    fn()
    torch.cuda.synchronize()
    one = time.time() - t0
    if one > 0.2:
        reps = max(5, min(reps, int(3.0 / one)))
    ev = []
    for _ in range(reps):
        if cold:
            flush()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        ev.append((s, e))
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in ev)
    return {"ms": ts[len(ts) // 2], "p10": ts[max(0, len(ts) // 10)], "p90": ts[min(len(ts) - 1, (9 * len(ts)) // 10)],
            "reps": len(ts)}


def tops(m, k, n, ms):
    return round(2.0 * m * k * n / (ms * 1e-3) / 1e12, 3)


def relerr(out, ref):
    out = out.float()
    ref = ref.float()
    d = (out - ref)
    return {"max_abs": d.abs().max().item(), "max_abs_ref": ref.abs().max().item(),
            "rel_max": (d.abs().max() / ref.abs().max().clamp_min(1e-30)).item(),
            "rel_fro": (d.norm() / ref.norm().clamp_min(1e-30)).item()}


# ----------------------------------------------------------------------------------------------------------- env
def sec_env(a):
    p = torch.cuda.get_device_properties(0)
    OUT["env"] = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None), "device": p.name, "arch": getattr(p, "gcnArchName", None),
        "cu": p.multi_processor_count, "mem_gb": round(p.total_memory / 2**30, 1),
        "hipblaslt_pref": str(getattr(torch.backends.cuda, "preferred_blas_library", lambda: None)()),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("TORCH", "PYTORCH", "HIP", "ROCR", "HSA", "MIOPEN"))},
    }
    try:
        import triton
        OUT["env"]["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        OUT["env"]["triton"] = repr(exc)
    save()


# --------------------------------------------------------------------------------------------------------- dense
def sec_dense(a):
    res = {}
    for (m, k, n) in SHAPES:
        r = {}
        for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            x = torch.randn(m, k, device=DEV, dtype=dt)
            w = torch.randn(n, k, device=DEV, dtype=dt) * 0.02
            b = bench(lambda: F.linear(x, w))
            r[name] = dict(b, tflops=tops(m, k, n, b["ms"]))
            del x, w
        res[key(m, k, n)] = r
        OUT["dense"] = res
        save()


# --------------------------------------------------------------------------------------------- Studio W8A8 replica
INT_MM_CHUNK_ELEMS = 1 << 25
INT_MM_MIN_ROWS = 17


def sylvester(n, device=DEV):
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def studio_int_mm_rows(x2, weight_q, w_scale, bias):
    # Verbatim maths of NativeWeightOnlyLinear._int_mm_rows (diffusion_native_quant.py, unsloth main).
    x_scale = torch.linalg.vector_norm(x2, ord=float("inf"), dim=1, keepdim=True, dtype=torch.float32)
    x_scale = x_scale.clamp_(min=1e-12).div_(127.0)
    xq = torch.mul(x2, x_scale.reciprocal()).round_().clamp_(-127, 127).to(torch.int8)
    if not xq.is_contiguous():
        xq = xq.contiguous()
    acc = torch._int_mm(xq, weight_q.t())
    out = torch.mul(acc, x_scale)
    if bias is not None:
        out = torch.addcmul(bias.float(), out, w_scale)
    else:
        out = out.mul_(w_scale)
    return out


def studio_forward_int_mm(x, weight_q, w_scale, bias, in_f, out_f):
    x2 = x.reshape(-1, in_f)
    rows = x2.shape[0]
    chunk = INT_MM_CHUNK_ELEMS // max(in_f, out_f)
    if rows <= chunk or chunk < 2 * INT_MM_MIN_ROWS:
        out = studio_int_mm_rows(x2, weight_q, w_scale, bias).to(x.dtype)
    else:
        out = torch.cat([studio_int_mm_rows(p, weight_q, w_scale, bias).to(x.dtype)
                         for p in torch.tensor_split(x2, -(-rows // chunk))])
    return out.reshape(*x.shape[:-1], out_f)


def studio_rotate(x, rot_h, g, in_f):
    grouped = x.reshape(*x.shape[:-1], in_f // g, g)
    return (grouped @ rot_h.to(x.dtype)).reshape(x.shape)


def studio_w8a8_rows(x2, rot_h, weight_q, weight_scale, bias):
    # Verbatim maths of _w8a8_rows (the function Studio compiles as _fused_w8a8 on non-ROCm).
    if rot_h is not None:
        g = rot_h.shape[0]
        x2 = (x2.reshape(x2.shape[0], -1, g) @ rot_h.to(x2.dtype)).reshape(x2.shape)
    x_scale = torch.linalg.vector_norm(x2, ord=float("inf"), dim=1, keepdim=True, dtype=torch.float32).clamp(min=1e-12)
    x_scale = x_scale * (1.0 / 127.0)
    x_recip = (1.0 / x_scale.double()).float()
    xq = torch.mul(x2, x_recip).round().clamp(-127, 127).to(torch.int8)
    if rot_h is None:
        xq = xq.contiguous()
    acc = torch._int_mm(xq, weight_q.t())
    out = torch.mul(acc, x_scale)
    w_scale = weight_scale.view(torch.float32)
    if bias is not None:
        out = torch.addcmul(bias.float(), out, w_scale)
    else:
        out = out * w_scale
    return out.to(x2.dtype)


def make_w8(n, k):
    w = torch.randn(n, k, device=DEV) * 0.02
    s = w.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127.0
    wq = (w / s).round_().clamp_(-127, 127).to(torch.int8)
    return w, wq, s.squeeze(1).float().contiguous()


def sec_intmm(a):
    res = {}
    g = 256
    for (m, k, n) in SHAPES:
        r = {}
        try:
            x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
            w, wq, ws = make_w8(n, k)
            wb = w.to(torch.bfloat16)
            bias = torch.randn(n, device=DEV, dtype=torch.bfloat16) * 0.01
            xq = torch.randint(-127, 128, (m, k), device=DEV, dtype=torch.int8)
            wqT = wq.t().contiguous()
            def tb(name, fn, unit="tops"):
                try:
                    b = bench(fn)
                    r[name] = dict(b, **{unit: tops(m, k, n, b["ms"])})
                except Exception as exc:  # noqa: BLE001
                    r[name] = err(exc)
            tb("bf16_linear_bias", lambda: F.linear(x, wb, bias), "tflops")
            tb("int_mm_colmajorB", lambda: torch._int_mm(xq, wq.t()))
            tb("int_mm_rowmajorB", lambda: torch._int_mm(xq, wqT))
            try:
                ref = (xq[:64].double() @ wq.double().t())
                got = torch._int_mm(xq, wq.t())[:64].double()
                r["int_mm_exact"] = bool(torch.equal(ref, got))
            except Exception as exc:  # noqa: BLE001
                r["int_mm_exact"] = repr(exc)[:300]
            tb("eager_w8a8_e2e", lambda: studio_forward_int_mm(x, wq, ws, bias, k, n))
            rot = sylvester(g).to(torch.bfloat16)
            tb("eager_w8a8_convrot256_e2e", lambda: studio_forward_int_mm(studio_rotate(x, rot, g, k), wq, ws, bias, k, n))
            # breakdown on the full row block (no chunking), same ops as _int_mm_rows

            def quant():
                xs = torch.linalg.vector_norm(x, ord=float("inf"), dim=1, keepdim=True, dtype=torch.float32)
                xs = xs.clamp_(min=1e-12).div_(127.0)
                return torch.mul(x, xs.reciprocal()).round_().clamp_(-127, 127).to(torch.int8), xs
            xq2, xs2 = quant()
            acc = torch._int_mm(xq2, wq.t())

            def epi():
                return torch.addcmul(bias.float(), torch.mul(acc, xs2), ws).to(torch.bfloat16)
            r["part_quant_ms"] = bench(quant)["ms"]
            r["part_int_mm_ms"] = bench(lambda: torch._int_mm(xq2, wq.t()))["ms"]
            r["part_epilogue_ms"] = bench(epi)["ms"]
            r["part_rotate256_ms"] = bench(lambda: studio_rotate(x, rot, g, k))["ms"]
            out = studio_forward_int_mm(x, wq, ws, bias, k, n)
            ref = x[:64].float() @ (wq.float() * ws[:, None]).t() + bias.float()
            r["eager_w8a8_err_vs_dequant_ref"] = relerr(out[:64], ref)
            del x, w, wq, wb, xq, wqT, acc, xq2
        except Exception as exc:  # noqa: BLE001
            r["error"] = err(exc)
        res[key(m, k, n)] = r
        OUT["intmm"] = res
        save()


# --------------------------------------------------------------------------------------------------------- triton
def triton_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def mm_kernel(a_ptr, b_ptr, c_ptr, sa_ptr, sb_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, INT8: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
        if INT8:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _k in range(0, K, BLOCK_K):
            av = tl.load(a_ptrs)
            bv = tl.load(b_ptrs)
            if INT8:
                acc += tl.dot(av, bv, out_dtype=tl.int32)
            else:
                acc += tl.dot(av, bv)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        if INT8:
            sa = tl.load(sa_ptr + offs_m)
            sb = tl.load(sb_ptr + offs_n)
            c = acc.to(tl.float32) * sa[:, None] * sb[None, :]
        else:
            c = acc
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c.to(tl.bfloat16))

    @triton.jit
    def w4a16_kernel(a_ptr, b_ptr, s_ptr, c_ptr, M, N, K, G: tl.constexpr, BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
        # b: [N, K//2] uint8; inside each BLOCK_K chunk byte j holds k=j (low nibble) and k=j+BLOCK_K/2 (high).
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        HALF: tl.constexpr = BLOCK_K // 2
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_h = tl.arange(0, HALF)
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_h[None, :]
        b_ptrs = b_ptr + offs_n[None, :] * (K // 2) + offs_h[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a_lo = tl.load(a_ptrs)
            a_hi = tl.load(a_ptrs + HALF)
            bp = tl.load(b_ptrs).to(tl.int32)
            lo = ((bp & 15) ^ 8) - 8
            hi = ((bp >> 4) ^ 8) - 8
            s_lo = tl.load(s_ptr + offs_n * (K // G) + k // G)
            s_hi = tl.load(s_ptr + offs_n * (K // G) + (k + HALF) // G)
            b_lo = (lo.to(tl.float32) * s_lo[None, :]).to(a_lo.dtype)
            b_hi = (hi.to(tl.float32) * s_hi[None, :]).to(a_hi.dtype)
            acc += tl.dot(a_lo, b_lo)
            acc += tl.dot(a_hi, b_hi)
            a_ptrs += BLOCK_K
            b_ptrs += HALF
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))

    @triton.jit
    def quant_rows_kernel(x_ptr, q_ptr, s_ptr, K, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        s = tl.maximum(amax, 1e-12) / 127.0
        v = x / s
        v = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))
        v = tl.minimum(tl.maximum(v, -127.0), 127.0)
        tl.store(q_ptr + row * K + offs, v.to(tl.int8), mask=mask)
        tl.store(s_ptr + row, s)

    return triton, mm_kernel, w4a16_kernel, quant_rows_kernel


MM_CONFIGS = [  # BLOCK_M, BLOCK_N, BLOCK_K(bf16), BLOCK_K(int8), num_warps, num_stages
    (64, 64, 32, 64, 4, 2), (64, 128, 32, 64, 4, 2), (128, 64, 32, 64, 4, 2), (128, 128, 32, 64, 4, 2),
    (128, 128, 32, 64, 8, 2), (128, 128, 64, 128, 8, 2), (128, 256, 32, 64, 8, 2), (256, 128, 32, 64, 8, 2),
]


def wmma_counts(compiled):
    import re
    try:
        asm = compiled.asm.get("amdgcn", "")
    except Exception:  # noqa: BLE001
        return None
    found = re.findall(r"v_wmma_[a-z0-9_]+|v_dot[0-9a-z_]+|v_mfma[a-z0-9_]+", asm)
    c = {}
    for f in found:
        c[f] = c.get(f, 0) + 1
    vg = re.findall(r"\.vgpr_count:\s+(\d+)", asm)
    sp = re.findall(r"\.vgpr_spill_count:\s+(\d+)", asm)
    return {"ops": c, "vgpr": vg[:1], "spill": sp[:1]}


def sec_triton(a):
    res = {"configs": MM_CONFIGS}
    try:
        triton, mm_kernel, w4a16_kernel, quant_rows_kernel = triton_kernels()
        res["triton"] = triton.__version__
        try:
            res["target"] = str(triton.runtime.driver.active.get_current_target())
        except Exception as exc:  # noqa: BLE001
            res["target"] = repr(exc)
    except Exception as exc:  # noqa: BLE001
        res["error"] = err(exc)
        OUT["triton"] = res
        save()
        return
    asm_seen = {}
    shapes = {}
    res["shapes"] = shapes
    for (m, k, n) in SHAPES:
        r = {}
        shapes[key(m, k, n)] = r
        x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
        wb = (torch.randn(n, k, device=DEV) * 0.02).to(torch.bfloat16)
        xq = torch.randint(-127, 128, (m, k), device=DEV, dtype=torch.int8)
        _, wq, ws = make_w8(n, k)
        sa = (torch.rand(m, device=DEV) + 0.5) * 1e-3
        c = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
        for mode in ("w8a8", "bf16"):
            best = None
            per = {}
            for cfg in MM_CONFIGS:
                bm, bn, bk16, bk8, nw, ns = cfg
                bk = bk8 if mode == "w8a8" else bk16
                if m % bm or n % bn or k % bk:
                    continue
                grid = ((m // bm) * (n // bn),)
                A, B = (xq, wq) if mode == "w8a8" else (x, wb)

                def run(A=A, B=B, bm=bm, bn=bn, bk=bk, nw=nw, ns=ns, grid=grid):
                    return mm_kernel[grid](A, B, c, sa, ws, m, n, k, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
                                           INT8=(mode == "w8a8"), num_warps=nw, num_stages=ns)
                tag = f"{bm}x{bn}x{bk}_w{nw}_s{ns}"
                try:
                    ck = run()
                    torch.cuda.synchronize()
                    akey = f"{mode}_{tag}"
                    if akey not in asm_seen:
                        asm_seen[akey] = wmma_counts(ck)
                    b = bench(run, warmup=2, reps=8)
                    per[tag] = round(b["ms"], 4)
                    if best is None or b["ms"] < best[1]:
                        best = (tag, b["ms"], run)
                except Exception as exc:  # noqa: BLE001
                    per[tag] = {"error": repr(exc)[:600]}
            entry = {"per_config_ms": per}
            if best is not None:
                tag, _, run = best
                b = bench(run)
                entry.update(best=tag, ms=b["ms"], tops=tops(m, k, n, b["ms"]))
                run()
                torch.cuda.synchronize()
                if mode == "w8a8":
                    ref = (xq[:64].float() @ wq.float().t()) * sa[:64, None] * ws[None, :]
                else:
                    ref = x[:64].float() @ wb.float().t()
                entry["err"] = relerr(c[:64], ref)
            r[mode] = entry
            OUT["triton"] = res
            save()
        # row quant kernel + end to end W8A8 (quant + fused GEMM)
        try:
            q = torch.empty(m, k, device=DEV, dtype=torch.int8)
            s = torch.empty(m, device=DEV, dtype=torch.float32)
            blk = triton.next_power_of_2(k)

            def qrun():
                return quant_rows_kernel[(m,)](x, q, s, k, BLOCK=blk, num_warps=8 if blk <= 4096 else 16)
            qrun()
            torch.cuda.synchronize()
            xs_ref = x.float().abs().amax(1).clamp_min(1e-12) / 127.0
            r["quant_rows"] = dict(bench(qrun), scale_max_rel=((s - xs_ref).abs() / xs_ref).max().item(),
                                   code_mismatch_frac=(q.float() != (x.float() / xs_ref[:, None]).round().clamp(-127, 127)).float().mean().item())
            if "best" in r.get("w8a8", {}):
                bm, bn, bk, nw, ns = [int(t) for t in r["w8a8"]["best"].replace("_w", "x").replace("_s", "x").split("x")]

                def e2e():
                    qrun()
                    return mm_kernel[((m // bm) * (n // bn),)](q, wq, c, s, ws, m, n, k, BLOCK_M=bm, BLOCK_N=bn,
                                                               BLOCK_K=bk, GROUP_M=8, INT8=True, num_warps=nw,
                                                               num_stages=ns)
                b = bench(e2e)
                r["w8a8_e2e_quant_plus_gemm"] = dict(b, tops=tops(m, k, n, b["ms"]))
                e2e()
                torch.cuda.synchronize()
                ref = x[:64].float() @ (wq.float() * ws[:, None]).t()
                r["w8a8_e2e_err_vs_dequant_ref"] = relerr(c[:64], ref)
        except Exception as exc:  # noqa: BLE001
            r["quant_error"] = err(exc)
        # W4A16 (group 128), BLOCK_K 64 packing
        try:
            G, BK = 128, 64
            w = torch.randn(n, k, device=DEV) * 0.02
            sg = w.view(n, k // G, G).abs().amax(-1).clamp_min(1e-12) / 7.0
            qw = (w.view(n, k // G, G) / sg[..., None]).round().clamp(-8, 7).to(torch.int32).view(n, k)
            v = qw.view(n, k // BK, 2, BK // 2)
            packed = ((v[:, :, 0, :] & 15) | ((v[:, :, 1, :] & 15) << 4)).to(torch.uint8).reshape(n, k // 2).contiguous()
            sgc = sg.float().contiguous()
            deq = (qw.view(n, k // G, G).float() * sg[..., None]).view(n, k)
            best = None
            per = {}
            for (bm, bn, _bk16, _bk8, nw, ns) in MM_CONFIGS:
                if m % bm or n % bn:
                    continue
                grid = ((m // bm) * (n // bn),)

                def run(bm=bm, bn=bn, nw=nw, ns=ns, grid=grid):
                    return w4a16_kernel[grid](x, packed, sgc, c, m, n, k, G=G, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=BK,
                                              GROUP_M=8, num_warps=nw, num_stages=ns)
                tag = f"{bm}x{bn}x{BK}_w{nw}_s{ns}"
                try:
                    ck = run()
                    torch.cuda.synchronize()
                    if f"w4a16_{tag}" not in asm_seen:
                        asm_seen[f"w4a16_{tag}"] = wmma_counts(ck)
                    b = bench(run, warmup=2, reps=8)
                    per[tag] = round(b["ms"], 4)
                    if best is None or b["ms"] < best[1]:
                        best = (tag, b["ms"], run)
                except Exception as exc:  # noqa: BLE001
                    per[tag] = {"error": repr(exc)[:600]}
            entry = {"per_config_ms": per, "group": G}
            if best is not None:
                tag, _, run = best
                b = bench(run)
                entry.update(best=tag, ms=b["ms"], tops=tops(m, k, n, b["ms"]))
                run()
                torch.cuda.synchronize()
                entry["err"] = relerr(c[:64], x[:64].float() @ deq.t())
            r["w4a16"] = entry
        except Exception as exc:  # noqa: BLE001
            r["w4a16_error"] = err(exc)
        res["asm"] = asm_seen
        OUT["triton"] = res
        save()
        del x, wb, xq, wq, c
    res["int4_dot"] = ("not attempted as a kernel: triton.language has no int4 dtype and tl.dot accepts "
                       "int8/fp8/fp16/bf16/fp32 operands only, so an int4 GEMM in Triton means unpacking to int8 "
                       "(same WMMA rate as iu8) or to bf16 (the W4A16 kernel above)")
    OUT["triton"] = res
    save()


# -------------------------------------------------------------------------------------------------------- compile
def sec_compile(a):
    res = {}
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(3072, elementwise_affine=False)
            self.fc1 = nn.Linear(3072, 12288)
            self.fc2 = nn.Linear(12288, 3072)

        def forward(self, x):
            return x + self.fc2(F.gelu(self.fc1(self.norm(x)), approximate="tanh"))

    # Studio's _w8a8_rows compiled exactly as _fused_w8a8 does (dynamic=True, fullgraph=True).
    for (m, k, n) in [(4096, 3072, 3072), (4096, 3072, 12288), (4096, 12288, 3072), (8192, 3072, 12288)]:
        for rot in (None, 256):
            label = f"studio_w8a8_rows_{key(m, k, n)}_rot{rot or 0}"
            r = {}
            try:
                torch._dynamo.reset()
                xx = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
                _, wq, ws = make_w8(n, k)
                wsi = ws.view(torch.int32)
                bias = torch.randn(n, device=DEV, dtype=torch.bfloat16) * 0.01
                rh = sylvester(rot).to(torch.bfloat16) if rot else None
                eager = studio_w8a8_rows(xx, rh, wq, wsi, bias)
                r["eager"] = dict(bench(lambda: studio_w8a8_rows(xx, rh, wq, wsi, bias)))
                r["eager"]["tops"] = tops(m, k, n, r["eager"]["ms"])
                fn = torch.compile(studio_w8a8_rows, dynamic=True, fullgraph=True)
                t0 = time.time()
                out = fn(xx, rh, wq, wsi, bias)
                torch.cuda.synchronize()
                r["first_call_s"] = round(time.time() - t0, 2)
                r["compiled"] = bench(lambda: fn(xx, rh, wq, wsi, bias))
                r["compiled"]["tops"] = tops(m, k, n, r["compiled"]["ms"])
                r["err_vs_eager"] = relerr(out, eager)
                wdq = (wq.float() * ws[:, None]).to(torch.bfloat16)
                r["bf16_ms"] = bench(lambda: F.linear(xx, wdq, bias))["ms"]
            except Exception as exc:  # noqa: BLE001
                r.update(err(exc))
            res[label] = r
            OUT["compile"] = res
            save()
    mod = MLP().to(DEV, torch.bfloat16).eval()
    x = torch.randn(4096, 3072, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = mod(x)
        res["mlp_eager"] = bench(lambda: mod(x))
        for label, kw in (("mlp_compile_default", {}), ("mlp_compile_dynamic", {"dynamic": True}),
                          ("mlp_compile_max_autotune_no_cg", {"mode": "max-autotune-no-cudagraphs"})):
            try:
                torch._dynamo.reset()
                cm = torch.compile(mod, **kw)
                t0 = time.time()
                out = cm(x)
                torch.cuda.synchronize()
                first = time.time() - t0
                res[label] = dict(bench(lambda: cm(x)), first_call_s=round(first, 2), err=relerr(out, ref))
            except Exception as exc:  # noqa: BLE001
                res[label] = err(exc)
            OUT["compile"] = res
            save()


# ------------------------------------------------------------------------------------------------------------ hip
def load_libs(lib_dir, kinds):
    import ctypes
    libs = {}
    for name in kinds:
        p = os.path.join(lib_dir, f"libwm_{name}.so")
        if not os.path.exists(p):
            libs[name] = "missing"
            continue
        try:
            # RTLD_LOCAL + hidden visibility: every libwm_*.so defines the same kernel names, and a global
            # load lets the first library's kernels interpose on the others (run 1 launched f16 kernels for all).
            lib = ctypes.CDLL(p, mode=ctypes.RTLD_LOCAL)
            if lib.wm_kind() != {"f16": 0, "bf16": 1, "iu8": 2, "iu4": 3, "w4a16": 4}[name]:
                raise RuntimeError(f"wm_kind mismatch for {name}")
            vp, ci = ctypes.c_void_p, ctypes.c_int
            lib.wm_gemm.argtypes = [ci, vp, vp, vp, vp, vp, ci, ci, ci, ci, vp]
            lib.wm_peak.argtypes = [ci, ci, vp, vp]
            lib.wm_tile.argtypes = [vp, vp, vp, vp]
            libs[name] = lib
        except Exception as exc:  # noqa: BLE001
            libs[name] = repr(exc)
    return libs


def pack_int4(q, lo_first=True):
    """q int tensor [..., K] in [-8, 7] -> uint8 [..., K/2]; element 2j in the low nibble when lo_first."""
    q = q.to(torch.int32)
    e, o = q[..., 0::2] & 15, q[..., 1::2] & 15
    v = (e | (o << 4)) if lo_first else (o | (e << 4))
    return v.to(torch.uint8).contiguous()


def sec_hip(a):
    res = {}
    kinds = [k for k in a.kinds.split(",") if k]
    libs = load_libs(a.lib_dir, kinds)
    res["libs"] = {k: (v if isinstance(v, str) else "loaded") for k, v in libs.items()}
    stream = lambda: ctypes_stream()  # noqa: E731
    OUT["hip"] = res
    save()
    cu = torch.cuda.get_device_properties(0).multi_processor_count
    # 1) peak issue rate
    peak = {}
    for name in ("f16", "bf16", "iu8", "iu4"):
        lib = libs.get(name)
        if lib is None or isinstance(lib, str):
            continue
        try:
            blocks, iters = cu * 16, 4000
            out = torch.empty(blocks * 128, device=DEV)

            def run(lib=lib, blocks=blocks, iters=iters, out=out):
                rc = lib.wm_peak(blocks, iters, out.data_ptr(), stream())
                if rc:
                    raise RuntimeError(f"wm_peak rc={rc}")
            b = bench(run, warmup=2, reps=7, cold=False)
            ops = blocks * 4 * iters * 8 * 16 * 16 * 16 * 2
            peak[name] = dict(b, tops=round(ops / (b["ms"] * 1e-3) / 1e12, 2), blocks=blocks, iters=iters)
        except Exception as exc:  # noqa: BLE001
            peak[name] = err(exc)
    res["peak"] = peak
    OUT["hip"] = res
    save()
    # 2) fragment layout check on one 16x16x16 tile
    tile = {}
    for name in ("f16", "bf16", "iu8", "iu4"):
        lib = libs.get(name)
        if lib is None or isinstance(lib, str):
            continue
        try:
            lo, hi = (-8, 8) if name == "iu4" else (-20, 21)
            A = torch.randint(lo, hi, (16, 16), device=DEV)
            W = torch.randint(lo, hi, (16, 16), device=DEV)
            ref = (A.double() @ W.double().t())
            assumed = torch.empty(32, 8, device=DEV, dtype=torch.float64)
            for lane in range(32):
                for e in range(8):
                    assumed[lane, e] = ref[2 * e + lane // 16, lane % 16]
            variants = {}
            packs = [("lo_first", True), ("hi_first", False)] if name == "iu4" else [("plain", None)]
            for vname, lo_first in packs:
                if name == "iu4":
                    Ad, Wd = pack_int4(A, lo_first), pack_int4(W, lo_first)
                elif name == "iu8":
                    Ad, Wd = A.to(torch.int8).contiguous(), W.to(torch.int8).contiguous()
                else:
                    dt = torch.float16 if name == "f16" else torch.bfloat16
                    Ad, Wd = A.to(dt).contiguous(), W.to(dt).contiguous()
                out = torch.zeros(32 * 8, device=DEV)
                rc = lib.wm_tile(Ad.data_ptr(), Wd.data_ptr(), out.data_ptr(), stream())
                torch.cuda.synchronize()
                got = out.view(32, 8).double()
                variants[vname] = {"rc": rc, "match_assumed_layout": bool(torch.equal(got, assumed)),
                                   "raw": got.cpu().tolist() if not torch.equal(got, assumed) else None}
            tile[name] = {"variants": variants, "ref": ref.cpu().tolist() if not all(
                v["match_assumed_layout"] for v in variants.values()) else None}
        except Exception as exc:  # noqa: BLE001
            tile[name] = err(exc)
    res["tile"] = tile
    lo_first = tile.get("iu4", {}).get("variants", {}).get("hi_first", {}).get("match_assumed_layout") is not True
    res["iu4_packing_used"] = "lo_first" if lo_first else "hi_first"
    OUT["hip"] = res
    save()
    # 3) GEMMs
    gem = {}
    res["gemm"] = gem
    for (m, k, n) in SHAPES:
        r = {}
        gem[key(m, k, n)] = r
        try:
            xb = torch.randn(m, k, device=DEV)
            wf = torch.randn(n, k, device=DEV) * 0.02
            sa = (torch.rand(m, device=DEV) + 0.5) * 1e-2
            sw = (torch.rand(n, device=DEV) + 0.5) * 1e-2
            c = torch.empty(m, n, device=DEV, dtype=torch.bfloat16)
            q8a = torch.randint(-127, 128, (m, k), device=DEV, dtype=torch.int8)
            q8w = torch.randint(-127, 128, (n, k), device=DEV, dtype=torch.int8)
            q4a = torch.randint(-8, 8, (m, k), device=DEV, dtype=torch.int8)
            q4w = torch.randint(-8, 8, (n, k), device=DEV, dtype=torch.int8)
            G = 128
            sg = wf.view(n, k // G, G).abs().amax(-1).clamp_min(1e-12) / 7.0
            qg = (wf.view(n, k // G, G) / sg[..., None]).round().clamp(-8, 7).view(n, k)
            inputs = {
                "f16": (xb.half(), wf.half(), None, None, 0),
                "bf16": (xb.bfloat16(), wf.bfloat16(), None, None, 0),
                "iu8": (q8a, q8w, sa, sw, 0),
                "iu4": (pack_int4(q4a, lo_first), pack_int4(q4w, lo_first), sa, sw, 0),
                "w4a16": (xb.bfloat16(), pack_int4(qg, True), None, sg.float().contiguous(), G),
            }
            for name, (A, W, s_a, s_w, g) in inputs.items():
                lib = libs.get(name)
                if isinstance(lib, str) or lib is None:
                    continue
                per = {}
                best = None
                for cfg in range(4):
                    def run(cfg=cfg, A=A, W=W, s_a=s_a, s_w=s_w, g=g, lib=lib):
                        rc = lib.wm_gemm(cfg, A.data_ptr(), W.data_ptr(), c.data_ptr(),
                                         s_a.data_ptr() if s_a is not None else None,
                                         s_w.data_ptr() if s_w is not None else None, m, n, k, g, stream())
                        if rc:
                            raise RuntimeError(f"rc={rc}")
                    try:
                        b = bench(run, warmup=2, reps=6)
                        per[cfg] = round(b["ms"], 4)
                        if best is None or b["ms"] < best[1]:
                            best = (cfg, b["ms"], run)
                    except Exception as exc:  # noqa: BLE001
                        per[cfg] = repr(exc)[:200]
                e = {"per_cfg_ms": per}
                if best is not None:
                    cfg, _, run = best
                    b = bench(run)
                    e.update(best_cfg=cfg, ms=b["ms"], tops=tops(m, k, n, b["ms"]))
                    run()
                    torch.cuda.synchronize()
                    if name in ("f16", "bf16"):
                        ref = A[:64].float() @ W.float().t()
                    elif name == "iu8":
                        ref = (q8a[:64].float() @ q8w.float().t()) * sa[:64, None] * sw[None, :]
                    elif name == "iu4":
                        ref = (q4a[:64].float() @ q4w.float().t()) * sa[:64, None] * sw[None, :]
                    else:
                        deq = (qg.view(n, k // G, G) * sg[..., None]).view(n, k)
                        ref = A[:64].float() @ deq.t()
                    e["err"] = relerr(c[:64], ref)
                r[name] = e
                OUT["hip"] = res
                save()
            del xb, wf, q8a, q8w, q4a, q4w, c, inputs
        except Exception as exc:  # noqa: BLE001
            r["error"] = err(exc)
        OUT["hip"] = res
        save()


def ctypes_stream():
    return torch.cuda.current_stream().cuda_stream


# -------------------------------------------------------------------------------------------------------- tunable
def sec_tunable(a):
    res = {}
    tun = torch.cuda.tunable
    tun.enable(False)
    try:
        tun.set_filename(os.path.join(a.work, f"tunableop_results_{os.getpid()}.csv"))
    except Exception as exc:  # noqa: BLE001
        res["set_filename_error"] = repr(exc)
    for (m, k, n) in SHAPES:
        r = {}
        try:
            x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
            w = (torch.randn(n, k, device=DEV) * 0.02).to(torch.bfloat16)
            tun.enable(False)
            off = F.linear(x, w)
            r["off"] = bench(lambda: F.linear(x, w))
            tun.enable(True)
            tun.tuning_enable(True)
            torch.cuda.synchronize()
            t0 = time.time()
            on = F.linear(x, w)
            torch.cuda.synchronize()
            r["tuning_s"] = round(time.time() - t0, 2)
            r["on"] = bench(lambda: F.linear(x, w))
            r["speedup"] = round(r["off"]["ms"] / r["on"]["ms"], 3)
            r["tflops_off"] = tops(m, k, n, r["off"]["ms"])
            r["tflops_on"] = tops(m, k, n, r["on"]["ms"])
            r["err_on_vs_off"] = relerr(on, off)
            tun.enable(False)
        except Exception as exc:  # noqa: BLE001
            r["error"] = err(exc)
        res[key(m, k, n)] = r
        OUT["tunable"] = res
        save()
    try:
        res["results"] = [list(map(str, t)) for t in tun.get_results()]
        res["validators"] = [list(map(str, t)) for t in tun.get_validators()]
    except Exception as exc:  # noqa: BLE001
        res["results_error"] = repr(exc)
    OUT["tunable"] = res
    save()


# ------------------------------------------------------------------------------------------------------- accuracy
REPO = "black-forest-labs/FLUX.2-klein-4B"
FILE = "transformer/diffusion_pytorch_model.safetensors"
PICKS = [  # (tensor, row_start, rows)
    ("transformer_blocks.1.attn.to_q.weight", 0, 3072),
    ("transformer_blocks.1.attn.to_out.0.weight", 0, 3072),
    ("transformer_blocks.1.ff.linear_in.weight", 0, 3072),
    ("transformer_blocks.1.ff.linear_out.weight", 0, 3072),
    ("single_transformer_blocks.10.attn.to_qkv_mlp_proj.weight", 0, 3072),
    ("single_transformer_blocks.10.attn.to_qkv_mlp_proj.weight", 9216, 3072),
    ("single_transformer_blocks.10.attn.to_out.weight", 0, 3072),
]


def fetch_rows(url, header_len, info, row_start, rows):
    import urllib.request
    shape = info["shape"]
    cols = shape[1]
    base = 8 + header_len + info["data_offsets"][0]
    lo = base + row_start * cols * 2
    hi = lo + rows * cols * 2 - 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
    data = urllib.request.urlopen(req, timeout=300).read()
    assert len(data) == rows * cols * 2, (len(data), rows * cols * 2)
    return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).view(rows, cols).clone()


def q_sym(t, bits, dim=-1, group=None):
    """Symmetric absmax fake-quant along dim (last), optionally in groups of `group` along the last dim."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1)) if bits == 4 else -qmax
    if group:
        shp = t.shape
        tg = t.reshape(*shp[:-1], shp[-1] // group, group)
        s = tg.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
        return ((tg / s).round().clamp(qmin, qmax) * s).reshape(shp)
    s = t.abs().amax(dim, keepdim=True).clamp_min(1e-12) / qmax
    return (t / s).round().clamp(qmin, qmax) * s


def rotate(t, g, signs=None):
    h = sylvester(g).to(t.device, t.dtype)
    if signs is not None:
        t = t * signs
    shp = t.shape
    return (t.reshape(*shp[:-1], shp[-1] // g, g) @ h).reshape(shp)


def synth_acts(T, K, kind, gen):
    x = torch.randn(T, K, device=DEV, generator=gen)
    if kind == "outliers":
        idx = torch.randperm(K, device=DEV, generator=gen)[: max(1, K // 200)]
        x[:, idx] *= 30.0
        x *= torch.exp(0.5 * torch.randn(T, 1, device=DEV, generator=gen))
        spikes = torch.rand(T, K, device=DEV, generator=gen) < 1e-4
        x = torch.where(spikes, x * 100.0, x)
    return x


def sec_accuracy(a):
    import urllib.request
    res = {"repo": REPO, "file": FILE, "picks": PICKS}
    url = f"https://huggingface.co/{REPO}/resolve/main/{FILE}"
    try:
        n = int.from_bytes(urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-7"}),
                                                  timeout=120).read(), "little")
        hdr = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + n - 1}"}),
                                                timeout=120).read())
    except Exception as exc:  # noqa: BLE001
        res["fetch_error"] = err(exc)
        OUT["accuracy"] = res
        save()
        return
    gen = torch.Generator(device=DEV)
    gen.manual_seed(0)
    T = int(os.environ.get("GEMM_PROBE_ACC_T", "2048"))
    methods = [
        "bf16_act_round", "w8a16_perchan", "w8a8_pertoken_perchan", "w8a8_convrot256",
        "w4a16_perchan", "w4a16_g128", "w4a16_g64", "w4a8_perchan", "w4a8_g128",
        "w4a4_pertoken_perchan", "w4a4_g128_both", "w4a4_had256", "w4a4_had1024", "w4a4_randhad256",
        "w4a4_had256_g128_both", "w4a4_randhad1024",
    ]
    table = {m: {"gaussian": [], "outliers": []} for m in methods}
    per = {}
    t0 = time.time()
    for (name, r0, rows) in PICKS:
        try:
            info = hdr[name]
            rows = int(os.environ.get("GEMM_PROBE_ACC_ROWS", rows))
            w = fetch_rows(url, n, info, r0, rows).to(DEV).float()
        except Exception as exc:  # noqa: BLE001
            per[f"{name}@{r0}"] = err(exc)
            continue
        K = w.shape[1]
        signs256 = (torch.randint(0, 2, (K,), device=DEV, generator=gen) * 2 - 1).float()
        signs1024 = (torch.randint(0, 2, (K,), device=DEV, generator=gen) * 2 - 1).float()
        pe = {"shape": [rows, K], "w_absmax": w.abs().max().item(), "w_std": w.std().item()}
        for kind in ("gaussian", "outliers"):
            x = synth_acts(T, K, kind, gen)
            ref = x.double() @ w.double().t()

            def rel(y):
                return ((y.double() - ref).norm() / ref.norm()).item()
            out = {}
            out["bf16_act_round"] = rel(x.bfloat16().float() @ w.t())
            out["w8a16_perchan"] = rel(x @ q_sym(w, 8).t())
            out["w8a8_pertoken_perchan"] = rel(q_sym(x, 8) @ q_sym(w, 8).t())
            out["w8a8_convrot256"] = rel(q_sym(rotate(x, 256), 8) @ q_sym(rotate(w, 256), 8).t())
            out["w4a16_perchan"] = rel(x @ q_sym(w, 4).t())
            out["w4a16_g128"] = rel(x @ q_sym(w, 4, group=128).t())
            out["w4a16_g64"] = rel(x @ q_sym(w, 4, group=64).t())
            out["w4a8_perchan"] = rel(q_sym(x, 8) @ q_sym(w, 4).t())
            out["w4a8_g128"] = rel(q_sym(x, 8, group=128) @ q_sym(w, 4, group=128).t())
            out["w4a4_pertoken_perchan"] = rel(q_sym(x, 4) @ q_sym(w, 4).t())
            out["w4a4_g128_both"] = rel(q_sym(x, 4, group=128) @ q_sym(w, 4, group=128).t())
            out["w4a4_had256"] = rel(q_sym(rotate(x, 256), 4) @ q_sym(rotate(w, 256), 4).t())
            out["w4a4_had1024"] = rel(q_sym(rotate(x, 1024), 4) @ q_sym(rotate(w, 1024), 4).t())
            out["w4a4_randhad256"] = rel(q_sym(rotate(x, 256, signs256), 4) @ q_sym(rotate(w, 256, signs256), 4).t())
            out["w4a4_randhad1024"] = rel(q_sym(rotate(x, 1024, signs1024), 4) @ q_sym(rotate(w, 1024, signs1024), 4).t())
            out["w4a4_had256_g128_both"] = rel(q_sym(rotate(x, 256), 4, group=128) @ q_sym(rotate(w, 256), 4, group=128).t())
            pe[kind] = out
            for mth, v in out.items():
                table[mth][kind].append(v)
            del x, ref
        per[f"{name}@{r0}"] = pe
        res["per_weight"] = per
        OUT["accuracy"] = res
        save()
    res["mean_rel_fro"] = {m: {k: (sum(v) / len(v) if v else None) for k, v in d.items()} for m, d in table.items()}
    res["seconds"] = round(time.time() - t0, 1)
    OUT["accuracy"] = res
    save()


# ---------------------------------------------------------------------------------------------------------- block
def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype) * w


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


def sec_block(a):
    res = {}
    D, H, HD = 3072, 24, 128
    bf = torch.bfloat16
    for S in (4096,):
        txt = 512
        L = S + txt
        pos = torch.arange(L, device=DEV).float()[:, None]
        freq = 1.0 / (10000 ** (torch.arange(0, HD, 2, device=DEV).float() / HD))
        cos, sin = (pos * freq).cos().to(bf), (pos * freq).sin().to(bf)
        nw = lambda o, i: (torch.randn(o, i, device=DEV) * 0.02).to(bf)  # noqa: E731
        nb = lambda o: (torch.randn(o, device=DEV) * 0.01).to(bf)  # noqa: E731
        # Qwen-Image-like MMDiT double block: per stream q,k,v,out (D x D), MLP D->4D->D (GELU), joint attention.
        Wq = {s: [nw(D, D) for _ in range(4)] + [nw(4 * D, D), nw(D, 4 * D)] for s in ("img", "txt")}
        Bq = {s: [nb(D) for _ in range(4)] + [nb(4 * D), nb(D)] for s in ("img", "txt")}
        nq, nk = torch.ones(HD, device=DEV, dtype=bf), torch.ones(HD, device=DEV, dtype=bf)
        mod = {s: [(torch.randn(D, device=DEV) * 0.1).to(bf) for _ in range(6)] for s in ("img", "txt")}
        xi = torch.randn(1, S, D, device=DEV, dtype=bf)
        xt = torch.randn(1, txt, D, device=DEV, dtype=bf)
        lin_log = []

        def L_(x, w, b, tag):
            if lin_log is not None and isinstance(lin_log, list):
                lin_log.append((x.shape[-2] * (x.shape[0] if x.dim() == 3 else 1), w.shape[1], w.shape[0], b is not None))
            return F.linear(x, w, b)

        def double_block(xi, xt):
            hs = {}
            for s, x in (("img", xi), ("txt", xt)):
                sh, sc, g1, sh2, sc2, g2 = mod[s]
                h = F.layer_norm(x, (D,)) * (1 + sc) + sh
                q, k, v = (L_(h, Wq[s][j], Bq[s][j], f"{s}_qkv{j}") for j in range(3))
                hs[s] = (q, k, v)
            q = torch.cat([hs["txt"][0], hs["img"][0]], 1).view(1, L, H, HD)
            k = torch.cat([hs["txt"][1], hs["img"][1]], 1).view(1, L, H, HD)
            v = torch.cat([hs["txt"][2], hs["img"][2]], 1).view(1, L, H, HD)
            q, k = rope(rms(q, nq), cos[:, None], sin[:, None]), rope(rms(k, nk), cos[:, None], sin[:, None])
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            o = o.transpose(1, 2).reshape(1, L, D)
            outs = []
            for s, x, sl in (("txt", xt, slice(0, txt)), ("img", xi, slice(txt, L))):
                sh, sc, g1, sh2, sc2, g2 = mod[s]
                x = x + g1 * L_(o[:, sl], Wq[s][3], Bq[s][3], f"{s}_out")
                h = F.layer_norm(x, (D,)) * (1 + sc2) + sh2
                x = x + g2 * L_(F.gelu(L_(h, Wq[s][4], Bq[s][4], f"{s}_fc1"), approximate="tanh"), Wq[s][5], Bq[s][5], f"{s}_fc2")
                outs.append(x)
            return outs[1], outs[0]

        # FLUX.2-like single block: fused qkv+mlp_in (D -> 3D + 2*3D), SwiGLU, fused out (D + 3D -> D), no bias.
        W1 = nw(3 * D + 6 * D, D)
        W2 = nw(D, D + 3 * D)
        smod = [(torch.randn(D, device=DEV) * 0.1).to(bf) for _ in range(3)]
        xs = torch.randn(1, L, D, device=DEV, dtype=bf)

        def single_block(x):
            sh, sc, g = smod
            h = F.layer_norm(x, (D,)) * (1 + sc) + sh
            p = L_(h, W1, None, "qkv_mlp")
            qkv, mlp = p.split([3 * D, 6 * D], -1)
            q, k, v = qkv.view(1, L, 3, H, HD).unbind(2)
            q, k = rope(rms(q, nq), cos[:, None], sin[:, None]), rope(rms(k, nk), cos[:, None], sin[:, None])
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            o = o.transpose(1, 2).reshape(1, L, D)
            m1, m2 = mlp.chunk(2, -1)
            return x + g * L_(torch.cat([o, F.silu(m1) * m2], -1), W2, None, "out")

        with torch.no_grad():
            for label, fn in (("qwen_like_double", lambda: double_block(xi, xt)), ("flux2_like_single", lambda: single_block(xs))):
                r = {}
                try:
                    lin_log.clear()
                    fn()
                    shapes = list(lin_log)
                    lin_log.clear()
                    blk = bench(fn)
                    r["block_ms"] = blk["ms"]
                    gemm_ms = 0.0
                    per = []
                    for (mm, kk, nn_, hasb) in shapes:
                        xx = torch.randn(mm, kk, device=DEV, dtype=bf)
                        ww = nw(nn_, kk)
                        bb = nb(nn_) if hasb else None
                        t = bench(lambda: F.linear(xx, ww, bb))["ms"]
                        gemm_ms += t
                        per.append([mm, kk, nn_, round(t, 3)])
                    qq = torch.randn(1, H, L, HD, device=DEV, dtype=bf)
                    attn_ms = bench(lambda: F.scaled_dot_product_attention(qq, qq, qq))["ms"]
                    r.update(gemm_ms_sum=round(gemm_ms, 3), attn_ms=round(attn_ms, 3), linears=per,
                             gemm_share=round(gemm_ms / blk["ms"], 3), attn_share=round(attn_ms / blk["ms"], 3),
                             other_share=round(1 - (gemm_ms + attn_ms) / blk["ms"], 3), tokens=L)
                except Exception as exc:  # noqa: BLE001
                    r.update(err(exc))
                res[label] = r
                OUT["block"] = res
                save()
        # profiler cross-check on the single block (kernel-level categories)
        try:
            from torch.profiler import profile, ProfilerActivity
            with torch.no_grad():
                for _ in range(2):
                    single_block(xs)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(3):
                        single_block(xs)
                    torch.cuda.synchronize()
            cats = {"gemm": 0.0, "attention": 0.0, "other": 0.0}
            top = []
            for ev in prof.key_averages():
                t = getattr(ev, "device_time_total", None)
                if t is None:
                    t = getattr(ev, "cuda_time_total", 0.0)
                if not t or ev.key.startswith(("ProfilerStep", "aten::", "cuda", "hip")):
                    continue
                nm = ev.key.lower()
                if any(s in nm for s in ("cijk", "gemm", "wmma", "matmul", "_mm", "hipblaslt", "tensile")):
                    cats["gemm"] += t
                elif any(s in nm for s in ("attn", "attention", "fmha", "flash", "aotriton", "sdpa")):
                    cats["attention"] += t
                else:
                    cats["other"] += t
                top.append([ev.key[:120], round(t / 3 / 1000, 3)])
            tot = sum(cats.values()) or 1.0
            res["profiler_single_block"] = {"share": {k: round(v / tot, 3) for k, v in cats.items()},
                                            "per_block_ms": round(tot / 3 / 1000, 3),
                                            "top": sorted(top, key=lambda x: -x[1])[:15]}
        except Exception as exc:  # noqa: BLE001
            res["profiler_single_block"] = err(exc)
        OUT["block"] = res
        save()


# --------------------------------------------------------------------------------------------------------- inline
def sec_inline(a):
    res = {}
    try:
        from torch.utils import cpp_extension
        res["ROCM_HOME"] = str(getattr(cpp_extension, "ROCM_HOME", None))
        src_cuda = r"""
#include <torch/extension.h>
__global__ void add1(float* x, int n){ int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += 1.0f; }
void run(torch::Tensor x){ int n = x.numel(); add1<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), n); }
"""
        os.makedirs(a.inline_dir, exist_ok=True)
        t0 = time.time()
        mod = cpp_extension.load_inline(name="gemm_lowp_inline_smoke", cpp_sources="void run(torch::Tensor x);",
                                        cuda_sources=src_cuda, functions=["run"], verbose=False,
                                        build_directory=a.inline_dir, extra_cuda_cflags=["-O2"])
        res["build_s"] = round(time.time() - t0, 1)
        x = torch.zeros(1000, device=DEV)
        mod.run(x)
        torch.cuda.synchronize()
        res["ok"] = bool((x == 1).all().item())
    except Exception as exc:  # noqa: BLE001
        res.update(err(exc))
    OUT["inline"] = res
    save()


SECTIONS = {"env": sec_env, "dense": sec_dense, "intmm": sec_intmm, "triton": sec_triton, "compile": sec_compile,
            "hip": sec_hip, "tunable": sec_tunable, "accuracy": sec_accuracy, "block": sec_block,
            "inline": sec_inline}


def main():
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True, choices=sorted(SECTIONS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib-dir", default="")
    ap.add_argument("--work", default=os.environ.get("AMD_CI_WORK", "."))
    ap.add_argument("--inline-dir", default="")
    ap.add_argument("--kinds", default="f16,bf16,iu8,iu4,w4a16")
    ap.add_argument("--shapes", default="", help="debug: comma list of MxKxN to restrict")
    a = ap.parse_args()
    OUT_PATH = a.out
    if a.shapes:
        global SHAPES
        SHAPES = [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(",")]
    OUT["section"] = a.section
    OUT["torch"] = torch.__version__
    OUT["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    try:
        OUT["device"] = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else None
        OUT["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if torch.cuda.is_available() else None
        OUT["hip_version"] = getattr(torch.version, "hip", None)
    except Exception as exc:  # noqa: BLE001
        OUT["device_error"] = repr(exc)
    save()
    t0 = time.time()
    try:
        SECTIONS[a.section](a)
        OUT["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        OUT["status"] = "crashed"
        OUT["crash"] = err(exc)
    OUT["seconds"] = round(time.time() - t0, 1)
    save()


if __name__ == "__main__":
    main()
