#!/usr/bin/env python3
"""gfx1151 INT8 GEMM kernel arms at DiT linear shapes (observes only; one JSON via --out).

  kern_gemm.py a1     --backend B --out F        GROUP_M {1,2,4,8} x {contiguous out, out row pitch N+128}  (A1 / carbon6 #1)
  kern_gemm.py bufops --backend B --out F        Studio's tile at M 4608 / 9728 (run with AMDGCN_USE_BUFFER_OPS=0/1)
  kern_gemm.py k338   --backend B --out F --kitchen PKG_ROOT --kname NAME
                                                 Studio w8a8_mm vs comfy-kitchen _C.int8_gemm (row-major A) (C9-3)

Every arm's output is compared with torch.equal to Studio's own kernel output (int32 accumulate, fp32 epilogue) and
Studio's output to ``reference_mm`` (fp64 accumulate). Timing: CUDA events over n calls, arms interleaved
(order reversed every round), median of --rounds rounds; the ``studio_aa`` arm is Studio's tile timed as a separate
arm (in-process A/A)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time

# (name, K, N). Qwen-Image-2.1: hidden 4096 (32 heads x 128), mlp_ratio 3 -> 12288. comfy-kitchen #338's own shapes
# (Qwen-Image 2.1 Turbo route, as the PR / triage list them, N x K): 3072x3072, 12288x3072, 3072x12288.
Q21 = [("q21_qkv", 4096, 12288), ("q21_out", 4096, 4096), ("q21_down", 12288, 4096)]
K338 = [("k338_sq", 3072, 3072), ("k338_up", 3072, 12288), ("k338_down", 12288, 3072)]
KLEIN = [("kl_fused", 3072, 18432), ("kl_down", 9216, 3072)]
M_RACE = [4096, 4352, 2049, 4097]
M_BUF = [4608, 9728]
PAD = 128


def strided_kernel():
    """Studio's w8a8_mm (diffusion_native_w8a8_rdna) with one change: the output row stride is an argument."""
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize = ["M"])
    def w8a8_mm_strided(a_ptr, w_ptr, xs_ptr, ws_ptr, b_ptr, c_ptr, M, N, K, stride_cm,
                        HAS_BIAS: tl.constexpr, EVEN_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
        rn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + rm[:, None] * K + offs_k[None, :]
        w_ptrs = w_ptr + rn[None, :] * K + offs_k[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.int32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(w_ptrs)
            else:
                kmask = offs_k < K - k * BLOCK_K
                a = tl.load(a_ptrs, mask = kmask[None, :], other = 0)
                b = tl.load(w_ptrs, mask = kmask[:, None], other = 0)
            acc = tl.dot(a, b, acc, out_dtype = tl.int32)
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        xs = tl.load(xs_ptr + rm)
        ws = tl.load(ws_ptr + rn)
        y = acc.to(tl.float32) * xs[:, None]
        y = y * ws[None, :]
        if HAS_BIAS:
            y = y + tl.load(b_ptr + rn).to(tl.float32)[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, y.to(c_ptr.dtype.element_ty), mask = mask)
    return w8a8_mm_strided


def inputs(m, k, n, seed):
    import torch
    g = torch.Generator(device = "cuda").manual_seed(seed)
    xq = torch.randint(-127, 128, (m, k), generator = g, device = "cuda", dtype = torch.int8)
    wq = torch.randint(-127, 128, (n, k), generator = g, device = "cuda", dtype = torch.int8)
    xs = torch.rand(m, generator = g, device = "cuda") * 1e-3 + 1e-4
    ws = torch.rand(n, generator = g, device = "cuda") * 1e-3 + 1e-4
    bias = torch.randn(n, generator = g, device = "cuda").to(torch.bfloat16)
    return xq, wq, xs, ws, bias


def launch_studio(kern, cfg, xq, wq, xs, ws, bias, out):
    bm, bn, bk, gm, warps, stages = cfg[:6]
    m, k = xq.shape
    n = wq.shape[0]
    kern.w8a8_mm[(kern.triton.cdiv(m, bm) * kern.triton.cdiv(n, bn),)](
        xq, wq, xs, ws, bias, out, m, n, k, HAS_BIAS = True, EVEN_K = k % bk == 0, BLOCK_M = bm, BLOCK_N = bn,
        BLOCK_K = bk, GROUP_M = gm, num_warps = warps, num_stages = stages, enable_fp_fusion = False)


def launch_strided(skern, triton, cfg, xq, wq, xs, ws, bias, out_view):
    bm, bn, bk, gm, warps, stages = cfg[:6]
    m, k = xq.shape
    n = wq.shape[0]
    skern[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        xq, wq, xs, ws, bias, out_view, m, n, k, out_view.stride(0), HAS_BIAS = True, EVEN_K = k % bk == 0,
        BLOCK_M = bm, BLOCK_N = bn, BLOCK_K = bk, GROUP_M = gm, num_warps = warps, num_stages = stages,
        enable_fp_fusion = False)


def time_round(fn, n_calls):
    import torch
    a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    a.record()
    for _ in range(n_calls):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / n_calls


def race(fns: dict, rounds: int) -> dict:
    """fns: name -> callable. Interleaved rounds; returns name -> {ms list, med}."""
    import torch
    for f in fns.values():
        f()
    torch.cuda.synchronize()
    est = time_round(next(iter(fns.values())), 3)
    n_calls = int(max(3, min(50, 30.0 / max(est, 1e-3))))
    res = {k: {"ms": []} for k in fns}
    names = list(fns)
    for r in range(rounds):
        for name in (names if r % 2 == 0 else names[::-1]):
            res[name]["ms"].append(time_round(fns[name], n_calls))
    for name in names:
        res[name]["med"] = st.median(res[name]["ms"])
    res["_n_calls"] = n_calls
    return res


def sha(t):
    return hashlib.sha256(t.contiguous().view(-1).view(__import__("torch").uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def mode_a1(R, kern, rounds, obs):
    import torch
    import triton
    skern = strided_kernel()
    rows = []
    for name, k, n in Q21:
        for m in M_RACE:
            xq, wq, xs, ws, bias = inputs(m, k, n, 7)
            base = tuple(R._mm_cfg(m, k))
            outs, fns, row = {}, {}, {"shape": name, "m": m, "k": k, "n": n, "studio_cfg": list(base)}
            ref = R.reference_mm(xq, xs, wq, ws, bias, torch.bfloat16)
            for gm in (1, 2, 4, 8):
                cfg = base[:3] + (gm,) + base[4:]
                o = torch.empty((m, n), device = "cuda", dtype = torch.bfloat16)
                outs[f"gm{gm}"] = o
                fns[f"gm{gm}"] = (lambda cfg = cfg, o = o: launch_studio(kern, cfg, xq, wq, xs, ws, bias, o))
                buf = torch.empty((m, n + PAD), device = "cuda", dtype = torch.bfloat16)
                ov = buf[:, :n]
                outs[f"pad_gm{gm}"] = ov
                fns[f"pad_gm{gm}"] = (lambda cfg = cfg, ov = ov: launch_strided(skern, triton, cfg, xq, wq, xs, ws, bias, ov))
            o = torch.empty((m, n), device = "cuda", dtype = torch.bfloat16)
            outs["copy_contig"] = o  # the strided copy at stride N: separates "copy of the kernel" from "pitch"
            fns["copy_contig"] = (lambda o = o: launch_strided(skern, triton, base, xq, wq, xs, ws, bias, o))
            o2 = torch.empty((m, n), device = "cuda", dtype = torch.bfloat16)
            outs["studio_aa"] = o2
            fns["studio_aa"] = (lambda o2 = o2: launch_studio(kern, base, xq, wq, xs, ws, bias, o2))
            try:
                res = race(fns, rounds)
                studio_name = f"gm{base[3]}"
                ref_out = outs[studio_name]
                row["studio_eq_reference_mm"] = bool(torch.equal(ref_out, ref))
                row["studio_maxabs_vs_reference_mm"] = float((ref_out.float() - ref.float()).abs().max())
                for nm, o in outs.items():
                    res[nm]["exact_vs_studio"] = bool(torch.equal(o, ref_out))
                row["res"] = res
            except Exception as exc:  # noqa: BLE001
                row["error"] = repr(exc)[:400]
            rows.append(row)
            obs["rows"] = rows
            del xq, wq, outs, fns
            torch.cuda.empty_cache()


def mode_bufops(R, kern, rounds, obs):
    import torch
    rows = []
    obs["buffer_ops_env"] = os.environ.get("AMDGCN_USE_BUFFER_OPS")
    for name, k, n in Q21 + KLEIN:
        for m in M_BUF:
            xq, wq, xs, ws, bias = inputs(m, k, n, 11)
            cfg = tuple(R._mm_cfg(m, k))
            o, o2 = (torch.empty((m, n), device = "cuda", dtype = torch.bfloat16) for _ in range(2))
            row = {"shape": name, "m": m, "k": k, "n": n, "cfg": list(cfg)}
            try:
                res = race({"studio": lambda: launch_studio(kern, cfg, xq, wq, xs, ws, bias, o),
                            "studio_aa": lambda: launch_studio(kern, cfg, xq, wq, xs, ws, bias, o2)}, rounds)
                row["res"] = res
                row["sha"] = sha(o)
                row["aa_equal"] = bool(torch.equal(o, o2))
            except Exception as exc:  # noqa: BLE001
                row["error"] = repr(exc)[:400]
            rows.append(row)
            obs["rows"] = rows
            del xq, wq, o, o2
            torch.cuda.empty_cache()


def mode_k338(R, kern, rounds, obs, kitchen, kname):
    import torch
    obs["kitchen"], obs["kname"] = kitchen, kname
    sys.path.insert(0, kitchen)
    try:
        from comfy_kitchen.backends import hip as KH
        obs["kitchen_file"] = KH.__file__
        C = KH._C
        codes = getattr(KH, "DTYPE_TO_CODE", None)
        if codes is None:
            from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE as codes
        obs["kitchen_has_wmma"] = bool(KH.has_wmma())
    except Exception as exc:  # noqa: BLE001
        obs["kitchen_error"] = repr(exc)[:600]
        return
    rows = []
    for name, k, n in K338 + Q21:
        for m in M_RACE:
            xq, wq, xs, ws, bias = inputs(m, k, n, 13)
            cfg = tuple(R._mm_cfg(m, k))
            row = {"shape": name, "m": m, "k": k, "n": n, "cfg": list(cfg)}
            try:
                row["kitchen_would_tile"] = bool(KH._use_tiled_int8(m, n, k, xq.device))
            except Exception as exc:  # noqa: BLE001
                row["kitchen_would_tile"] = repr(exc)[:120]
            o_s = torch.empty((m, n), device = "cuda", dtype = torch.bfloat16)
            o_k = torch.empty((m, n), device = "cuda", dtype = torch.bfloat16)
            wq_a = KH._aligned(wq.contiguous())
            xs_f, ws_f = xs.reshape(-1).contiguous(), ws.reshape(-1).contiguous()
            b_k = KH._bias_operand(bias, n, xq.device)
            stream = KH._stream(xq)

            def kit():
                C.int8_gemm(KH._dl(xq), KH._dl(wq_a), KH._dl(o_k), KH._dl(xs_f), KH._dl(ws_f), 1, KH._dl(b_k),
                            m, n, k, codes[torch.bfloat16], stream)
            fns = {"studio": lambda: launch_studio(kern, cfg, xq, wq, xs, ws, bias, o_s), "kitchen": kit}
            # round-9 fold-in (carbon6 #5): rectangular Studio tiles 256x128x64 / 128x256x64 as extra race arms
            # (GROUP_M / warps / stages from the Studio cfg); a tile that fails to compile is recorded, not raced.
            tile_out = {}
            for tname, (tbm, tbn, tbk) in (("t256x128x64", (256, 128, 64)), ("t128x256x64", (128, 256, 64))):
                tcfg = (tbm, tbn, tbk) + tuple(cfg[3:6])
                o_t = torch.empty((m, n), device = "cuda", dtype = torch.bfloat16)
                try:
                    launch_studio(kern, tcfg, xq, wq, xs, ws, bias, o_t)
                    torch.cuda.synchronize()
                    tile_out[tname] = o_t
                    fns[tname] = (lambda c = tcfg, o = o_t: launch_studio(kern, c, xq, wq, xs, ws, bias, o))
                    row.setdefault("tile_cfg", {})[tname] = list(tcfg)
                except Exception as exc:  # noqa: BLE001
                    row.setdefault("tile_error", {})[tname] = repr(exc)[:300]
            try:
                res = race(fns, rounds)
                ref = R.reference_mm(xq, xs, wq, ws, bias, torch.bfloat16)
                row["res"] = res
                row["kitchen_eq_studio"] = bool(torch.equal(o_k, o_s))
                row["kitchen_eq_reference_mm"] = bool(torch.equal(o_k, ref))
                row["studio_eq_reference_mm"] = bool(torch.equal(o_s, ref))
                row["kitchen_maxabs_vs_reference_mm"] = float((o_k.float() - ref.float()).abs().max())
                row["kitchen_sha"] = sha(o_k)
                row["kitchen_finite"] = bool(torch.isfinite(o_k).all())
                for tname, o_t in tile_out.items():
                    row.setdefault("tile_eq_studio", {})[tname] = bool(torch.equal(o_t, o_s))
                    row.setdefault("tile_eq_reference_mm", {})[tname] = bool(torch.equal(o_t, ref))
            except Exception as exc:  # noqa: BLE001
                row["error"] = repr(exc)[:400]
            rows.append(row)
            obs["rows"] = rows
            del xq, wq, wq_a, o_s, o_k, tile_out, fns
            torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices = ("a1", "bufops", "k338"))
    ap.add_argument("--backend", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--rounds", type = int, default = 9)
    ap.add_argument("--kitchen", default = "")
    ap.add_argument("--kname", default = "")
    a = ap.parse_args()
    obs = {"mode": a.mode, "t0": time.time()}

    def save():
        open(a.out, "w", encoding = "utf-8").write(json.dumps(obs, default = str))
    try:
        sys.path.insert(0, a.backend)
        import torch
        import triton

        from core.inference import diffusion_native_w8a8_rdna as R
        dev = torch.device("cuda", 0)
        obs.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None), triton = triton.__version__,
                   arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
                   prepare = R.prepare(dev), studio_cfg = {k: list(v) for k, v in R._CFG.items()})
        kern = R._kernels()
        if a.mode == "a1":
            mode_a1(R, kern, a.rounds, obs)
        elif a.mode == "bufops":
            mode_bufops(R, kern, a.rounds, obs)
        else:
            mode_k338(R, kern, a.rounds, obs, a.kitchen, a.kname)
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = traceback.format_exc()[-1500:]
    obs["seconds"] = round(time.time() - obs["t0"], 1)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
