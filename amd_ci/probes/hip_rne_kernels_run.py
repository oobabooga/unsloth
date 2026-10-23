#!/usr/bin/env python3
"""One arm of the HIP 16-bit rounding kernel probe, in its own process. Observes only, writes JSON to --out.

--kind zoo  imports the state's unsloth_zoo.mxfp4_dequant and compares the fused kernel with the torch
            reference bit for bit on (a) every byte under every e8m0 scale and (b) the self-check's own
            data, bf16 / fp16, both layouts; records mxfp4_kernel_available, and per-call ms of the
            product route vs the torch path on the gpt-oss-20b gate_up stack.
--kind fp8  imports the state's unsloth.kernels.fp8 and compares the e4m3fn byte decoder and the
            fp8e4nv weight_dequant_block kernel with the torch loop, fp16 / bf16 output, on every byte
            under 256 x 128 per-block scales (incl. double-rounding ties, fp16 subnormal / overflow
            products) and on a realistic 4096 x 4096 128-block weight; per-call ms on 4096 x 14336.
Mismatches are classified as neg_zero (torch -0, kernel +0), ftz (torch nonzero subnormal, kernel 0),
single_round (kernel = exact product rounded once to fp16; FP8 only), or other.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback

import numpy as np


def _classify(torch, got, want, exact = None):
    gi = got.contiguous().view(torch.int16).view(-1)
    wi = want.contiguous().view(torch.int16).view(-1)
    g, w = got.reshape(-1), want.reshape(-1)
    nan = torch.isnan(g) & torch.isnan(w)
    bad = torch.nonzero((gi != wi) & ~nan).view(-1)
    out = {"n": int(gi.numel()), "mismatch": int(bad.numel()), "nan_bits_differ": int(((gi != wi) & nan).sum())}
    if not bad.numel():
        return out
    gb, wb = g[bad], w[bad]
    zero = (gb == 0) & (wb == 0)
    negz = zero & torch.signbit(wb) & ~torch.signbit(gb)
    ftz = (gb == 0) & (wb != 0) & (wb.float().abs() < torch.finfo(want.dtype).tiny)
    cls = {"neg_zero": int(negz.sum()), "pos_zero": int((zero & ~negz).sum()), "ftz": int(ftz.sum())}
    rest = ~zero & ~ftz
    if exact is not None and want.dtype == torch.float16:
        p = exact.reshape(-1)[bad].cpu().numpy()
        once = torch.from_numpy(p.astype(np.float16).view(np.int16).astype(np.int64)).to(gi.device)
        sr = rest & ((gi[bad].long() & 0xFFFF) == (once & 0xFFFF))
        cls["single_round"] = int(sr.sum())
        rest = rest & ~sr
    cls["other"] = int(rest.sum())
    out["classes"] = {k: v for k, v in cls.items() if v}
    out["examples"] = [{"i": int(i), "torch": hex(int(wi[i]) & 0xFFFF), "kernel": hex(int(gi[i]) & 0xFFFF)} for i in bad[:8].tolist()]
    return out


def _measure(torch, fn, n = 10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing = True) for _ in range(n)]
    en = [torch.cuda.Event(enable_timing = True) for _ in range(n)]
    for i in range(n):
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in zip(st, en))


def _timed(torch, arms, rounds = 3):
    res = {a: [] for a in arms}
    for r in range(rounds):
        for a in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
            res[a].append(_measure(torch, arms[a]))
    return {a: dict(median_ms = statistics.median(v), runs = v) for a, v in res.items()}


def run_zoo(torch, rec, quick):
    import unsloth_zoo.mxfp4_dequant as M

    rec.update(module_file = M.__file__, has_triton = M._HAS_TRITON, is_hip_flag = getattr(M, "_IS_HIP", None),
               exact_round = (M._exact_round() if hasattr(M, "_exact_round") else None))
    dev = torch.device("cuda", 0)
    byte = torch.arange(256, device = dev, dtype = torch.int32)
    every = (byte.view(1, 1, 256).expand(256, 1, 256).reshape(2, 128, 16, 16).to(torch.uint8),
             torch.arange(256, device = dev, dtype = torch.int32).view(256, 1).expand(256, 16).reshape(2, 128, 16).to(torch.uint8).contiguous())

    def selfcheck_data(dt):
        E, N, G = 2, 131, 9
        blocks = ((torch.arange(E * N * G * 16, device = dev) * 7) % 256).to(torch.uint8).reshape(E, N, G, 16)
        scales = (112 + (torch.arange(E * N * G, device = dev) * 5) % 24) if dt == torch.float16 else (torch.arange(E * N * G, device = dev) * 37) % 255
        scales = scales.to(torch.uint8).reshape(E, N, G)
        edges = [0, 127, 254, 255] if dt != torch.float16 else [0, 127, 135, 255]
        scales.view(-1)[: len(edges)] = torch.tensor(edges, dtype = torch.uint8)
        return blocks, scales

    parity, avail = {}, {}
    for dt in (torch.bfloat16, torch.float16):
        name = str(dt).split(".")[-1]
        M._KERNEL_VERIFIED.clear()
        avail[name] = {"transposed": bool(M.mxfp4_kernel_available(dtype = dt)), "plain": bool(M._kernel_verified(dev, dt, False))}
        for data_name, (blocks, scales) in (("every_byte_x_scale", every), ("selfcheck", selfcheck_data(dt))):
            for transpose in (False, True):
                key = f"{name}/{data_name}/{'T' if transpose else 'plain'}"
                try:
                    got = M._kernel_dequantize(blocks, scales, dt, transpose, None, None, None)
                    want = M.mxfp4_dequantize_torch(blocks, scales, dtype = dt, transpose = transpose)
                    torch.cuda.synchronize()
                    parity[key] = _classify(torch, got, want)
                except Exception as e:  # noqa: BLE001
                    parity[key] = {"error": f"{type(e).__name__}: {e}"[:800]}
    rec.update(available = avail, parity = parity)
    g = torch.Generator(device = "cuda").manual_seed(0)
    E, R, K = (32, 5760, 2880) if not quick else (4, 5760, 2880)
    blocks = torch.randint(0, 256, (E, R, K // 32, 16), dtype = torch.uint8, device = "cuda", generator = g)
    scales = torch.randint(118, 128, (E, R, K // 32), dtype = torch.uint8, device = "cuda", generator = g)
    timing = {}
    for dt in (torch.bfloat16, torch.float16):
        name = str(dt).split(".")[-1]
        timing[name] = _timed(torch, {
            "product": lambda dt = dt: M.mxfp4_dequantize(blocks, scales, dtype = dt, transpose = True),
            "kernel": lambda dt = dt: M._kernel_dequantize(blocks, scales, dt, True, None, None, None),
            "torch": lambda dt = dt: M.mxfp4_dequantize_torch(blocks, scales, dtype = dt, transpose = True),
        })
    rec["timing_gate_up_T"] = timing


def run_fp8(torch, rec, quick):
    from unsloth.kernels import fp8 as F

    rec.update(module_file = F.__file__, exact_flag = getattr(F, "_FP8_EXACT_F16_ROUND", None),
               exact_f16 = (F._fp8_exact_f16_round(torch.float16) if hasattr(F, "_fp8_exact_f16_round") else None))
    dev = torch.device("cuda", 0)
    g = torch.Generator(device = "cpu").manual_seed(0)
    rows, cols, blk = 256, 128, 16
    w = torch.arange(256, dtype = torch.uint8).view(16, 16).repeat(rows, cols)
    s = torch.rand(rows, cols, generator = g, dtype = torch.float32) * 1e-2 + 1e-4
    wide = (torch.rand(rows // 4, cols, generator = g, dtype = torch.float64) + 1.0) * 2.0 ** torch.randint(-40, 11, (rows // 4, cols), generator = g).double()
    s[: rows // 4] = wide.float()
    every = (w.view(torch.float8_e4m3fn).to(dev), s.to(dev), blk)
    g2 = torch.Generator(device = "cpu").manual_seed(1)
    real_w = (torch.randn(4096, 4096, generator = g2) * 0.3).to(torch.float8_e4m3fn).to(dev)
    real_s = (torch.rand(32, 32, generator = g2) * 1e-2 + 1e-4).to(dev)
    real = (real_w, real_s, 128)
    F._fp8_device_lacks_kernel.cache_clear()
    rec["fp8e4nv_supported"] = not F._fp8_kernel_unsupported(real_w)
    parity = {}
    for data_name, (wt, sc, b) in (("every_byte_x_scale", every), ("real_4096_b128", real)):
        exact = (wt.double() * sc.double().repeat_interleave(b, 0).repeat_interleave(b, 1)).cpu()
        for dt in (torch.float16, torch.bfloat16):
            name = str(dt).split(".")[-1]
            want = F._torch_blockwise_dequant(wt, sc, [b, b], dt)
            for kname in ("bytes", "fp8e4nv"):
                key = f"{kname}/{name}/{data_name}"
                try:
                    if kname == "bytes":
                        got = F._e4m3fn_bytes_dequant(wt, sc, b, b, dt)
                    elif rec["fp8e4nv_supported"]:
                        got = F.weight_dequant_block(wt, sc, block_size = b, dtype = dt)
                    else:
                        parity[key] = {"skipped": "fp8e4nv unsupported here"}
                        continue
                    torch.cuda.synchronize()
                    parity[key] = _classify(torch, got, want, exact)
                except Exception as e:  # noqa: BLE001
                    parity[key] = {"error": f"{type(e).__name__}: {e}"[:800]}
    rec["parity"] = parity
    g3 = torch.Generator(device = "cpu").manual_seed(2)
    tw = (torch.randn(4096, 14336, generator = g3) * 0.3).to(torch.float8_e4m3fn).to(dev)
    ts = (torch.rand(32, 112, generator = g3) * 1e-2 + 1e-4).to(dev)
    timing = {}
    for dt in (torch.float16, torch.bfloat16):
        name = str(dt).split(".")[-1]
        arms = {"bytes": lambda dt = dt: F._e4m3fn_bytes_dequant(tw, ts, 128, 128, dt),
                "torch_loop": lambda dt = dt: F._torch_blockwise_dequant(tw, ts, [128, 128], dt)}
        if rec["fp8e4nv_supported"]:
            arms["fp8e4nv"] = lambda dt = dt: F.weight_dequant_block(tw, ts, block_size = 128, dtype = dt)
        timing[name] = _timed(torch, arms)
    rec["timing_4096x14336"] = timing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices = ("zoo", "fp8"), required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--quick", action = "store_true")
    args = ap.parse_args()
    rec: dict = {"stage": "import", "kind": args.kind}

    def dump():
        with open(args.out, "w", encoding = "utf-8") as fh:
            json.dump(rec, fh, indent = 2, default = str)

    try:
        import torch
        import triton

        props = torch.cuda.get_device_properties(0)
        rec.update(torch = torch.__version__, hip = torch.version.hip, triton = triton.__version__,
                   gpu = torch.cuda.get_device_name(0), arch = getattr(props, "gcnArchName", None))
        rec["stage"] = "run"
        dump()
        (run_zoo if args.kind == "zoo" else run_fp8)(torch, rec, args.quick)
        rec.update(stage = "done", ok = True)
    except Exception as e:  # noqa: BLE001
        rec.update(ok = False, error = f"{type(e).__name__}: {e}", tb = traceback.format_exc()[-3000:])
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
