#!/usr/bin/env python3
"""One arm of the FP8 HIP block-dequant probe, in its own process. Observes only, writes JSON to --out.

Imports the state's unsloth (PYTHONPATH set by the probe) and records, for weight_dequant on a
128x128 block-scaled e4m3fn weight (the Triton audit's b_fp8.py dequant_block_* shapes, plus a
ragged and a rectangular-block one):
  * route: which implementation _blockwise_weight_dequant_any_shape calls
    (_e4m3fn_bytes_dequant = byte decoder, weight_dequant_block = fp8e4nv Triton kernel,
    _torch_blockwise_dequant = chunked torch loop)
  * bit equality of the product output vs the torch loop and vs the fp8e4nv Triton kernel
    (where it compiles and the tiling is even), NaN positions included
  * per-call time (CUDA events per call, median, arms interleaved; the audit's method) of the
    product route, forced byte decoder, forced fp8e4nv Triton kernel, torch loop, and
    torch.compile of the plain dequant (one pass over the weight, the audit's reference)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import traceback

# name: (rows, cols, block)
SHAPES = {
    "q": (4096, 4096, (128, 128)),
    "up": (14336, 4096, (128, 128)),
    "down": (4096, 14336, (128, 128)),
    "ragged": (14400, 4160, (128, 128)),
    "rect_block": (4096, 4096, (64, 128)),
}
TIMED = ("q", "up", "down", "ragged")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--quick", action = "store_true")
    args = ap.parse_args()
    rec: dict = {"stage": "import", "env_UNSLOTH_FP8_HIP_BYTES_DEQUANT": os.environ.get("UNSLOTH_FP8_HIP_BYTES_DEQUANT")}

    def dump():
        with open(args.out, "w", encoding = "utf-8") as fh:
            json.dump(rec, fh, indent = 2, default = str)

    try:
        import torch
        import triton
        from unsloth.kernels import fp8 as F

        dev = torch.device("cuda", 0)
        props = torch.cuda.get_device_properties(0)
        probe_w = torch.zeros(128, 128, device = dev, dtype = torch.float8_e4m3fn)
        rec.update(
            unsloth_file = F.__file__,
            torch = torch.__version__, hip = torch.version.hip, triton = triton.__version__,
            gpu = torch.cuda.get_device_name(0), arch = getattr(props, "gcnArchName", None),
            hip_bytes_flag = getattr(F, "_HIP_FP8_BYTES_DEQUANT", None),
            lacks_ocp_fp8 = (F._rocm_lacks_ocp_fp8(dev) if hasattr(F, "_rocm_lacks_ocp_fp8") else None),
            fp8e4nv_unsupported = bool(F._fp8_kernel_unsupported(probe_w)),
            has_bytes_kernel = hasattr(F, "_e4m3fn_bytes_dequant"),
        )
        rec["stage"] = "route"
        dump()

        names = {"_e4m3fn_bytes_dequant": "bytes", "weight_dequant_block": "triton_fp8",
                 "_torch_blockwise_dequant": "torch"}
        real = {n: getattr(F, n) for n in names}
        counts = {v: 0 for v in names.values()}

        def counting(on):
            for n, f in real.items():
                if on:
                    setattr(F, n, (lambda *a, _f = f, _k = names[n], **k: counts.__setitem__(_k, counts[_k] + 1) or _f(*a, **k)))
                else:
                    setattr(F, n, f)

        def make(m, n, block, seed = 0):
            g = torch.Generator(device = "cpu").manual_seed(seed)
            raw = torch.randint(0, 256, (m, n), generator = g, dtype = torch.uint8)
            raw.view(-1)[:256] = torch.arange(256, dtype = torch.uint8)  # every code: subnormals, -0, NaN
            w = raw.view(torch.float8_e4m3fn).to(dev)
            s = (torch.rand(-(-m // block[0]), -(-n // block[1]), generator = g) * 1e-2 + 1e-4).to(dev)
            return w, s

        def same_bits(a, b):
            if a.shape != b.shape or a.dtype != b.dtype:
                return False
            na, nb = torch.isnan(a), torch.isnan(b)
            if not torch.equal(na, nb):
                return False
            ai = a.masked_fill(na, 0).contiguous().view(-1).view(torch.uint8)
            bi = b.masked_fill(nb, 0).contiguous().view(-1).view(torch.uint8)
            return bool(torch.equal(ai, bi))

        route, parity = {}, {}
        for name, (m, n, block) in SHAPES.items():
            w, s = make(m, n, block)
            for dt in (torch.bfloat16, torch.float16):
                key = f"{name}/{str(dt).split('.')[-1]}"
                for k in counts:
                    counts[k] = 0
                counting(True)
                o = F._blockwise_weight_dequant_any_shape(w, s, list(block), dt)
                o2 = F._blockwise_weight_dequant_any_shape(w, s, list(block), dt)
                counting(False)
                torch.cuda.synchronize()
                route[key] = dict(counts)
                ref = real["_torch_blockwise_dequant"](w, s, list(block), dt)
                p = dict(equal_torch_loop = same_bits(o, ref), repeat_equal = same_bits(o, o2))
                even = m % block[0] == 0 and n % block[1] == 0 and block[0] == block[1]
                if even and not F._fp8_kernel_unsupported(w):
                    # NaN-free copy (0x7F / 0xFF -> 0x7E / 0xFE): the emulated fp8e4nv cast's NaN handling is
                    # not what is under test; the torch loop above is the reference for every code.
                    raw = w.view(torch.uint8)
                    wf = torch.where((raw & 127) == 127, raw - 1, raw).view(torch.float8_e4m3fn)
                    counting(True)
                    of = F._blockwise_weight_dequant_any_shape(wf, s, list(block), dt)
                    counting(False)
                    p["equal_triton_fp8"] = same_bits(of, real["weight_dequant_block"](wf, s, block[0], dt))
                    del of, wf
                if rec["has_bytes_kernel"]:
                    p["bytes_equal_torch_loop"] = same_bits(
                        real["_e4m3fn_bytes_dequant"](w, s, block[0], block[1], dt), ref)
                parity[key] = p
                del o, o2, ref
            del w, s
            torch.cuda.empty_cache()
        rec.update(route = route, parity = parity, stage = "kernel_time")
        dump()

        def measure(fn, n = 50):
            for _ in range(5):
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

        dt = torch.bfloat16
        kt = {}
        for name in TIMED:
            m, n, block = SHAPES[name]
            w, s = make(m, n, block)
            plain = lambda w = w, s = s, m = m, n = n: (
                w.view(m // 128, 128, n // 128, 128).float() * s[:, None, :, None]).to(dt).view(m, n)
            arms = {
                "product": lambda: F.weight_dequant(w, s, dt),
                "torch_loop": lambda: real["_torch_blockwise_dequant"](w, s, list(block), dt),
            }
            if rec["has_bytes_kernel"]:
                arms["bytes"] = lambda: real["_e4m3fn_bytes_dequant"](w, s, block[0], block[1], dt)
            if m % 128 == 0 and n % 128 == 0 and not F._fp8_kernel_unsupported(w):
                arms["triton_fp8"] = lambda: real["weight_dequant_block"](w, s, 128, dt)
                arms["compiled_one_pass"] = torch.compile(plain, dynamic = False)
            res = {a: [] for a in arms}
            for r in range(2 if args.quick else 3):
                order = list(arms) if r % 2 == 0 else list(arms)[::-1]
                for a in order:
                    res[a].append(measure(arms[a]))
            kt[name] = {a: dict(median_ms = statistics.median(v), runs = v) for a, v in res.items()}
            rec["kernel_ms"] = kt
            dump()
            w = s = arms = None  # free before the next shape
            torch.cuda.empty_cache()
        rec.update(kernel_ms = kt, stage = "done", ok = True)
    except Exception as e:  # noqa: BLE001
        rec.update(ok = False, error = f"{type(e).__name__}: {e}", tb = traceback.format_exc()[-3000:])
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
