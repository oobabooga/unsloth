#!/usr/bin/env python3
"""One arm of the MXFP4 HIP dequant probe, in its own process. Observes only, writes JSON to --out.

Imports the state's unsloth_zoo (PYTHONPATH set by the probe) and records:
  * selfcheck: mxfp4_kernel_available per (dtype, layout), the warnings it raised, _KERNEL_FAILED,
    and WHY: on the self-check's own data, mismatches vs an fp64 CPU truth grouped by e8m0 scale
    byte for the fused kernel, the device torch reference as this arm runs it, and both reference
    formulations forced (ldexp / 2^e from bits), so a failing check names the side that is wrong
  * route: which implementation mxfp4_dequantize calls (_kernel_dequantize = fused Triton kernel,
    mxfp4_dequantize_torch = torch path) on the audit's gpt-oss-20b expert stacks
  * parity: product output bit-equal to the torch reference, and expert 0 to the fp64 truth
  * per-call time (CUDA events per call, median, arms interleaved; the audit's method) of the
    product route, forced torch path, forced kernel, and torch.compile of the plain dequant
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import traceback
import warnings

# gpt-oss-20b expert stacks, the audit's b_mxfp4.py dequant_* cases: (E, rows, K).
SHAPES = {"gate_up": (32, 5760, 2880), "down": (32, 2880, 2880)}
FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True)
    ap.add_argument("--quick", action = "store_true")
    args = ap.parse_args()
    rec: dict = {"stage": "import", "env_UNSLOTH_MXFP4_HIP_EXACT_SCALE": os.environ.get("UNSLOTH_MXFP4_HIP_EXACT_SCALE")}

    def dump():
        with open(args.out, "w", encoding = "utf-8") as fh:
            json.dump(rec, fh, indent = 2, default = str)

    try:
        import torch
        import triton
        import unsloth_zoo.mxfp4_dequant as M

        props = torch.cuda.get_device_properties(0)
        rec.update(
            zoo_file = M.__file__, torch = torch.__version__, hip = torch.version.hip, triton = triton.__version__,
            gpu = torch.cuda.get_device_name(0), arch = getattr(props, "gcnArchName", None),
            has_triton = M._HAS_TRITON, ldexp_exact = M._LDEXP_EXACT,
            hip_exact_scale = getattr(M, "_HIP_EXACT_SCALE", None),
        )
        rec["stage"] = "selfcheck"
        dump()

        def bits(t):
            return t.contiguous().view(-1).view(torch.uint8 if t.element_size() == 1 else
                                                 {2: torch.int16, 4: torch.int32}[t.element_size()]).cpu()

        def truth(blocks, scales, dtype):
            b, s = blocks.cpu(), scales.cpu()
            lut = torch.tensor(FP4, dtype = torch.float64)
            *prefix, G, _ = b.shape
            v = torch.empty(*prefix, G, 32, dtype = torch.float64)
            v[..., 0::2] = lut[(b & 0x0F).long()]
            v[..., 1::2] = lut[(b >> 4).long()]
            v = v * torch.exp2(s.double() - 127).unsqueeze(-1)
            return v.reshape(*prefix, G * 32).to(dtype)

        def by_scale(got, want, scales):
            # Mismatching elements, keyed by the e8m0 scale byte of their block (got/want: (..., G*32)).
            g, w = bits(got), bits(want)
            diff = (g != w).view(*scales.shape, 32)
            n = diff.sum(-1).view(-1)
            sc = scales.cpu().reshape(-1)
            out = {}
            for k in torch.nonzero(n).view(-1).tolist():
                out[int(sc[k])] = out.get(int(sc[k]), 0) + int(n[k])
            return dict(total = int(diff.sum()), by_scale = dict(sorted(out.items())))

        # The self-check's own data (_kernel_verified), untransposed so the layouts compare directly.
        sc = {}
        for dt in (torch.bfloat16, torch.float16):
            E, N, G = 2, 131, 9
            dev = torch.device("cuda", 0)
            blocks = ((torch.arange(E * N * G * 16, device = dev) * 7) % 256).to(torch.uint8).reshape(E, N, G, 16)
            if dt == torch.float16:
                scales = 112 + (torch.arange(E * N * G, device = dev) * 5) % 24
            else:
                scales = (torch.arange(E * N * G, device = dev) * 37) % 255
            scales = scales.to(torch.uint8).reshape(E, N, G)
            edges = [0, 127, 254, 255] if dt != torch.float16 else [0, 127, 135, 255]
            scales.view(-1)[: len(edges)] = torch.tensor(edges, dtype = torch.uint8)
            want = truth(blocks, scales, dt)
            d: dict = {}
            with warnings.catch_warnings(record = True) as caught:
                warnings.simplefilter("always")
                M._KERNEL_VERIFIED.clear()
                d["available_transposed"] = bool(M.mxfp4_kernel_available(dtype = dt))
                d["verified_plain"] = bool(M._kernel_verified(dev, dt, False))
            d["warnings"] = [str(w.message)[:300] for w in caught]
            d["kernel_failed"] = bool(M._KERNEL_FAILED)
            try:
                d["kernel_vs_truth"] = by_scale(M._kernel_dequantize(blocks, scales, dt, False, None, None, None), want, scales)
            except Exception as e:  # noqa: BLE001
                d["kernel_error"] = f"{type(e).__name__}: {e}"[:1500]
            d["reference_vs_truth"] = by_scale(M.mxfp4_dequantize_torch(blocks, scales, dtype = dt), want, scales)
            saved = (getattr(M, "_HIP_EXACT_SCALE", None), M._LDEXP_EXACT)
            forced = {"ldexp": (False, M._LDEXP_EXACT), "ldexp_fp32": (False, False)}
            if hasattr(M, "_HIP_EXACT_SCALE"):
                forced["exact_bits"] = (True, M._LDEXP_EXACT)
            for name, (flag, ldexp_exact) in forced.items():
                if hasattr(M, "_HIP_EXACT_SCALE"):
                    M._HIP_EXACT_SCALE = flag
                M._LDEXP_EXACT = ldexp_exact
                d[f"reference_{name}_vs_truth"] = by_scale(M.mxfp4_dequantize_torch(blocks, scales, dtype = dt), want, scales)
            if hasattr(M, "_HIP_EXACT_SCALE"):
                M._HIP_EXACT_SCALE = saved[0]
            M._LDEXP_EXACT = saved[1]
            sc[str(dt).split(".")[-1]] = d
        # Re-run the bf16 checks now, so route counting below sees only the product call.
        M._KERNEL_VERIFIED.clear()
        dev = torch.device("cuda", 0)
        rec["bf16_verified"] = {t: bool(M._kernel_verified(dev, torch.bfloat16, t)) for t in (True, False)}
        rec.update(selfcheck = sc, stage = "route")
        dump()

        g = torch.Generator(device = "cuda").manual_seed(0)
        calls = {"kernel": 0, "torch": 0}
        real_k, real_t = M._kernel_dequantize, M.mxfp4_dequantize_torch

        def counting(on):
            if on:
                M._kernel_dequantize = lambda *a, **k: calls.__setitem__("kernel", calls["kernel"] + 1) or real_k(*a, **k)
                M.mxfp4_dequantize_torch = lambda *a, **k: calls.__setitem__("torch", calls["torch"] + 1) or real_t(*a, **k)
            else:
                M._kernel_dequantize, M.mxfp4_dequantize_torch = real_k, real_t

        dt = torch.bfloat16
        route, parity, kt = {}, {}, {}
        for name, (E, R, K) in SHAPES.items():
            blocks = torch.randint(0, 256, (E, R, K // 32, 16), dtype = torch.uint8, device = "cuda", generator = g)
            scales = torch.randint(118, 128, (E, R, K // 32), dtype = torch.uint8, device = "cuda", generator = g)
            for transpose in (True, False):
                key = f"{name}/{'T' if transpose else 'plain'}"
                calls.update(kernel = 0, torch = 0)
                counting(True)
                o = M.mxfp4_dequantize(blocks, scales, dtype = dt, transpose = transpose)
                counting(False)
                torch.cuda.synchronize()
                route[key] = dict(calls)
                ref = real_t(blocks, scales, dtype = dt, transpose = transpose)
                t0 = truth(blocks[:1], scales[:1], dt)[0]
                o0 = o[0].transpose(0, 1) if transpose else o[0]
                parity[key] = dict(equal_reference = bool(torch.equal(bits(o), bits(ref))),
                                   expert0_equal_fp64_truth = bool(torch.equal(bits(o0), bits(t0))))
                del o, ref
                torch.cuda.empty_cache()
            rec.update(route = route, parity = parity)
            dump()

            def measure(fn, n = 5 if args.quick else 10):
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

            lut = torch.tensor(FP4, dtype = torch.float32, device = "cuda")

            def plain(b, s):
                lo = lut[(b & 0x0F).long()]
                hi = lut[(b >> 4).long()]
                v = torch.stack((lo, hi), dim = -1).flatten(-2)
                v = torch.ldexp(v, (s.to(torch.int32) - 127).unsqueeze(-1).to(torch.float32))
                return v.flatten(-2).to(dt)

            for transpose in (True, False):
                key = f"{name}/{'T' if transpose else 'plain'}"
                comp = torch.compile((lambda b, s: plain(b, s).transpose(-1, -2).contiguous()) if transpose else plain,
                                     dynamic = False)
                arms = {
                    "product": lambda: M.mxfp4_dequantize(blocks, scales, dtype = dt, transpose = transpose),
                    "torch": lambda: real_t(blocks, scales, dtype = dt, transpose = transpose),
                    "compiled_plain": lambda: comp(blocks, scales),
                }
                try:
                    real_k(blocks[:1], scales[:1], dt, transpose, None, None, None)
                    arms["kernel"] = lambda: real_k(blocks, scales, dt, transpose, None, None, None)
                except Exception as e:  # noqa: BLE001
                    rec.setdefault("kernel_errors", {})[key] = f"{type(e).__name__}: {e}"[:500]
                res = {a: [] for a in arms}
                for r in range(2 if args.quick else 3):
                    order = list(arms) if r % 2 == 0 else list(arms)[::-1]
                    for a in order:
                        res[a].append(measure(arms[a]))
                kt[key] = {a: dict(median_ms = statistics.median(v), runs = v) for a, v in res.items()}
                rec["kernel_ms"] = kt
                dump()
                torch.cuda.empty_cache()
            blocks = scales = None
            torch.cuda.empty_cache()
        rec.update(stage = "done", ok = True)
    except Exception as e:  # noqa: BLE001
        rec.update(ok = False, error = f"{type(e).__name__}: {e}", tb = traceback.format_exc()[-3000:])
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
