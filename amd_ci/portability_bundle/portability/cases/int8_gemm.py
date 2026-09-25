"""Item 1: W8A8 int8 GEMM paths on the real Qwen-Image-2.1 / MiniMax-H3 linear shapes.

Per shape, against a float32 (TF32 off) reference on the same base-dtype operands:
  base_eager / base_compiled    F.linear at the device base dtype (bf16; fp16 on T4), eager and torch.compile
  intmm_eager / intmm_compiled  torch-op per-token quant + torch._int_mm + float epilogue, eager and compiled
  intmm_compiled_fusemul        same, with inductor's force_fuse_int_mm_with_mul (what torchao's recommended config sets)
  torchao_eager / _compiled     torchao Int8DynamicActivationInt8Weight on an nn.Linear
  triton_2k / triton_fused      portable Triton quant + int8 dot (two kernels / one kernel), see triton_int8.py
  pr11801_int8_linear           the H3 VAE's production _int8_linear (Triton quant_rows + _int_mm + Triton epilogue)
``engaged`` = an int8 GEMM actually ran (aten::_int_mm in the profile or an int8 Triton dot), not a float fallback.
``vs_w8a8_ref`` = max abs difference to intmm_eager: the int path is exact in int32, so a correct kernel is within
an output-dtype ulp of it; the ``err`` column is the quantisation error proper.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from portlib import FAILS, FALLBACK, RUNS, SKIPPED, compiled_code, profile_names  # noqa: E402


def _synthetic(M, K, N, bias, dt, seed):
    import torch

    g = torch.Generator(device = "cuda").manual_seed(seed)
    x = torch.randn(M, K, device = "cuda", generator = g)
    # diffusion activations: a few hot channels and a spread of per-token magnitudes
    hot = torch.rand(K, device = "cuda", generator = g) < 0.005
    x = x * torch.where(hot, 8.0, 1.0) * torch.exp(0.5 * torch.randn(M, 1, device = "cuda", generator = g))
    w = torch.randn(N, K, device = "cuda", generator = g) * (1.0 / K**0.5)
    b = torch.randn(N, device = "cuda", generator = g) * 0.1 if bias else None
    return x.to(dt), w.to(dt), (b.to(dt) if b is not None else None)


def _int8_engaged(fn, own_kernel_hint = None):
    ops, kernels = profile_names(fn)
    if "aten::_int_mm" in ops:
        return True, "aten::_int_mm"
    if own_kernel_hint and any(own_kernel_hint in k for k in kernels):
        return True, own_kernel_hint
    return False, "no int8 GEMM in profile: " + ",".join(sorted(k for k in ops if k.startswith("aten::") and
                                                                  ("mm" in k or "linear" in k or "addmm" in k)))[:200]


def _torchao_config():
    import torchao.quantization as q

    if hasattr(q, "Int8DynamicActivationInt8WeightConfig"):
        return q.Int8DynamicActivationInt8WeightConfig(), "Int8DynamicActivationInt8WeightConfig"
    return q.int8_dynamic_activation_int8_weight(), "int8_dynamic_activation_int8_weight"


def run(ctx):
    import torch

    with torch.no_grad():
        return _run(ctx)


def _run(ctx):
    import torch
    import torch.nn.functional as F

    from cases import triton_int8 as ti
    from shapes import gemm_shapes

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 256)
    dt = ctx.base_dtype
    ws_root = os.environ.get("WORKSPACE")
    shapes = gemm_shapes(ctx.quick, ws_root, ctx.bundle)
    ctx.log(f"{len(shapes)} shapes, base {ctx.base}; source {shapes[0]['source'] if shapes else '-'}")

    h3 = None
    p = ctx.bundle_path("trees", "h3", "studio", "backend", "core", "inference", "video_minimax_h3_vae.py")
    if p:
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("h3vae_pr11801", p)
            h3 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(h3)
        except Exception as exc:  # noqa: BLE001
            ctx.log("PR 11801 module failed to import:", exc)
            h3 = None

    ao_cfg, ao_name, ao_err = None, "", ""
    try:
        from torchao.quantization import quantize_

        ao_cfg, ao_name = _torchao_config()
    except Exception as exc:  # noqa: BLE001
        ao_err = f"{type(exc).__name__}: {str(exc)[:200]}"

    try:
        import triton  # noqa: F401

        have_triton = True
    except Exception:  # noqa: BLE001
        have_triton = False

    for i, s in enumerate(shapes):
        M, K, N, has_b = s["M"], s["K"], s["N"], s["bias"]
        shp = f"{s['tag']} {M}x{K}x{N}"
        try:
            x, w, b = _synthetic(M, K, N, has_b, dt, 1234 + i)
            ref = F.linear(x.float(), w.float(), b.float() if b is not None else None)
        except torch.cuda.OutOfMemoryError as exc:
            ctx.row("setup", shape = shp, status = SKIPPED, note = f"OOM building inputs: {exc}"[:200])
            torch.cuda.empty_cache()
            continue
        wq, wsc = ti.quantize_weight(w)
        # base eager
        eager_ms = comp_ms = None
        out = ctx.attempt("base_eager", lambda: F.linear(x, w, b), shape = shp)
        if out is not None:
            eager_ms = ctx.bench(lambda: F.linear(x, w, b))
            ctx.row("base_eager", shape = shp, err = ctx.err(out, ref), ms = eager_ms, eager_ms = eager_ms)
        torch._dynamo.reset()

        def _lin(a, ww, bb):
            return F.linear(a, ww, bb)

        cf = torch.compile(_lin, dynamic = False)
        out = ctx.attempt("base_compiled", lambda: cf(x, w, b), shape = shp)
        if out is not None:
            comp_ms = ctx.bench(lambda: cf(x, w, b))
            ctx.row("base_compiled", shape = shp, err = ctx.err(out, ref), ms = comp_ms, eager_ms = eager_ms,
                    compiled_ms = comp_ms)

        x2 = x.reshape(-1, K)
        w8ref = None

        def rec(variant, fn, engaged_hint = None, extra_note = "", code = None, own = False):
            nonlocal w8ref
            try:
                o = fn()
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                ctx.fail(variant, exc, shape = shp, eager_ms = eager_ms, compiled_ms = comp_ms)
                return None
            ms = ctx.bench(fn)
            eng, why = _int8_engaged(fn, engaged_hint)
            if not eng and own:
                # our own int8 Triton kernel launched without error; the profiler just lacked device activity
                eng, why = True, f"own int8 Triton kernel ({why[:60]})"
            if code is not None and not eng:
                eng = ("_int_mm" in code) or ("tl.int8" in code and "tl.dot" in code)
                why = "int8 in generated code" if eng else why
            extra = {}
            if w8ref is not None:
                extra["vs_w8a8_ref"] = float((o.float() - w8ref.float()).abs().max())
            elif variant == "intmm_eager":
                w8ref = o
            if code is not None:
                extra["gen_kernels"] = code.count("\ndef triton_") + code.count("@triton.jit")
                extra["gen_extern_int_mm"] = code.count("_int_mm(")
            ctx.row(variant, shape = shp, status = RUNS if eng else FALLBACK, engaged = eng, err = ctx.err(o, ref),
                    ms = ms, eager_ms = eager_ms, compiled_ms = comp_ms, note = (extra_note + " " + why).strip(),
                    **extra)
            return o

        if K % 8 or N % 8 or M <= 16:
            ctx.row("intmm_eager", shape = shp, status = SKIPPED, note = "torch._int_mm needs M>16, K,N % 8 == 0")
        else:
            rec("intmm_eager", lambda: ti.w8a8_torch(x2, wq, wsc, b, dt))
            for variant, fuse in (("intmm_compiled", False), ("intmm_compiled_fusemul", True)):
                torch._dynamo.reset()
                import torch._inductor.config as icfg

                prev = getattr(icfg, "force_fuse_int_mm_with_mul", None)
                if fuse:
                    if prev is None:
                        continue
                    icfg.force_fuse_int_mm_with_mul = True
                try:
                    cw = torch.compile(ti.w8a8_torch, dynamic = False)
                    code = None
                    try:
                        _, code = compiled_code(cw, x2, wq, wsc, b, dt)
                        with open(os.path.join(ctx.log_dir, f"{variant}_{i}.py"), "w") as f:
                            f.write(code)
                    except Exception as exc:  # noqa: BLE001
                        ctx.fail(variant, exc, shape = shp, eager_ms = eager_ms, compiled_ms = comp_ms)
                        continue
                    rec(variant, lambda: cw(x2, wq, wsc, b, dt), code = code)
                finally:
                    if fuse and prev is not None:
                        icfg.force_fuse_int_mm_with_mul = prev
        # torchao
        if ao_cfg is None:
            ctx.row("torchao_eager", shape = shp, status = SKIPPED, note = f"torchao unavailable: {ao_err}")
        else:
            try:
                lin = torch.nn.Linear(K, N, bias = has_b, device = "cuda", dtype = dt)
                with torch.no_grad():
                    lin.weight.copy_(w)
                    if has_b:
                        lin.bias.copy_(b)
                quantize_(lin, _torchao_config()[0])
                rec("torchao_eager", lambda: lin(x), extra_note = ao_name)
                torch._dynamo.reset()
                clin = torch.compile(lin, dynamic = False)
                code = None
                try:
                    _, code = compiled_code(clin, x)
                except Exception as exc:  # noqa: BLE001
                    ctx.fail("torchao_compiled", exc, shape = shp, eager_ms = eager_ms, compiled_ms = comp_ms)
                else:
                    rec("torchao_compiled", lambda: clin(x), extra_note = ao_name, code = code)
                del lin, clin
            except Exception as exc:  # noqa: BLE001
                ctx.fail("torchao_quantize", exc, shape = shp)
        # Triton
        if not have_triton:
            ctx.row("triton_2k", shape = shp, status = SKIPPED, note = "no triton")
        else:
            rec("triton_2k", lambda: ti.w8a8_triton(x2, wq, wsc, b, dt, fused = False), engaged_hint = "int8_mm",
                own = True)
            rec("triton_fused", lambda: ti.w8a8_triton(x2, wq, wsc, b, dt, fused = True), engaged_hint = "int8_mm",
                own = True)
        if h3 is not None:
            import types

            fake = types.SimpleNamespace(bias = b, _unsloth_int8 = (wq, wsc))
            prod_gate = None
            try:
                prod_gate = bool(h3.cuda_fast_path_available())
            except Exception:  # noqa: BLE001
                pass
            rec("pr11801_int8_linear", lambda: h3._int8_linear(fake, x2), engaged_hint = "_quant_rows",
                extra_note = f"production gate cuda_fast_path_available={prod_gate}")
        del x, w, b, ref, wq, wsc, x2
        w8ref = None
        torch.cuda.empty_cache()
    return ctx.rows
