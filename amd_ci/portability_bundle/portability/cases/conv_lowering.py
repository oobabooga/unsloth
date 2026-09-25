"""Item 3: Inductor's own Triton conv template (torch/_inductor/kernel/conv.py, via max-autotune) vs cuDNN / MIOpen,
on the H3 encoder and Qwen-Image-2.1 VAE Conv3d shapes, fp16 and bf16, channels-last like Studio.

  eager            F.conv3d, cudnn.benchmark on (cuDNN on NVIDIA, MIOpen on ROCm)
  compiled         torch.compile default: Inductor keeps the conv as an extern aten call
  ma_aten_triton   max-autotune-no-cudagraphs, conv backends ATEN,TRITON: Inductor picks the faster
  ma_triton        max-autotune-no-cudagraphs, conv backends TRITON only: the template itself
``engaged`` = the generated code launches the Triton conv template instead of ``extern_kernels.convolution``.
Error is against a float32 conv (TF32 off) of the same rounded operands.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from portlib import FAILS, FALLBACK, RUNS, SKIPPED, compiled_code  # noqa: E402


def _uses_template(code: str) -> tuple[bool, str]:
    ext = code.count("extern_kernels.convolution(")
    tem = len(set(re.findall(r"\b(triton_tem_\w+)", code)))
    mm = code.count("extern_kernels.mm") + code.count("extern_kernels.addmm") + code.count("extern_kernels.bmm")
    return (ext == 0 and (tem > 0 or mm > 0)), f"extern_conv={ext} triton_template={tem} extern_mm={mm}"


def run(ctx):
    import torch

    with torch.no_grad():
        return _run(ctx)


def _run(ctx):
    import torch
    import torch._inductor.config as icfg
    import torch.nn.functional as F

    from shapes import conv_shapes

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.cache_size_limit = 256
    have_backends = hasattr(icfg, "max_autotune_conv_backends")
    engine = "MIOpen" if ctx.is_rocm else "cuDNN"
    dts = [("fp16", torch.float16), ("bf16", torch.bfloat16)]
    for si, (tag, cin, cout, T, H, W, k, st) in enumerate(conv_shapes(ctx.quick)):
        shp = f"{tag} {cin}->{cout} {T}x{H}x{W} k{k[0]}"
        for dn, dt in dts:
            try:
                g = torch.Generator(device = "cuda").manual_seed(100 + si)
                x = torch.randn(1, cin, T, H, W, device = "cuda", generator = g).to(dt)
                x = x.contiguous(memory_format = torch.channels_last_3d)
                w = (torch.randn(cout, cin, *k, device = "cuda", generator = g) / (cin * k[0] * k[1] * k[2]) ** 0.5)
                w = w.to(dt).contiguous(memory_format = torch.channels_last_3d)
                b = (torch.randn(cout, device = "cuda", generator = g) * 0.1).to(dt)
                ref = F.conv3d(x.float(), w.float(), b.float(), st)
            except torch.cuda.OutOfMemoryError as exc:
                ctx.row("setup", shape = shp, dtype = dn, status = SKIPPED, note = f"OOM: {exc}"[:200])
                torch.cuda.empty_cache()
                continue

            def conv(a, ww, bb):
                return F.conv3d(a, ww, bb, st)

            eager_ms = comp_ms = None
            try:
                o = conv(x, w, b)
                torch.cuda.synchronize()
                eager_ms = ctx.bench(lambda: conv(x, w, b))
                ctx.row("eager", shape = shp, dtype = dn, err = ctx.err(o, ref), ms = eager_ms, eager_ms = eager_ms,
                        note = engine)
            except Exception as exc:  # noqa: BLE001
                ctx.fail("eager", exc, shape = shp, dtype = dn)
            variants = [("compiled", {}, None)]
            if have_backends:
                variants += [("ma_aten_triton", {"max_autotune_conv_backends": "ATEN,TRITON"},
                              "max-autotune-no-cudagraphs"),
                             ("ma_triton", {"max_autotune_conv_backends": "TRITON"}, "max-autotune-no-cudagraphs")]
            for vname, patch, mode in variants:
                torch._dynamo.reset()
                try:
                    with icfg.patch(patch):
                        cf = torch.compile(conv, dynamic = False, **({"mode": mode} if mode else {}))
                        o, code = compiled_code(cf, x, w, b)
                        torch.cuda.synchronize()
                        with open(os.path.join(ctx.log_dir, f"{vname}_{si}_{dn}.py"), "w") as f:
                            f.write(code)
                        ms = ctx.bench(lambda: cf(x, w, b))
                    if vname == "compiled":
                        comp_ms = ms
                    tmpl, why = _uses_template(code)
                    if vname == "ma_triton" and not tmpl:
                        status = FALLBACK
                    else:
                        status = RUNS
                    ctx.row(vname, shape = shp, dtype = dn, status = status, engaged = tmpl, err = ctx.err(o, ref),
                            ms = ms, eager_ms = eager_ms, compiled_ms = comp_ms, note = why)
                except Exception as exc:  # noqa: BLE001
                    ctx.fail(vname, exc, shape = shp, dtype = dn, eager_ms = eager_ms, compiled_ms = comp_ms)
            del x, w, b, ref
            torch.cuda.empty_cache()
    return ctx.rows
