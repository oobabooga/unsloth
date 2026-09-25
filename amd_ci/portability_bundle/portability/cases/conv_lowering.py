"""Item 3: Inductor's own Triton conv template (torch/_inductor/kernel/conv.py, via max-autotune) vs cuDNN / MIOpen,
on the H3 encoder Conv3d and Qwen-Image-2.1 VAE decode Conv2d shapes, fp16 and bf16, channels-last like Studio.

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
    slow_bf16 = (not ctx.is_rocm) and ctx.cap < (8, 0)
    ws_root = os.environ.get("WORKSPACE")
    for si, (tag, xs, wsh, st) in enumerate(conv_shapes(ctx.quick, ws_root, ctx.bundle)):
        nd = len(xs) - 2
        cl = torch.channels_last_3d if nd == 3 else torch.channels_last
        convf = F.conv3d if nd == 3 else F.conv2d
        shp = f"{tag} {'x'.join(map(str, xs[1:]))} k{'x'.join(map(str, wsh[2:]))}"
        for dn, dt in dts:
            if dt is torch.bfloat16 and slow_bf16 and si > 0:
                ctx.row("eager", shape = shp, dtype = dn, status = SKIPPED,
                        note = "bf16 conv has no tensor-core path below sm80 (~14x fp16); measured on the first shape only")
                continue
            try:
                g = torch.Generator(device = "cuda").manual_seed(100 + si)
                x = torch.randn(*xs, device = "cuda", generator = g).to(dt).contiguous(memory_format = cl)
                fan = 1
                for d in wsh[1:]:
                    fan *= d
                w = (torch.randn(*wsh, device = "cuda", generator = g) / fan ** 0.5).to(dt).contiguous(memory_format = cl)
                b = (torch.randn(wsh[0], device = "cuda", generator = g) * 0.1).to(dt)
                ref = convf(x.float(), w.float(), b.float(), st)
            except torch.cuda.OutOfMemoryError as exc:
                ctx.row("setup", shape = shp, dtype = dn, status = SKIPPED, note = f"OOM: {exc}"[:200])
                torch.cuda.empty_cache()
                continue

            def conv(a, ww, bb, _f = convf):
                return _f(a, ww, bb, st)

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
            if eager_ms is not None and eager_ms > 250:
                ctx.row("compiled", shape = shp, dtype = dn, status = SKIPPED,
                        note = f"eager {eager_ms:.0f} ms per call: autotuning this shape would exceed the case budget")
                del x, w, b, ref
                torch.cuda.empty_cache()
                continue
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
