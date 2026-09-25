"""Item 4: Studio's speed-tier compile settings on a DiT block of real Qwen-Image-2.1 width, bf16 AND fp16 everywhere.

Settings mirror studio/backend/core/inference/diffusion_speed.py on main (_compile_repeated_blocks) and
diffusion_cuda_graph.py (denoiser-level capture), checked against the bundled copy of that source at run time:
  eager    no compile
  default  regional compile of every repeated block: fullgraph=True, dynamic=True
  max      regional compile, mode="max-autotune-no-cudagraphs", dynamic=None (automatic), recompile_limit >= 64,
           inductor emulate_precision_casts=True
  +graph   CUDA graph of the whole denoiser forward over static inputs, after warmup on a side stream
Per (dtype, tier): does it compile / capture at all, the error vs a float32 eager copy of the same weights, warm step
time, first-call compile seconds, and the recompile cost of a new prompt length (+37 text tokens).
Studio's own gates are reported beside the result: regional compile only for a bf16 denoiser (fp16 on T4 -> eager),
CUDA graphs only on backend "cuda" (never ROCm).
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from portlib import FAILS, FALLBACK, RUNS, SKIPPED, WRONG  # noqa: E402

MAIN_MARKERS = ('kwargs["mode"] = "max-autotune-no-cudagraphs"', '"dynamic": None if max_autotune else True',
                "emulate_precision_casts = True")


def _model(D, heads, n_blocks, mlp_ratio = 3):
    import torch
    import torch.nn.functional as F
    from torch import nn

    class RMS(nn.Module):
        def __init__(self, d, affine = True):
            super().__init__()
            self.w = nn.Parameter(torch.ones(d)) if affine else None
            self.d = d

        def forward(self, x):
            return F.rms_norm(x.float(), (self.d,), self.w.float() if self.w is not None else None, 1e-6).to(x.dtype)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            hd = D // heads
            self.heads = heads
            self.norm1, self.norm2 = RMS(D, False), RMS(D, False)
            self.to_q, self.to_k, self.to_v = (nn.Linear(D, D, bias = False) for _ in range(3))
            self.norm_q, self.norm_k = RMS(hd), RMS(hd)
            self.to_out = nn.Linear(D, D, bias = False)
            self.proj = nn.Linear(D, mlp_ratio * D, bias = False)
            self.gate_layer = nn.Linear(D, mlp_ratio * D, bias = False)
            self.out = nn.Linear(mlp_ratio * D, D, bias = False)

        def forward(self, x, mod, cos, sin):
            s1, c1, g1, s2, c2, g2 = mod.unbind(1)
            B, L, _ = x.shape
            h = self.norm1(x) * (1 + c1[:, None]) + s1[:, None]
            q = self.norm_q(self.to_q(h).view(B, L, self.heads, -1)).transpose(1, 2)
            k = self.norm_k(self.to_k(h).view(B, L, self.heads, -1)).transpose(1, 2)
            v = self.to_v(h).view(B, L, self.heads, -1).transpose(1, 2)

            def rope(t):
                a, b = t.chunk(2, dim = -1)
                return t * cos + torch.cat([-b, a], dim = -1) * sin

            a = F.scaled_dot_product_attention(rope(q), rope(k), v)
            x = x + g1[:, None] * self.to_out(a.transpose(1, 2).reshape(B, L, D))
            h = self.norm2(x) * (1 + c2[:, None]) + s2[:, None]
            return x + g2[:, None] * self.out(F.silu(self.gate_layer(h)) * self.proj(h))

    class DiT(nn.Module):
        _repeated_blocks = ["Block"]

        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(Block() for _ in range(n_blocks))

        def forward(self, x, mod, cos, sin):
            for b in self.blocks:
                x = b(x, mod, cos, sin)
            return x

    return DiT()


def _inputs(L, D, heads, dt, seed):
    import torch

    g = torch.Generator(device = "cuda").manual_seed(seed)
    hd = D // heads
    x = torch.randn(1, L, D, device = "cuda", generator = g).to(dt)
    mod = (torch.randn(1, 6, D, device = "cuda", generator = g) * 0.1).to(dt)
    pos = torch.arange(L, device = "cuda").float()[:, None]
    inv = 1.0 / (10000 ** (torch.arange(0, hd // 2, device = "cuda").float() / (hd // 2)))
    ang = torch.cat([pos * inv, pos * inv], dim = -1)
    return x, mod, ang.cos().to(dt)[None, None], ang.sin().to(dt)[None, None]


def run(ctx):
    import torch

    with torch.no_grad():
        return _run(ctx)


def _run(ctx):
    import copy

    import torch
    import torch._dynamo
    import torch._inductor.config as icfg

    src = ctx.bundle_path("trees", "main", "studio", "backend", "core", "inference", "diffusion_speed.py")
    if src:
        text = open(src, encoding = "utf-8").read()
        missing = [m for m in MAIN_MARKERS if m not in text]
        ctx.row("tier_settings_vs_main", status = RUNS if not missing else WRONG, engaged = not missing,
                note = "mirrors main" if not missing else f"main changed, re-check: missing {missing}")
    D, heads = 4096, 32
    L = 4352 if not ctx.quick else 1280
    n_blocks = 2
    ref_model = _model(D, heads, n_blocks).to("cuda", torch.float32).eval()
    gate_compile = {"bf16": True, "fp16": False}
    gate_graph = not ctx.is_rocm
    ctx.row("studio_gates", status = RUNS, dtype = ctx.base,
            note = f"Studio computes in {ctx.base} here; regional compile engages for "
                   f"{'bf16 only' if True else ''} -> {'ON' if gate_compile[ctx.base] else 'OFF (fp16 denoiser)'}; "
                   f"CUDA graphs {'ON' if gate_graph else 'OFF (backend rocm)'}")
    for dn, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        model = copy.deepcopy(ref_model).to(dt)
        x, mod, cos, sin = _inputs(L, D, heads, dt, 7)
        ref = ref_model(x.float(), mod.float(), cos.float(), sin.float())
        x2, mod2, cos2, sin2 = _inputs(L + 37, D, heads, dt, 8)
        eager_ms = None
        try:
            o = model(x, mod, cos, sin)
            torch.cuda.synchronize()
            eager_ms = ctx.bench(lambda: model(x, mod, cos, sin), iters = max(5, ctx.iters // 2))
            e = ctx.err(o, ref)
            ctx.row("eager", shape = f"{n_blocks}x block D{D} L{L}", dtype = dn,
                    status = WRONG if e.get("nonfinite") else RUNS, err = e, ms = eager_ms, eager_ms = eager_ms)
        except Exception as exc:  # noqa: BLE001
            ctx.fail("eager", exc, dtype = dn)
            del model
            torch.cuda.empty_cache()
            continue
        default_ms = None
        for tier in ("default", "max"):
            torch._dynamo.reset()
            m = copy.deepcopy(model)
            patch = {}
            kw = {"fullgraph": True, "dynamic": True}
            if tier == "max":
                kw = {"fullgraph": True, "dynamic": None, "mode": "max-autotune-no-cudagraphs"}
                patch = {"emulate_precision_casts": True} if hasattr(icfg, "emulate_precision_casts") else {}
            for attr in ("recompile_limit", "cache_size_limit"):
                if hasattr(torch._dynamo.config, attr):
                    setattr(torch._dynamo.config, attr, max(getattr(torch._dynamo.config, attr) or 0, 64))
            shp = f"{n_blocks}x block D{D} L{L}"
            try:
                with icfg.patch(patch):
                    for b in m.blocks:
                        b.compile(**kw)
                    t0 = time.time()
                    o = m(x, mod, cos, sin)
                    torch.cuda.synchronize()
                    compile_s = time.time() - t0
                    ms = ctx.bench(lambda: m(x, mod, cos, sin), iters = max(5, ctx.iters // 2))
                    t0 = time.time()
                    m(x2, mod2, cos2, sin2)
                    torch.cuda.synchronize()
                    new_prompt_s = time.time() - t0
                    t0 = time.time()
                    m(x2, mod2, cos2, sin2)
                    torch.cuda.synchronize()
                    warm_new_s = time.time() - t0
                if tier == "default":
                    default_ms = ms
                e = ctx.err(o, ref)
                gated = not gate_compile[dn]
                ctx.row(f"{tier}", shape = shp, dtype = dn, status = WRONG if e.get("nonfinite") else RUNS,
                        engaged = True, err = e, ms = ms, eager_ms = eager_ms, compiled_ms = default_ms,
                        compile_s = round(compile_s, 2), new_prompt_first_call_s = round(new_prompt_s, 3),
                        new_prompt_warm_s = round(warm_new_s, 4),
                        note = f"compile {compile_s:.1f}s; new prompt length first call {new_prompt_s:.2f}s "
                               f"(warm {warm_new_s * 1000:.1f}ms)" + ("; Studio gates this dtype to eager" if gated
                                                                      else ""))
            except Exception as exc:  # noqa: BLE001
                ctx.fail(tier, exc, shape = shp, dtype = dn, eager_ms = eager_ms)
                del m
                torch.cuda.empty_cache()
                continue
            # denoiser-level CUDA graph over the compiled blocks, as Studio's install_cuda_graphs does
            try:
                sx, smod, scos, ssin = (t.clone() for t in (x, mod, cos, sin))
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        m(sx, smod, scos, ssin)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with icfg.patch(patch):
                    with torch.cuda.graph(g):
                        sout = m(sx, smod, scos, ssin)
                g.replay()
                torch.cuda.synchronize()
                gms = ctx.bench(g.replay, iters = max(5, ctx.iters // 2))
                diff = float((sout.float() - o.float()).abs().max())
                e = ctx.err(sout, ref)
                st = RUNS if gate_graph else FALLBACK
                if e.get("nonfinite") or diff > 1e-2 * max(1.0, float(o.float().abs().max())):
                    st = WRONG
                ctx.row(f"{tier}+graph", shape = shp, dtype = dn, status = st, engaged = True, err = e, ms = gms,
                        eager_ms = eager_ms, compiled_ms = default_ms, graph_vs_nograph_max_abs = diff,
                        note = f"capture ok; replay vs compiled max abs {diff:.3g}" +
                               ("" if gate_graph else "; Studio gates graphs off on ROCm (capture works here)"))
            except Exception as exc:  # noqa: BLE001
                ctx.fail(f"{tier}+graph", exc, shape = shp, dtype = dn, eager_ms = eager_ms, compiled_ms = default_ms)
                try:
                    torch.cuda.synchronize()
                except Exception:  # noqa: BLE001
                    pass
            del m
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    return ctx.rows
