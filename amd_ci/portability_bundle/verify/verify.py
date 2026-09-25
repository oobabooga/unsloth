#!/usr/bin/env python3
"""Hand-written Triton kernels vs what torch.compile (Inductor) generates for the same math.

For every kernel in scope the harness writes the computation as plain eager PyTorch ops copied from the STOCK
reference (Diffusers' AutoencoderKLMiniMaxH3 / FeedForward / attention processor, the module's own torch fallback for
the int8 path, ``Tensor.add_`` for the NVFP4 bias), then runs four things on identical inputs:

  ref     the stock function on float64 copies of the inputs (the ground truth)
  ours    the production wrapper from the tree under test (``--tree``), which launches our @triton.jit kernel
  ind     torch.compile(stock function, backend="inductor") at the production dtype; its generated Triton is
          saved to ``--log-dir/<kernel>/<case>__<dtype>.py`` and the kernel names are recorded
  eager   the stock function eager at the production dtype

and reports max abs / max rel / mean abs error of each against ``ref``, ours/ind error ratio, bit identity of ours
against ind and against eager, and CUDA-event median timings.

Usage (from a clean cwd):
  python verify.py --tree <worktree with PR 11801> --bias-tree <worktree with PR 10731> [--kernels a,b] [--big]
  python verify.py --report-only       # merge parts/*.json into results.json + report.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import zlib
from types import SimpleNamespace

WS = os.environ.get("WORKSPACE", "/mnt/disks/unslothai/ubuntu/workspace_81")
H3_REL = "studio/backend/core/inference/video_minimax_h3_vae.py"
BIAS_REL = "studio/backend/core/inference/diffusion_nvfp4_bias.py"

ALL_KERNELS = [
    "gn_stats",  # _gn_partials + _gn_combine
    "gn_silu_pad",  # full norm_silu_pad (partials + combine + apply) and the pad-only path
    "add_residual",
    "add_rmsnorm",
    "qk_norm_rope",
    "swiglu",
    "quant_rows",
    "dequant_epilogue",
    "nvfp4_bias_add",
]
# the constituent @triton.jit kernels each harness entry exercises
TRITON_KERNELS = {
    "gn_stats": ["_gn_partials", "_gn_combine"],
    "gn_silu_pad": ["_gn_partials", "_gn_combine", "_gn_silu_pad"],
    "add_residual": ["_add_residual"],
    "add_rmsnorm": ["_add_rmsnorm"],
    "qk_norm_rope": ["_qk_norm_rope"],
    "swiglu": ["_swiglu"],
    "quant_rows": ["_quant_rows"],
    "dequant_epilogue": ["_dequant_epilogue"],
    "nvfp4_bias_add": ["_bias_add_kernel"],
}


# ── GPU pick (before torch is imported) ────────────────────────────────────────────────────────────────────────────


def pick_gpu(req: str) -> str:
    mask = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if req != "auto":
        if mask and req not in mask:
            raise SystemExit(f"--gpu {req} is outside CUDA_VISIBLE_DEVICES={mask}")
        return req
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output = True, text = True, check = True,
    ).stdout
    best = None
    for line in q.strip().splitlines():
        idx, mem, util = [s.strip() for s in line.split(",")]
        if mask and idx not in mask:
            continue
        key = (int(util) > 5, int(mem))
        if best is None or key < best[0]:
            best = (key, idx)
    if best is None:
        raise SystemExit("no GPU in CUDA_VISIBLE_DEVICES")
    return best[1]


# ── helpers ────────────────────────────────────────────────────────────────────────────────────────────────────────


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_head(tree: str) -> str:
    try:
        return subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output = True, text = True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


class Spy:
    """Wraps a JITFunction: records (grid, args, kwargs) of every launch, so the harness can read the kernel's
    intermediate outputs and replay exactly the production launch for timing."""

    def __init__(self, kern):
        self.kern = kern
        self.calls = []

    def __getitem__(self, grid):
        launcher = self.kern[grid]

        def run(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return launcher(*args, **kwargs)

        return run

    def replay(self, i = -1):
        grid, args, kwargs = self.calls[i]
        self.kern[grid](*args, **kwargs)


class spying:
    def __init__(self, ns, *names):
        self.ns, self.names, self.spies = ns, names, {}

    def __enter__(self):
        for n in self.names:
            self.spies[n] = Spy(getattr(self.ns, n))
            setattr(self.ns, n, self.spies[n])
        return self.spies

    def __exit__(self, *exc):
        for n, s in self.spies.items():
            setattr(self.ns, n, s.kern)


def seed_of(case: str) -> int:
    return zlib.crc32(case.encode()) & 0xFFFF


def flat_pair(a, b):
    """Both tensors flattened in the SAME logical order, as views when both are channels-last 5D."""
    import torch

    cl = torch.channels_last_3d
    if a.dim() == 5 and a.is_contiguous(memory_format = cl) and b.is_contiguous(memory_format = cl):
        return a.permute(0, 2, 3, 4, 1).reshape(-1), b.permute(0, 2, 3, 4, 1).reshape(-1)
    return a.reshape(-1), b.reshape(-1)


def err_stats(out, ref, rel_floor = 1e-3):
    """max abs, max rel (|d| / max(|ref|, rel_floor)), mean abs, all in float64, chunked to bound memory."""
    import torch

    a, r = flat_pair(out, ref)
    assert a.numel() == r.numel(), (out.shape, ref.shape)
    n = a.numel()
    mx = mr = 0.0
    sm = 0.0
    nonfinite = 0
    step = 1 << 26
    for s in range(0, n, step):
        x = a[s : s + step].double()
        y = r[s : s + step].double()
        d = (x - y).abs()
        bad = ~torch.isfinite(d)
        if bad.any():
            nonfinite += int(bad.sum())
            d = torch.where(bad, torch.zeros_like(d), d)
        mx = max(mx, float(d.max()))
        mr = max(mr, float((d / y.abs().clamp_min(rel_floor)).max()))
        sm += float(d.sum())
    res = {"max_abs": mx, "max_rel": mr, "mean_abs": sm / max(n, 1)}
    if nonfinite:
        res["nonfinite"] = nonfinite
    return res


def bit_mismatch(a, b):
    """Number of elements whose bit patterns differ (None if the two cannot be compared)."""
    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return None
    x, y = flat_pair(a, b)
    n = 0
    step = 1 << 27
    for s in range(0, x.numel(), step):
        xs, ys = x[s : s + step], y[s : s + step]
        if xs.is_floating_point():
            iv = {2: torch.int16, 4: torch.int32, 8: torch.int64}[xs.element_size()]
            xs, ys = xs.contiguous().view(iv), ys.contiguous().view(iv)
        n += int((xs != ys).sum())
    return n


def cuda_time(fn, iters, warmup = 10):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing = True)
        e = torch.cuda.Event(enable_timing = True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def inductor_compile(fn, args, save_path, emulate = False):
    """torch.compile(fn) on a fresh dynamo state; returns (output, compiled_fn, kernel_names, code)."""
    import torch
    import torch._inductor.config as icfg
    from torch._inductor.utils import run_and_get_code

    torch._dynamo.reset()
    prev = icfg.emulate_precision_casts
    icfg.emulate_precision_casts = emulate
    try:
        compiled = torch.compile(fn, backend = "inductor", dynamic = False, fullgraph = True)
        out, codes = run_and_get_code(compiled, *args)
    finally:
        icfg.emulate_precision_casts = prev
    code = "\n\n# ======== next graph ========\n\n".join(codes)
    names = re.findall(r"^def (triton_\w+)\(", code, flags = re.M)
    names = list(dict.fromkeys(names))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        with open(save_path, "w") as f:
            f.write(f"# Inductor output_code, kernels: {names}\n")
            f.write(code)
    return out, compiled, names, code


def rbf(t):
    """Round to the bfloat16 grid so a parameter is exactly representable in fp16, bf16 and fp32 alike."""
    import torch

    return t.to(torch.bfloat16).float()


DT = {}


def dtypes(names):
    import torch

    m = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    return [(n, m[n]) for n in names]


# ── stock reference functions (from Diffusers / the module's own torch fallback / add_) ────────────────────────


def stock_norm_silu_pad(x, bias, gw, gb, groups: int, eps: float, pad: tuple, front: int):
    """Diffusers: conv output (+ its bias) -> MiniMaxH3VideoGroupNorm (per-frame) -> F.silu -> CausalConv3d pads
    (reflect in space, zero frames in front), handed to the next conv channels-last. Runs in x.dtype."""
    import torch
    import torch.nn.functional as F

    h = x
    if bias is not None:
        h = h + bias.to(h.dtype).view(1, -1, 1, 1, 1)
    if gw is not None:
        b, c, t, hh, ww = h.shape
        h = h.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, hh, ww)
        h = F.group_norm(h, groups, gw.to(h.dtype), gb.to(h.dtype), eps)
        h = h.view(b, t, c, hh, ww).permute(0, 2, 1, 3, 4).contiguous()
        h = F.silu(h)
    if any(pad):
        h = F.pad(h, (pad[0], pad[1], pad[2], pad[3], 0, 0), mode = "reflect")
    if front:
        h = F.pad(h, (0, 0, 0, 0, front, 0), mode = "constant")
    return h.contiguous(memory_format = torch.channels_last_3d)


def stock_gn_stats(x, bias, groups: int, eps: float):
    """GroupNorm statistics of the per-frame stock GroupNorm, in float32 as autocast runs group_norm (fp64 for the
    reference). Returns (mean, rstd) of shape [B*T*G]."""
    import torch

    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    h = x.to(acc)
    if bias is not None:
        h = h + bias.to(acc).view(1, -1, 1, 1, 1)
    b, c, t, hh, ww = h.shape
    h = h.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, hh * ww)
    _, mean, rstd = torch.ops.aten.native_group_norm(h, None, None, b * t, c, hh * ww, groups, eps)
    return mean.reshape(-1), rstd.reshape(-1)


def stock_add_residual(o, ob, r, rb):
    """MiniMaxH3VideoResnetBlock3d: residual (+ shortcut bias) + conv2 output (+ its bias), channels-last."""
    import torch

    oo = o if ob is None else o + ob.view(1, -1, 1, 1, 1)
    rr = r if rb is None else r + rb.view(1, -1, 1, 1, 1)
    return (rr + oo).to(o.dtype).contiguous(memory_format = torch.channels_last_3d)


def stock_add_rmsnorm(h, o, s, w, eps: float, out_dtype):
    """MiniMaxH3VideoTransformerBlock: hidden = hidden + attn(...) * scale ; norm(hidden.float()).to(hidden.dtype),
    which the next autocast Linear rounds to out_dtype."""
    import torch.nn.functional as F

    import torch

    if o is not None:
        h = h + o * s
    hf = h if h.dtype == torch.float64 else h.float()
    n = F.rms_norm(hf, (h.shape[-1],), w.to(hf.dtype), eps)
    return h, n.to(h.dtype).to(out_dtype)


def stock_qk_norm_rope(x, cos, sin, heads: int, eps: float):
    """MiniMaxH3VideoAttnProcessor: norm_q(query.float()).to(query.dtype), then the partial split-half rope in
    query.dtype. x is [tokens, heads*D], cos/sin are [tokens, R] (already cast to the query dtype)."""
    import torch
    import torch.nn.functional as F

    q = x.unflatten(-1, (heads, -1))
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    q = F.rms_norm(q.to(acc), (q.shape[-1],), None, eps).to(x.dtype)
    c = cos.to(q.dtype).unsqueeze(-2)
    s = sin.to(q.dtype).unsqueeze(-2)
    rot = c.shape[-1]
    q_rot, q_pass = q[..., :rot], q[..., rot:]
    first, second = q_rot.chunk(2, dim = -1)
    rotated = torch.cat([-second, first], dim = -1)
    q = torch.cat([q_rot * c + rotated * s, q_pass], dim = -1)
    return q.flatten(-2)


def stock_swiglu(x):
    """diffusers.models.activations.SwiGLU.forward after the projection."""
    import torch.nn.functional as F

    hidden, gate = x.chunk(2, dim = -1)
    return hidden * F.silu(gate)


def stock_quant_rows(x):
    """video_minimax_h3_vae._int8_linear's torch path: per-row symmetric int8, torch.round (half to even)."""
    import torch

    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    xf = x.to(acc)
    xs = xf.abs().amax(dim = 1).clamp(min = 1e-12) / 127.0
    q = (xf / xs[:, None]).round().clamp(-127, 127)
    return q, xs


def stock_dequant(acc, xs, ws, bias, out_dtype):
    """_int8_linear's torch path epilogue: acc.float() * xs[:, None] * ws[None, :] + bias.float()."""
    import torch

    a = torch.float64 if xs.dtype == torch.float64 else torch.float32
    y = acc.to(a) * xs.to(a)[:, None] * ws.to(a)[None, :]
    if bias is not None:
        y = y + bias.to(a)
    return y.to(out_dtype)


def stock_bias_add(out, bias):
    return out + bias


# ── the harness ────────────────────────────────────────────────────────────────────────────────────────────────────


class Harness:
    def __init__(self, args, h3, bias_mod):
        import torch

        self.args = args
        self.h3 = h3
        self.bias_mod = bias_mod
        self.k = h3._kernels() if h3 is not None else None
        self.torch = torch
        self.dev = torch.device("cuda", 0)
        self.log_dir = args.log_dir
        self.rows = []

    # generic runner -------------------------------------------------------------------------------------------------
    def record(self, kernel, case, dt_name, shape_class, **kw):
        row = {"kernel": kernel, "case": case, "dtype": dt_name, "shape_class": shape_class, **kw}
        ours, ind = row.get("ours"), row.get("inductor")
        if ours and ind:
            oe, ie = ours["max_abs"], ind["max_abs"]
            row["ratio_max"] = 1.0 if oe == ie == 0 else (math.inf if ie == 0 else oe / ie)
            om, im = ours["mean_abs"], ind["mean_abs"]
            row["ratio_mean"] = 1.0 if om == im == 0 else (math.inf if im == 0 else om / im)
        row["verdict"] = self.verdict(row)
        self.rows.append(row)
        rm = row.get("ratio_max")
        print(
            f"[{kernel}] {case:<34} {dt_name:<5} ours={_f(ours and ours['max_abs'])} ind={_f(ind and ind['max_abs'])} "
            f"eager={_f(row.get('eager') and row['eager']['max_abs'])} ratio={_f(rm)} "
            f"bits(ours!=ind)={row.get('bits_vs_inductor')} bits(ours!=eager)={row.get('bits_vs_eager')} "
            f"ms ours={_f(row.get('ours_ms'))} ind={_f(row.get('inductor_ms'))} eager={_f(row.get('eager_ms'))} "
            f"-> {row['verdict']}",
            flush = True,
        )
        return row

    @staticmethod
    def verdict(row):
        if row.get("error"):
            return "ERROR"
        if row.get("fallback"):
            return "FALLBACK(stock path, not our kernel)"
        rm = row.get("ratio_max")
        if rm is None:
            return "n/a"
        if rm <= 2.0:
            return "PASS"
        if row.get("bits_vs_eager") == 0:
            return "EXPECTED(bit-identical to eager rounding)"
        e = row.get("eager")
        if e and row["ours"]["max_abs"] <= 2.0 * e["max_abs"]:
            return "EXPECTED(within 2x of eager; Inductor skips an intermediate rounding)"
        return "FLAG"

    def timings(self, ours_fn, ind_fn, eager_fn):
        it = self.args.iters
        res = {}
        for name, fn in (("ours_ms", ours_fn), ("inductor_ms", ind_fn), ("eager_ms", eager_fn)):
            if fn is None or it <= 0:
                continue
            try:
                res[name] = cuda_time(fn, it)
            except Exception as exc:  # noqa: BLE001
                res[name] = None
                print("timing failed", name, exc)
        return res

    def code_path(self, kernel, case, dt_name, suffix = ""):
        return os.path.join(self.log_dir, kernel, f"{case}__{dt_name}{suffix}.py")

    def run_case(self, kernel, case, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.rows.append({"kernel": kernel, "case": case, "error": repr(exc)[:500], "verdict": "ERROR"})
        finally:
            self.torch.cuda.empty_cache()

    def sel(self, lst):
        return lst[:1] if self.args.quick else lst

    # 1+2. GroupNorm statistics and 3. the fused GroupNorm + SiLU + pad -------------------------------------------
    def encoder_cases(self):
        """(case, shape class, B, C, T, H, W, layout, has_norm, has_bias, pad, front, dtypes). Real shapes: the
        encoder at a 256x256 tile (Diffusers tiles both 1344x768 and 960x544 into 256x256 tiles), clip of 17 frames
        and the single-frame T=1 path (front=0, last tap); untiled 960x544; adversarial ones."""
        full = ["fp16", "bf16", "fp32"]
        f16 = ["fp16"]
        P1 = (1, 1, 1, 1)
        DS = (0, 1, 0, 1)
        cases = [
            ("L0_res0_norm1_T17", "real tile256 (NCDHW in)", 1, 128, 17, 256, 256, "ncdhw", True, True, P1, 2, full),
            ("L0_norm2_T17", "real tile256", 1, 128, 17, 256, 256, "cl", True, True, P1, 2, f16),
            ("L1_T17", "real tile256", 1, 256, 17, 128, 128, "cl", True, True, P1, 2, f16),
            ("L2_T9", "real tile256", 1, 256, 9, 64, 64, "cl", True, True, P1, 2, f16),
            ("L3_T5", "real tile256", 1, 512, 5, 32, 32, "cl", True, True, P1, 2, ["fp16", "fp32"]),
            ("L5_normout_T5", "real tile256 (CPG=32)", 1, 1024, 5, 16, 16, "cl", True, False, P1, 2, full),
            ("L0_T1", "real T=1 single frame", 1, 128, 1, 256, 256, "cl", True, True, P1, 0, full),
            ("L5_T1", "real T=1 single frame", 1, 1024, 1, 16, 16, "cl", True, True, P1, 0, f16),
            ("ds_pad_L0_T17", "real downsample pad-only", 1, 128, 17, 256, 256, "cl", False, False, DS, 2, full),
            ("ds_pad_L1_T1_bias", "real downsample pad-only T=1", 1, 256, 1, 128, 128, "cl", False, True, DS, 0, f16),
            ("untiled_960x544_L0_T17", "real untiled 960x544", 1, 128, 17, 544, 960, "ncdhw", True, True, P1, 2, f16),
            ("odd_37x67_T3", "adversarial W%BLOCK!=0", 1, 128, 3, 37, 67, "cl", True, True, P1, 2, full),
            ("C48_padonly_bias", "adversarial non-pow2 C (pad-only)", 1, 48, 3, 19, 21, "ncdhw", False, True, P1, 2, full),
            ("C96_norm", "adversarial non-pow2 C with norm", 1, 96, 2, 16, 16, "cl", True, True, P1, 2, f16),
            ("tiny_1x1_C32_G32", "adversarial 1x1 spatial, CPG=1", 1, 32, 2, 1, 1, "cl", True, True, (0, 0, 0, 0), 2, full),
            ("tiny_2x2_reflect", "adversarial reflect border 2x2", 1, 64, 1, 2, 2, "ncdhw", True, True, P1, 0, full),
            ("B2_strided_T5", "adversarial batch 2, sliced input", 2, 128, 5, 40, 72, "slice", True, True, P1, 2, f16),
            ("dc_offset_mu30_sd0.1", "adversarial |mean|/std=300", 1, 128, 2, 32, 48, "cl_dc", True, False, P1, 2,
             ["fp16", "fp32"]),
        ]
        if self.args.big:
            cases.append(
                ("BIG_untiled_1344x768_L0_T17", "large numel>2^31 (untiled 1344x768)", 1, 128, 17, 768, 1344, "ncdhw",
                 True, True, P1, 2, ["fp16"])
            )
        if self.args.quick:
            cases = [c for c in cases if c[0] in ("L0_res0_norm1_T17", "odd_37x67_T3", "tiny_1x1_C32_G32")]
        return cases

    def make_enc_inputs(self, B, C, T, H, W, layout, dt, has_norm, has_bias, seed):
        torch = self.torch
        g = torch.Generator(device = "cuda").manual_seed(seed)
        if layout == "slice":
            # a view into a wider tensor, the tile slice Diffusers hands the encoder
            base = torch.randn(B, C, T, H + 3, W + 5, device = "cuda", generator = g, dtype = torch.float32)
            chan_mean = torch.randn(1, C, 1, 1, 1, device = "cuda", generator = g) * 3
            x = (base * 2 + chan_mean).to(dt)[..., 1 : H + 1, 2 : W + 2]
        elif layout == "cl_dc":
            # a large DC offset with a small spread: stresses the folded affine x * a + (beta - mean * a)
            x = (30.0 + 0.1 * torch.randn(B, C, T, H, W, device = "cuda", generator = g)).to(dt)
            x = x.contiguous(memory_format = torch.channels_last_3d)
        else:
            # per-channel offsets so the variance is not the trivial one
            x = torch.empty(B, C, T, H, W, device = "cuda", dtype = dt)
            for t0 in range(T):
                chan_mean = torch.randn(1, C, 1, 1, device = "cuda", generator = g) * 3
                x[:, :, t0] = (torch.randn(B, C, H, W, device = "cuda", generator = g) * 2 + chan_mean).to(dt)
            if layout == "cl":
                x = x.contiguous(memory_format = torch.channels_last_3d)
        gw = rbf(torch.rand(C, device = "cuda", generator = g) + 0.5) if has_norm else None
        gb = rbf(torch.randn(C, device = "cuda", generator = g) * 0.5) if has_norm else None
        bias = rbf(torch.randn(C, device = "cuda", generator = g)).to(dt) if has_bias else None
        return x, gw, gb, bias

    def k_gn_silu_pad(self, stats_only = False):
        torch = self.torch
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoGroupNorm

        kname = "gn_stats" if stats_only else "gn_silu_pad"
        for (case, sclass, B, C, T, H, W, layout, has_norm, has_bias, pad, front, dts) in self.encoder_cases():
            if stats_only and not has_norm:
                continue
            for dt_name, dt in dtypes(dts):
                self.run_case(kname, case, lambda: self._gn_case(
                    kname, stats_only, case, sclass, B, C, T, H, W, layout, has_norm, has_bias, pad, front,
                    dt_name, dt, MiniMaxH3VideoGroupNorm))

    def _gn_case(self, kname, stats_only, case, sclass, B, C, T, H, W, layout, has_norm, has_bias, pad, front,
                 dt_name, dt, GN):
        torch = self.torch
        G = 32 if C % 32 == 0 else 8
        eps = 1e-6
        big = case.startswith("BIG")
        x, gw, gb, bias = self.make_enc_inputs(B, C, T, H, W, layout, dt, has_norm, has_bias, seed = seed_of(case))
        norm = None
        if has_norm:
            norm = GN(G, C, eps = eps, affine = True).cuda()
            with torch.no_grad():
                norm.weight.copy_(gw)
                norm.bias.copy_(gb)
        fusable = self.h3._fusable(x, pad, norm)
        extra = {"fallback": not fusable, "B": B, "C": C, "T": T, "H": H, "W": W, "layout": layout, "pad": pad,
                 "front": front, "numel": x.numel()}
        if stats_only:
            if not fusable:
                return self.record(kname, case, dt_name, sclass, **extra)
            with spying(self.k, "gn_partials", "gn_combine") as sp:
                self.h3.norm_silu_pad(x, norm, pad, front, in_bias = bias)
            call = sp["gn_combine"].calls[-1][1]
            mean_o, rstd_o = call[3].clone(), call[4].clone()
            args = (x, bias, G, eps)
            mean_i, rstd_i = None, None
            (mean_i, rstd_i), comp, names, _ = inductor_compile(
                stock_gn_stats, args, self.code_path(kname, case, dt_name))
            mean_e, rstd_e = stock_gn_stats(*args)
            mean_r, rstd_r = stock_gn_stats(x.double(), None if bias is None else bias.double(), G, eps)
            # rstd error relative to rstd itself (scale-free); mean error absolute
            res = dict(
                ours = err_stats(rstd_o, rstd_r), inductor = err_stats(rstd_i, rstd_r), eager = err_stats(rstd_e, rstd_r),
                ours_mean = err_stats(mean_o, mean_r), inductor_mean = err_stats(mean_i, mean_r),
                eager_mean = err_stats(mean_e, mean_r),
                bits_vs_inductor = bit_mismatch(rstd_o, rstd_i), bits_vs_eager = bit_mismatch(rstd_o, rstd_e),
                inductor_kernels = names, ours_launches = 2, n_out = rstd_o.numel(),
                metric_note = "primary = rstd (fp32); *_mean = per-group mean",
            )

            def ours_t():
                sp["gn_partials"].replay()
                sp["gn_combine"].replay()

            res.update(self.timings(ours_t, lambda: comp(*args), lambda: stock_gn_stats(*args)))
            return self.record(kname, case, dt_name, sclass, **extra, **res)

        # full norm_silu_pad
        with spying(self.k, "gn_partials", "gn_combine", "gn_silu_pad") as sp:
            ours = self.h3.norm_silu_pad(x, norm, pad, front, in_bias = bias)
        launches = sum(len(s.calls) for s in sp.values())
        args = (x, bias, gw, gb, G, eps, pad, front)
        ind, comp, names, _ = inductor_compile(stock_norm_silu_pad, args, self.code_path(kname, case, dt_name))
        if big:
            res = self._gn_big_compare(ours, ind, x, bias, gw, gb, G, eps, pad, front)
            eager = None
        else:
            eager = stock_norm_silu_pad(*args)
            ref = stock_norm_silu_pad(x.double(), None if bias is None else bias.double(),
                                      None if gw is None else gw.double(), None if gb is None else gb.double(),
                                      G, eps, pad, front)
            res = dict(ours = err_stats(ours, ref), inductor = err_stats(ind, ref), eager = err_stats(eager, ref))
            # the zero frames in front must be exactly zero (a pending bias must not leak into them)
            if front:
                res["front_frames_exact_zero"] = bool((ours[:, :, :front] == 0).all())
            # reflect border rows/cols: error on the padded border alone
            if any(pad):
                bm = torch.zeros(ours.shape[-2:], dtype = torch.bool, device = "cuda")
                if pad[2]:
                    bm[: pad[2]] = True
                if pad[3]:
                    bm[-pad[3]:] = True
                if pad[0]:
                    bm[:, : pad[0]] = True
                if pad[1]:
                    bm[:, -pad[1]:] = True
                d = (ours.double() - ref)[..., bm]
                res["border_max_abs"] = float(d.abs().max()) if d.numel() else 0.0
                di = (ind.double() - ref)[..., bm]
                res["border_max_abs_inductor"] = float(di.abs().max()) if di.numel() else 0.0
            del ref
        res["bits_vs_inductor"] = bit_mismatch(ours, ind)
        if eager is not None:
            res["bits_vs_eager"] = bit_mismatch(ours, eager)
        res["out_strides_match_inductor"] = tuple(ours.stride()) == tuple(ind.stride())
        del ind, eager
        res.update(inductor_kernels = names, ours_launches = launches)
        res.update(self.timings(
            lambda: self.h3.norm_silu_pad(x, norm, pad, front, in_bias = bias),
            lambda: comp(*args),
            None if big else (lambda: stock_norm_silu_pad(*args)),
        ))
        return self.record(kname, case, dt_name, sclass, **extra, **res)

    def _gn_big_compare(self, ours, ind, x, bias, gw, gb, G, eps, pad, front):
        """numel > 2^31: float64 reference on the first and last input frames only (the last frame carries the
        largest offsets, where an int32 index would wrap); ours vs Inductor over the whole tensor."""
        T = x.shape[2]
        out = {}
        worst = {"ours": None, "inductor": None}
        for t0 in (0, T - 1):
            xs = x[:, :, t0 : t0 + 1]
            ref = stock_norm_silu_pad(xs.double(), bias.double() if bias is not None else None, gw.double(),
                                      gb.double(), G, eps, pad, 0)
            for nm, full in (("ours", ours), ("inductor", ind)):
                e = err_stats(full[:, :, t0 + front : t0 + front + 1], ref)
                if worst[nm] is None or e["max_abs"] > worst[nm]["max_abs"]:
                    worst[nm] = e
            del ref
        out.update(worst)
        out["big_ref_frames"] = [0, T - 1]
        out["front_frames_exact_zero"] = bool((ours[:, :, :front] == 0).all())
        out["ours_vs_inductor_full"] = err_stats(ours, ind)
        return out

    # 4. add_residual ------------------------------------------------------------------------------------------------
    def k_add_residual(self):
        torch = self.torch
        full = ["fp16", "bf16", "fp32"]
        # (case, class, B, C, T, H, W, res layout, has_ob, has_rb, rb fp32 (folded 1x1 bias), dts)
        cases = [
            ("L0_res0_ncdhw_res", "real tile256 (conv_in out as skip)", 1, 128, 17, 256, 256, "ncdhw", True, True, False, full),
            ("L1_shortcut_folded", "real tile256 (1x1 shortcut, folded fp32 bias)", 1, 256, 17, 128, 128, "cl", True, True, True, ["fp16"]),
            ("L3_T5", "real tile256", 1, 512, 5, 32, 32, "cl", True, False, False, ["fp16", "fp32"]),
            ("L5_T5", "real tile256", 1, 1024, 5, 16, 16, "cl", True, False, False, full),
            ("L0_T1", "real T=1", 1, 128, 1, 256, 256, "cl", True, True, False, full),
            ("untiled_960x544_L0_T17", "real untiled 960x544", 1, 128, 17, 544, 960, "ncdhw", True, True, False, ["fp16"]),
            ("odd_37x67_T3", "adversarial pixels%BLOCK_P!=0", 1, 128, 3, 37, 67, "ncdhw", True, True, False, full),
            ("tiny_1x1_C8", "adversarial 1x1, C=8", 1, 8, 1, 1, 1, "cl", True, True, False, full),
            ("C48_fallback", "adversarial non-pow2 C", 1, 48, 3, 9, 11, "cl", True, True, False, ["fp16"]),
            ("B2_T5_sliced_res", "adversarial batch 2, sliced res", 2, 256, 5, 40, 72, "slice", False, True, False, ["fp16"]),
        ]
        if self.args.big:
            cases.append(("BIG_untiled_1344x768_L0_T17", "large numel>2^31", 1, 128, 17, 768, 1344, "ncdhw", True, True,
                          False, ["fp16"]))
        if self.args.quick:
            cases = cases[:1] + cases[6:8]
        for c in cases:
            for dt_name, dt in dtypes(c[-1]):
                self.run_case("add_residual", c[0], lambda: self._res_case(*c[:-1], dt_name, dt))

    def _res_case(self, case, sclass, B, C, T, H, W, rlayout, has_ob, has_rb, rb32, dt_name, dt):
        torch = self.torch
        g = torch.Generator(device = "cuda").manual_seed(seed_of(case))
        big = case.startswith("BIG")
        o = torch.randn(B, C, T, H, W, device = "cuda", generator = g, dtype = torch.float32).to(dt) \
            .contiguous(memory_format = torch.channels_last_3d)
        if rlayout == "slice":
            base = torch.randn(B, C, T, H + 2, W + 4, device = "cuda", generator = g).to(dt)
            r = base[..., 1 : H + 1, 3 : W + 3]
        else:
            r = torch.randn(B, C, T, H, W, device = "cuda", generator = g, dtype = torch.float32).to(dt)
            if rlayout == "cl":
                r = r.contiguous(memory_format = torch.channels_last_3d)
        ob = rbf(torch.randn(C, device = "cuda", generator = g)).to(dt) if has_ob else None
        rb = None
        if has_rb:
            rb = torch.randn(C, device = "cuda", generator = g)
            rb = rb if rb32 else rbf(rb).to(dt)
        extra = {"B": B, "C": C, "T": T, "H": H, "W": W, "res_layout": rlayout, "numel": o.numel(),
                 "fallback": bool(C & (C - 1) or C > 1024)}
        with spying(self.k, "add_residual") as sp:
            ours = self.h3.add_residual(o.clone(memory_format = torch.channels_last_3d), ob, r, rb)
        args = (o, ob, r, rb)
        ind, comp, names, _ = inductor_compile(stock_add_residual, args, self.code_path("add_residual", case, dt_name))
        res = {}
        if big:
            ts = [0, T - 1]
            worst = {}
            for t0 in ts:
                ref = stock_add_residual(o[:, :, t0:t0 + 1].double(), ob.double(), r[:, :, t0:t0 + 1].double(), rb.double())
                for nm, full in (("ours", ours), ("inductor", ind)):
                    e = err_stats(full[:, :, t0:t0 + 1], ref)
                    if nm not in worst or e["max_abs"] > worst[nm]["max_abs"]:
                        worst[nm] = e
                del ref
            res.update(worst)
            res["ours_vs_inductor_full"] = err_stats(ours, ind)
            eager = None
        else:
            eager = stock_add_residual(*args)
            ref = stock_add_residual(o.double(), None if ob is None else ob.double(), r.double(),
                                     None if rb is None else rb.double())
            res.update(ours = err_stats(ours, ref), inductor = err_stats(ind, ref), eager = err_stats(eager, ref))
            res["bits_vs_eager"] = bit_mismatch(ours, eager)
            del ref
        res["bits_vs_inductor"] = bit_mismatch(ours, ind)
        res.update(inductor_kernels = names, ours_launches = len(sp["add_residual"].calls))
        scratch = o.clone(memory_format = torch.channels_last_3d)
        res.update(self.timings(
            lambda: self.h3.add_residual(scratch, ob, r, rb) if not sp["add_residual"].calls else sp["add_residual"].replay(),
            lambda: comp(*args),
            None if big else (lambda: stock_add_residual(*args)),
        ))
        if sp["add_residual"].calls:
            # replay() ran on the ORIGINAL output buffer `ours` (its args); timing only, accuracy was taken above
            pass
        return self.record("add_residual", case, dt_name, sclass, **extra, **res)

    # 5. add_rmsnorm ---------------------------------------------------------------------------------------------------
    def decoder_rows(self):
        """Decoder token counts: 256x256 tiles (every 1344x768 / 960x544 tile) over a 7-latent-frame clip = 16*16*7
        + 4 registers + 1 cls = 1797 tokens, T=1 = 261, batched 4 tiles; and the untiled decodes."""
        return [
            ("tile256_T7_batch4", "real 1344x768/960x544 tile batch", 4 * 1797),
            ("tile256_T1_batch4", "real T=1 tile batch", 4 * 261),
            ("untiled_1344x768_T7", "real untiled 1344x768", 7 * 48 * 84 + 5),
            ("untiled_960x544_T7", "real untiled 960x544", 7 * 34 * 60 + 5),
        ]

    def k_add_rmsnorm(self):
        torch = self.torch
        cases = []
        for case, sclass, rows in self.decoder_rows():
            cases.append((case, sclass, rows, 2048, True, "prod"))
        cases += [
            ("first_norm1_no_res", "real (HAS_RES=False)", 4 * 1797, 2048, False, "prod"),
            ("N1000_nonpow2", "adversarial non-pow2 N", 333, 1000, True, "prod"),
            ("N1_rows1", "adversarial tiny", 1, 1, True, "prod"),
            ("N3_rows7", "adversarial tiny odd", 7, 3, True, "prod"),
        ]
        if self.args.big:
            cases.append(("BIG_rows_x_N_gt_2^31", "large numel>2^31", (2**31) // 2048 + 3, 2048, True, "prod"))
        if self.args.quick:
            cases = cases[:1] + cases[-3:-1]
        # (name, residual-stream dtype, o dtype, gemm dtype): fp16 autocast (production), bf16 autocast, no autocast
        # fp32, and a float16 residual stream (float16 weights, no autocast)
        variants = [("fp16", "fp32", "fp16", "fp16"), ("bf16", "fp32", "bf16", "bf16"), ("fp32", "fp32", "fp32", "fp32"),
                    ("fp16_h16", "fp16", "fp16", "fp16")]
        m = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        for c in cases:
            vs = variants if (c[0] in ("tile256_T7_batch4", "N1000_nonpow2", "N3_rows7")) else variants[:1]
            for v in vs:
                self.run_case("add_rmsnorm", c[0], lambda: self._rms_case(*c[:5], v[0], m[v[1]], m[v[2]], m[v[3]]))

    def _rms_case(self, case, sclass, rows, N, has_res, dt_name, hdt, odt, gdt):
        torch = self.torch
        g = torch.Generator(device = "cuda").manual_seed(seed_of(case))
        big = case.startswith("BIG")
        h = (torch.randn(rows, N, device = "cuda", generator = g) * 4).to(hdt)
        o = (torch.randn(rows, N, device = "cuda", generator = g)).to(odt) if has_res else None
        s = (torch.randn(N, device = "cuda", generator = g) * 0.1).to(hdt)
        w = (torch.rand(N, device = "cuda", generator = g) + 0.5)
        eps = 1e-5
        norm = SimpleNamespace(weight = w, eps = eps)
        h_ours = h.clone()
        with spying(self.k, "add_rmsnorm") as sp:
            n_ours = self.h3._add_rmsnorm(h_ours, o, s if has_res else None, norm, gdt)
        args = (h, o, s, w, eps, gdt)
        (h_ind, n_ind), comp, names, _ = inductor_compile(stock_add_rmsnorm, args,
                                                          self.code_path("add_rmsnorm", case, dt_name))
        res = {"rows": rows, "N": N, "has_res": has_res, "h_dtype": str(hdt), "o_dtype": str(odt), "gemm_dtype": str(gdt)}
        if big:
            idx = [slice(0, 1024), slice(rows - 1024, rows)]
            worst = {}
            for sl in idx:
                hr, nr = stock_add_rmsnorm(h[sl].double(), o[sl].double(), s.double(), w.double(), eps, torch.float64)
                for nm, (hh, nn) in (("ours", (h_ours, n_ours)), ("inductor", (h_ind, n_ind))):
                    e = err_stats(nn[sl], nr)
                    eh = err_stats(hh[sl], hr)
                    if nm not in worst or e["max_abs"] > worst[nm]["max_abs"]:
                        worst[nm] = e
                        worst[nm + "_h"] = eh
            res.update(worst)
            res["ours_vs_inductor_full"] = err_stats(n_ours, n_ind)
            res["bits_vs_inductor_h"] = bit_mismatch(h_ours, h_ind)
        else:
            h_e, n_e = stock_add_rmsnorm(*args)
            h_r, n_r = stock_add_rmsnorm(h.double(), None if o is None else o.double(), s.double(), w.double(), eps,
                                         torch.float64)
            res.update(ours = err_stats(n_ours, n_r), inductor = err_stats(n_ind, n_r), eager = err_stats(n_e, n_r),
                       ours_h = err_stats(h_ours, h_r), inductor_h = err_stats(h_ind, h_r), eager_h = err_stats(h_e, h_r),
                       bits_vs_eager = bit_mismatch(n_ours, n_e), bits_vs_eager_h = bit_mismatch(h_ours, h_e),
                       bits_vs_inductor_h = bit_mismatch(h_ours, h_ind))
        res["bits_vs_inductor"] = bit_mismatch(n_ours, n_ind)
        res.update(inductor_kernels = names, ours_launches = len(sp["add_rmsnorm"].calls),
                   metric_note = "primary = normed output (gemm dtype); *_h = residual stream after the add")
        res.update(self.timings(lambda: sp["add_rmsnorm"].replay(), lambda: comp(*args),
                                None if big else (lambda: stock_add_rmsnorm(*args))))
        return self.record("add_rmsnorm", case, dt_name, sclass, **res)

    # 6. qk_norm_rope -------------------------------------------------------------------------------------------------
    def rope_tables(self, batch, frames, hgt, wid, head_dim, ratio, dt):
        """cos/sin exactly as MiniMaxH3VideoViTDecoder3d.forward builds them, cast to the query dtype."""
        torch = self.torch
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed

        rope = MiniMaxH3VideoRotaryPosEmbed(int(head_dim * ratio), theta = 100.0).cuda()
        grids = [2.0 * (torch.arange(0.5, size, dtype = torch.float32, device = "cuda") / size) - 1.0
                 for size in (frames, hgt, wid)]
        pos = torch.stack(torch.meshgrid(*grids, indexing = "ij"), dim = -1).flatten(0, 2)
        pos = pos.unsqueeze(0).expand(batch, -1, -1)
        pos = torch.cat([pos, pos.new_zeros((batch, 5, 3))], dim = 1)
        cos, sin = rope(pos)
        cos = cos.to(dt).contiguous().reshape(-1, cos.shape[-1])
        sin = sin.to(dt).contiguous().reshape(-1, sin.shape[-1])
        return cos, sin

    def k_qk_norm_rope(self):
        torch = self.torch
        full = ["fp16", "bf16", "fp32"]
        # (case, class, batch, frames, h, w, heads, head_dim, ratio, dts)
        cases = [
            ("tile256_T7_batch4", "real 1344x768/960x544 tile batch", 4, 7, 16, 16, 32, 64, 0.75, full),
            ("tile256_T1_batch4", "real T=1 tile batch", 4, 1, 16, 16, 32, 64, 0.75, ["fp16"]),
            ("untiled_1344x768_T7", "real untiled 1344x768", 1, 7, 48, 84, 32, 64, 0.75, ["fp16"]),
            ("untiled_960x544_T7", "real untiled 960x544", 1, 7, 34, 60, 32, 64, 0.75, ["fp16"]),
            ("heads3_rows_not_mult_32", "adversarial rows%BLOCK!=0", 1, 1, 2, 3, 3, 64, 0.75, full),
            ("D16_R12_heads1_1tok", "adversarial tiny head", 1, 1, 1, 1, 1, 16, 0.75, full),
            ("D128_R96", "adversarial wider head", 2, 2, 5, 7, 4, 128, 0.75, full),
            ("D32_R24", "adversarial narrow head", 1, 2, 4, 5, 8, 32, 0.75, ["fp16"]),
        ]
        if self.args.big:
            # tokens * 2048 > 2^31
            cases.append(("BIG_tokens_x_2048_gt_2^31", "large numel>2^31", 1, 1, 1024, 1025, 32, 64, 0.75, ["fp16"]))
        if self.args.quick:
            cases = cases[:1] + cases[4:6]
        for c in cases:
            for dt_name, dt in dtypes(c[-1]):
                self.run_case("qk_norm_rope", c[0], lambda: self._qk_case(*c[:-1], dt_name, dt))

    def _qk_case(self, case, sclass, batch, frames, hh, ww, heads, D, ratio, dt_name, dt):
        torch = self.torch
        g = torch.Generator(device = "cuda").manual_seed(seed_of(case))
        big = case.startswith("BIG")
        cos, sin = self.rope_tables(batch, frames, hh, ww, D, ratio, dt)
        tokens = cos.shape[0]
        # to_q output: per-channel scale spread like a projection's, so the RMS actually normalises
        x = (torch.randn(tokens, heads * D, device = "cuda", generator = g)
             * (torch.rand(heads * D, device = "cuda", generator = g) * 3 + 0.2)).to(dt)
        eps = 1e-5
        xo = x.clone()
        with spying(self.k, "qk_norm_rope") as sp:
            self.h3._qk_norm_rope_(xo, cos, sin, heads, eps)
        args = (x, cos, sin, heads, eps)
        ind, comp, names, _ = inductor_compile(stock_qk_norm_rope, args, self.code_path("qk_norm_rope", case, dt_name))
        res = {"tokens": tokens, "heads": heads, "head_dim": D, "rot": cos.shape[-1]}
        if big:
            worst = {}
            for sl in (slice(0, 4096), slice(tokens - 4096, tokens)):
                ref = stock_qk_norm_rope(x[sl].double(), cos[sl].double(), sin[sl].double(), heads, eps)
                for nm, full in (("ours", xo), ("inductor", ind)):
                    e = err_stats(full[sl], ref)
                    if nm not in worst or e["max_abs"] > worst[nm]["max_abs"]:
                        worst[nm] = e
            res.update(worst)
            res["ours_vs_inductor_full"] = err_stats(xo, ind)
        else:
            eager = stock_qk_norm_rope(*args)
            ref = stock_qk_norm_rope(x.double(), cos.double(), sin.double(), heads, eps)
            res.update(ours = err_stats(xo, ref), inductor = err_stats(ind, ref), eager = err_stats(eager, ref),
                       bits_vs_eager = bit_mismatch(xo, eager))
            # Inductor with emulate_precision_casts: rounds every intermediate like eager
            try:
                ind_em, _, _, _ = inductor_compile(stock_qk_norm_rope, args,
                                                   self.code_path("qk_norm_rope", case, dt_name, "__emulate"), emulate = True)
                res["inductor_emulate"] = err_stats(ind_em, ref)
                res["bits_vs_inductor_emulate"] = bit_mismatch(xo, ind_em)
                res["bits_inductor_emulate_vs_eager"] = bit_mismatch(ind_em, eager)
            except Exception as exc:  # noqa: BLE001
                res["inductor_emulate_error"] = repr(exc)[:200]
        res["bits_vs_inductor"] = bit_mismatch(xo, ind)
        res.update(inductor_kernels = names, ours_launches = len(sp["qk_norm_rope"].calls))
        res.update(self.timings(lambda: sp["qk_norm_rope"].replay(), lambda: comp(*args),
                                None if big else (lambda: stock_qk_norm_rope(*args))))
        return self.record("qk_norm_rope", case, dt_name, sclass, **res)

    # 7. swiglu ---------------------------------------------------------------------------------------------------------
    def k_swiglu(self):
        full = ["fp16", "bf16", "fp32"]
        cases = [(c, s, r, 8192, full if c == "tile256_T7_batch4" else ["fp16"]) for c, s, r in self.decoder_rows()]
        cases += [
            ("N1000_not_mult_BLOCK", "adversarial N%2048!=0", 77, 1000, full),
            ("N3_rows1", "adversarial tiny", 1, 3, full),
            ("N2049", "adversarial N=BLOCK+1", 5, 2049, ["fp16"]),
        ]
        if self.args.big:
            cases.append(("BIG_rows_x_2N_gt_2^31", "large numel>2^31", (2**31) // 16384 + 7, 8192, ["fp16"]))
        if self.args.quick:
            cases = cases[:1] + cases[4:6]
        for c in cases:
            for dt_name, dt in dtypes(c[-1]):
                self.run_case("swiglu", c[0], lambda: self._swiglu_case(*c[:-1], dt_name, dt))

    def _swiglu_case(self, case, sclass, rows, N, dt_name, dt):
        torch = self.torch
        g = torch.Generator(device = "cuda").manual_seed(seed_of(case))
        big = case.startswith("BIG")
        x = (torch.randn(rows, 2 * N, device = "cuda", generator = g) * 3).to(dt)
        with spying(self.k, "swiglu") as sp:
            ours = self.h3._swiglu(x)
        ind, comp, names, _ = inductor_compile(stock_swiglu, (x,), self.code_path("swiglu", case, dt_name))
        res = {"rows": rows, "N": N}
        if big:
            worst = {}
            for sl in (slice(0, 256), slice(rows - 256, rows)):
                ref = stock_swiglu(x[sl].double())
                for nm, full in (("ours", ours), ("inductor", ind)):
                    e = err_stats(full[sl], ref)
                    if nm not in worst or e["max_abs"] > worst[nm]["max_abs"]:
                        worst[nm] = e
            res.update(worst)
            res["ours_vs_inductor_full"] = err_stats(ours, ind)
        else:
            eager = stock_swiglu(x)
            ref = stock_swiglu(x.double())
            res.update(ours = err_stats(ours, ref), inductor = err_stats(ind, ref), eager = err_stats(eager, ref),
                       bits_vs_eager = bit_mismatch(ours, eager))
            try:
                ind_em, _, _, _ = inductor_compile(stock_swiglu, (x,), self.code_path("swiglu", case, dt_name, "__emulate"),
                                                   emulate = True)
                res["inductor_emulate"] = err_stats(ind_em, ref)
                res["bits_vs_inductor_emulate"] = bit_mismatch(ours, ind_em)
                res["bits_inductor_emulate_vs_eager"] = bit_mismatch(ind_em, eager)
            except Exception as exc:  # noqa: BLE001
                res["inductor_emulate_error"] = repr(exc)[:200]
        res["bits_vs_inductor"] = bit_mismatch(ours, ind)
        res.update(inductor_kernels = names, ours_launches = len(sp["swiglu"].calls))
        res.update(self.timings(lambda: sp["swiglu"].replay(), lambda: comp(x),
                                None if big else (lambda: stock_swiglu(x))))
        return self.record("swiglu", case, dt_name, sclass, **res)

    # 8+9. int8 quant / dequant through the production _int8_linear ---------------------------------------------------
    def k_int8(self, which):
        full = ["fp16", "bf16", "fp32"]
        cases = [
            ("tile256_T7_b4_to_q", "real to_q/k/v/out (in 2048)", 4 * 1797, 2048, 2048, full),
            ("tile256_T7_b4_ff_out", "real ff out (in 8192)", 4 * 1797, 8192, 2048, ["fp16"]),
            ("tile256_T7_b4_ff_in", "real ff proj (out 16384)", 4 * 1797, 2048, 16384, ["fp16"]),
            ("tile256_T1_b4_to_q", "real T=1", 4 * 261, 2048, 2048, ["fp16"]),
            ("rows17_n1000", "adversarial non-pow2 N, rows just over the GEMV cut", 17, 1000, 1000, full),
            ("n_out_1025", "adversarial n_out%BLOCK!=0", 33, 256, 1025 + 7, ["fp16"]),
            ("zero_and_tie_rows", "adversarial all-zero row + exact .5 ties", 32, 256, 256, full),
        ]
        if self.args.quick:
            cases = cases[:1] + cases[-2:]
        for c in cases:
            for dt_name, dt in dtypes(c[-1]):
                self.run_case(which, c[0], lambda: self._int8_case(which, *c[:-1], dt_name, dt))

    def _int8_case(self, which, case, sclass, rows, n_in, n_out, dt_name, dt):
        torch = self.torch
        g = torch.Generator(device = "cuda").manual_seed(seed_of(case))
        x = (torch.randn(rows, n_in, device = "cuda", generator = g)
             * (torch.rand(n_in, device = "cuda", generator = g) * 4 + 0.1)).to(dt)
        if case == "zero_and_tie_rows":
            x[0] = 0
            # a row whose absmax is 127 so s == 1 exactly, holding exact ties k + 0.5 (representable in every dtype)
            x[1] = 0
            x[1, 0] = 127
            ties = torch.arange(-40, 40, device = "cuda", dtype = torch.float32) + 0.5
            x[1, 1 : 1 + ties.numel()] = ties.to(dt)
            # 0.5 - 2^-25 would need a float32 input; with s == 1 the fp16 nearest is exactly representable only as
            # 0.4998, so this row checks the floor(q + 0.5) path near .5 as well
            x[1, 100] = torch.tensor(0.49975586, dtype = dt)
        wq = torch.randint(-127, 128, (n_out, n_in), device = "cuda", generator = g, dtype = torch.int8)
        ws = torch.rand(n_out, device = "cuda", generator = g) * 0.01 + 1e-4
        bias = torch.randn(n_out, device = "cuda", generator = g)
        lin = SimpleNamespace(_unsloth_int8 = (wq, ws), bias = bias, _unsloth_rot = None)
        with spying(self.k, "quant_rows", "dequant_epilogue") as sp:
            with torch.autocast("cuda", enabled = False):
                y_ours = self.h3._int8_linear(lin, x)
        if not sp["quant_rows"].calls:
            return self.record(which, case, dt_name, sclass, fallback = True)
        _, qargs, _ = sp["quant_rows"].calls[-1]
        x2, xq_o, xs_o = qargs[0], qargs[1], qargs[2]
        _, dargs, _ = sp["dequant_epilogue"].calls[-1]
        acc, dxs, dws, dbias, dout = dargs[0], dargs[1], dargs[2], dargs[3], dargs[4]
        res = {"rows": rows, "n_in": n_in, "n_out": n_out}
        if which == "quant_rows":
            (q_i, s_i), comp, names, _ = inductor_compile(stock_quant_rows, (x2,), self.code_path(which, case, dt_name))
            q_e, s_e = stock_quant_rows(x2)
            q_r, s_r = stock_quant_rows(x2.double())
            xd = x2.double()
            # reconstruction error of q * s against the input, the quantity the GEMM consumes
            rec = lambda q, s: q.double() * s.double()[:, None]  # noqa: E731
            res.update(
                ours = err_stats(rec(xq_o, xs_o), xd), inductor = err_stats(rec(q_i, s_i), xd),
                eager = err_stats(rec(q_e, s_e), xd),
                q_mismatch_vs_fp64 = int((xq_o.double() != q_r).sum()),
                q_mismatch_inductor_vs_fp64 = int((q_i.double() != q_r).sum()),
                q_mismatch_eager_vs_fp64 = int((q_e.double() != q_r).sum()),
                bits_vs_inductor = int((xq_o.to(torch.int16) != q_i.to(torch.int16)).sum())
                + (bit_mismatch(xs_o, s_i) or 0),
                bits_vs_eager = int((xq_o.to(torch.int16) != q_e.to(torch.int16)).sum()) + (bit_mismatch(xs_o, s_e) or 0),
                scale_bits_vs_eager = bit_mismatch(xs_o, s_e),
                metric_note = "primary = |q*s - x| reconstruction; bits = int8 codes + fp32 scales differing",
            )
            if case == "zero_and_tie_rows":
                res["tie_row_ours"] = xq_o[1, 1:81].tolist()
                res["tie_row_torch_round"] = q_e[1, 1:81].to(torch.int8).tolist()
                res["near_half_ours_vs_torch"] = [int(xq_o[1, 100]), int(q_e[1, 100])]
                res["zero_row_ok"] = bool((xq_o[0] == 0).all() and torch.isfinite(xs_o[0]))
            res.update(inductor_kernels = names, ours_launches = 1)
            res.update(self.timings(lambda: sp["quant_rows"].replay(), lambda: comp(x2), lambda: stock_quant_rows(x2)))
        else:
            out_dtype = dout.dtype
            args = (acc, dxs, dws, dbias, out_dtype)
            ind, comp, names, _ = inductor_compile(stock_dequant, args, self.code_path(which, case, dt_name))
            eager = stock_dequant(*args)
            ref = stock_dequant(acc, dxs.double(), dws.double(), dbias.double(), torch.float64)
            res.update(ours = err_stats(dout, ref), inductor = err_stats(ind, ref), eager = err_stats(eager, ref),
                       bits_vs_inductor = bit_mismatch(dout, ind), bits_vs_eager = bit_mismatch(dout, eager),
                       inductor_kernels = names, ours_launches = 1)
            res.update(self.timings(lambda: sp["dequant_epilogue"].replay(), lambda: comp(*args),
                                    lambda: stock_dequant(*args)))
        return self.record(which, case, dt_name, sclass, **res)

    # NVFP4 bias add (PR 10731) ---------------------------------------------------------------------------------------
    def k_bias(self):
        torch = self.torch
        fb = self.bias_mod
        full = ["bf16", "fp16"]
        cases = [
            ("flux_4608x3072", "real (Flux 1024^2 joint tokens x hidden)", 4608, 3072, full),
            ("flux_4608x12288", "real (Flux MLP up)", 4608, 12288, full),
            ("bench_4096x12288", "real (PR bench shape)", 4096, 12288, ["bf16"]),
            ("bench_16384x12288", "real (PR bench shape)", 16384, 12288, ["bf16"]),
            ("wan_32760x5120", "real (Wan 720p tokens x hidden)", 32760, 5120, ["bf16"]),
            ("below_floor_1024x3840", "adversarial below 12M floor (direct launch)", 1024, 3840, full),
            ("odd_N3_rows1001", "adversarial N=3, numel%BLOCK!=0 (direct launch)", 1001, 3, full),
            ("N1_rows5", "adversarial N=1 (direct launch)", 5, 1, full),
            ("fp32_direct", "adversarial fp32 (not eligible; direct launch)", 777, 1537, ["fp32"]),
        ]
        if self.args.big:
            cases += [
                ("BIG_numel_2^31-8192", "large numel just under 2^31-1 (eligible)", 174762, 12288, ["bf16"]),
                ("BIG_numel_gt_2^31", "large numel > 2^31-1 (must fall back)", 174763, 12288, ["bf16"]),
            ]
        if self.args.quick:
            cases = cases[:1] + cases[6:8]
        for c in cases:
            for dt_name, dt in dtypes(c[-1]):
                self.run_case("nvfp4_bias_add", c[0], lambda: self._bias_case(fb, *c[:-1], dt_name, dt))

    def _bias_case(self, fb, case, sclass, M, N, dt_name, dt):
        torch = self.torch
        g = torch.Generator(device = "cuda").manual_seed(seed_of(case))
        big = case.startswith("BIG")
        out = torch.randn(M, N, device = "cuda", generator = g).to(dt)
        bias = torch.randn(N, device = "cuda", generator = g).to(dt)
        eligible = bool(fb._eligible(out, bias)) and fb.fast_bias_enabled()
        launches = []
        orig = fb._bias_add_kernel
        spy = Spy(orig)
        fb._bias_add_kernel = spy
        try:
            o = out.clone()
            if eligible:
                got = fb.fused_bias_add_(o, bias)
                mode = "wrapper"
            elif not big:
                grid = ((o.numel() + fb._BLOCK - 1) // fb._BLOCK,)
                spy[grid](o, bias, o.numel(), N, fb._BLOCK, num_warps = 8)
                got, mode = o, "direct"
            else:
                got = fb.fused_bias_add_(o, bias)
                mode = "wrapper(fallback expected)"
        finally:
            fb._bias_add_kernel = orig
        res = {"M": M, "N": N, "numel": out.numel(), "eligible": eligible, "mode": mode,
               "kernel_launched": len(spy.calls) > 0}
        ind, comp, names, _ = inductor_compile(stock_bias_add, (out, bias), self.code_path("nvfp4_bias_add", case, dt_name))
        eager = out.clone().add_(bias)
        if big:
            worst = {}
            for sl in (slice(0, 1024), slice(M - 1024, M)):
                ref = out[sl].double() + bias.double()
                for nm, full in (("ours", got), ("inductor", ind), ("eager", eager)):
                    e = err_stats(full[sl], ref)
                    if nm not in worst or e["max_abs"] > worst[nm]["max_abs"]:
                        worst[nm] = e
            res.update(worst)
        else:
            ref = out.double() + bias.double()
            res.update(ours = err_stats(got, ref), inductor = err_stats(ind, ref), eager = err_stats(eager, ref))
        res.update(bits_vs_inductor = bit_mismatch(got, ind), bits_vs_eager = bit_mismatch(got, eager),
                   inductor_kernels = names, ours_launches = len(spy.calls))
        scratch = out.clone()
        if spy.calls:
            gr, a, kw = spy.calls[-1]
            a = (scratch,) + tuple(a[1:])
            ours_t = lambda: orig[gr](*a, **kw)  # noqa: E731
        else:
            ours_t = lambda: fb.fused_bias_add_(scratch, bias)  # noqa: E731
        res.update(self.timings(ours_t, lambda: comp(out, bias), lambda: scratch.add_(bias)))
        return self.record("nvfp4_bias_add", case, dt_name, sclass, **res)


def _f(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        return f"{v:.3g}"
    return str(v)


def explain(r):
    """Why a row that is not a plain PASS is still acceptable (or not). Rule-based, so a rerun regenerates it."""
    k, c, v, dt = r["kernel"], r.get("case", ""), r.get("verdict", ""), r.get("dtype", "")
    if v.startswith("PASS"):
        return None
    if v.startswith("FALLBACK"):
        return ("not our kernel: the wrapper's guard sends this shape (non-power-of-two C) to the stock torch path; "
                "listed to show the guard is correct")
    if c.startswith("BIG") and r.get("eager") is None and k in ("qk_norm_rope", "swiglu"):
        return ("numel > 2^31: eager skipped for memory, so the verdict falls to FLAG; the error equals the real-shape "
                "row, where ours is (near) bit-identical to eager. No index wrap: first/last rows match the reference")
    if k == "gn_stats":
        return ("rstd within 1-2 ulp: ours is 1.0 / tl.sqrt (approximate sqrt + div.full) where eager/Inductor use rsqrt. "
                "On these tiny groups the error is an ulp of rstd (a 1x1 group has var = 0, so rstd = eps^-1/2 = 1000 where "
                "1 ulp = 6.1e-5, relative 6e-8)")
    if k == "gn_silu_pad" and "tiny_1x1" in c:
        return ("zero-variance group (1x1, CPG=1): the folded affine x*(rstd*g) + (b - mean*rstd*g) cancels two ~3000-sized "
                "terms. Stock eager GroupNorm folds the same way for HxW>1 and shows the identical 1.7e-4 error on a "
                "constant 4x4 group; eager avoids it here only through its HxW==1 special case. Inductor computes "
                "(x-mean)*rstd. Invisible at fp16 (the encoder's default tier)")
    if k in ("qk_norm_rope", "swiglu"):
        return ("by design: rounds each product/activation to x.dtype exactly like eager's separate kernels (bits!=eager ~0); "
                "Inductor by default keeps them in fp32, so it is MORE accurate than the stock reference. With "
                "emulate_precision_casts=True Inductor's error equals ours (see extras)")
    return None


# ── report ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def build_report(out_dir):
    parts_dir = os.path.join(out_dir, "parts")
    parts = []
    for fn in sorted(os.listdir(parts_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(parts_dir, fn)) as f:
                parts.append(json.load(f))
    rows, meta = [], {}
    for p in parts:
        rows += p["rows"]
        meta[p["kernel"]] = p["meta"]
    order = {k: i for i, k in enumerate(ALL_KERNELS)}
    rows.sort(key = lambda r: order.get(r["kernel"], 99))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent = 1, default = str)

    L = ["# Hand-written Triton vs torch.compile (Inductor)\n"]
    any_meta = next(iter(meta.values()), {})
    L.append(f"torch {any_meta.get('torch')}, triton {any_meta.get('triton')}, GPU {any_meta.get('gpu')}\n")
    for k, m in meta.items():
        L.append(f"- `{k}`: tree `{m.get('tree')}` @ `{m.get('head')}`")
    L.append("\nErrors are against a float64 run of the stock function on the same inputs. `ratio` = ours max abs / "
             "Inductor max abs. `bits!=ind` / `bits!=eager` = elements whose bit patterns differ. ms = CUDA-event "
             "median of the timed iterations after warmup.\n")
    findings = []
    qr = [r for r in rows if r["kernel"] == "quant_rows" and "q_mismatch_vs_fp64" in r]
    tie = [r for r in qr if r.get("tie_row_ours") is not None and r["tie_row_ours"] != r.get("tie_row_torch_round")]
    worse = [r for r in qr if r["q_mismatch_vs_fp64"] > max(r["q_mismatch_eager_vs_fp64"], r["q_mismatch_inductor_vs_fp64"])]
    if tie or worse:
        ex = next((r for r in qr if r["case"].startswith("tile256") and r["dtype"] == "fp16"), qr[0])
        findings.append(
            "- `_quant_rows` (video_minimax_h3_vae.py, `q = x / s` and `tl.floor(q + 0.5)` / `tl.ceil(q - 0.5)`) does "
            "not reproduce the module's own torch path `(xf / xs).round()`: it rounds exact ties away from zero "
            "(torch.round is half-to-even) and uses Triton's default approximate fp32 division. Real shape fp16: "
            f"{ex['q_mismatch_vs_fp64']} int8 codes differ from the float64 quantiser vs eager "
            f"{ex['q_mismatch_eager_vs_fp64']} / Inductor {ex['q_mismatch_inductor_vs_fp64']}; tie row (s == 1, "
            "x = k + 0.5): half of the ties differ by one code. Reconstruction error is unchanged (max |q*s - x| "
            "equal), so it is a +-1-code consistency issue, not an accuracy one. Fix: `q = libdevice.div_rn(x, s)` "
            "and `q = libdevice.rint(q)`, measured bit-identical to the torch path (0 / 14.7M codes, fp16/bf16/fp32) "
            "by scripts/triton_vs_inductor/quant_rounding_probe.py.")
    if findings:
        L.append("\n## Findings\n\n" + "\n".join(findings) + "\n")
    for k in ALL_KERNELS:
        kr = [r for r in rows if r["kernel"] == k]
        if not kr:
            continue
        L.append(f"\n## {k}  (Triton: {', '.join(TRITON_KERNELS[k])})\n")
        names = sorted({tuple(r.get("inductor_kernels") or ()) for r in kr if r.get("inductor_kernels")}, key = len)
        if names:
            L.append("Inductor kernels generated (distinct sets): " + "; ".join(
                f"{len(n)}: `{', '.join(n)}`" for n in names[:4]) + "\n")
        L.append("| case | shape class | dtype | ours max abs | ind max abs | eager max abs | ours mean | ind mean | "
                 "ratio | bits!=ind | bits!=eager | ours ms | ind ms | eager ms | launches ours/ind | verdict |")
        L.append("|" + "---|" * 16)
        for r in kr:
            o, i, e = r.get("ours") or {}, r.get("inductor") or {}, r.get("eager") or {}
            L.append(
                f"| {r['case']} | {r.get('shape_class', '')} | {r.get('dtype', '')} | {_f(o.get('max_abs'))} | "
                f"{_f(i.get('max_abs'))} | {_f(e.get('max_abs'))} | {_f(o.get('mean_abs'))} | {_f(i.get('mean_abs'))} | "
                f"{_f(r.get('ratio_max'))} | {_f(r.get('bits_vs_inductor'))} | {_f(r.get('bits_vs_eager'))} | "
                f"{_f(r.get('ours_ms'))} | {_f(r.get('inductor_ms'))} | {_f(r.get('eager_ms'))} | "
                f"{_f(r.get('ours_launches'))}/{len(r.get('inductor_kernels') or [])} | {r['verdict']} |"
            )
        notes = [f"- `{r['case']}` {r.get('dtype', '')} ({r['verdict'].split('(')[0]}): {explain(r)}" for r in kr
                 if explain(r)]
        if notes:
            L.append("\nNon-PASS rows:\n\n" + "\n".join(notes) + "\n")
        extras = []
        for r in kr:
            ex = {kk: r[kk] for kk in ("front_frames_exact_zero", "border_max_abs", "border_max_abs_inductor",
                                       "ours_vs_inductor_full", "inductor_emulate", "bits_vs_inductor_emulate",
                                       "bits_inductor_emulate_vs_eager", "ours_h", "inductor_h", "bits_vs_inductor_h",
                                       "bits_vs_eager_h", "q_mismatch_vs_fp64", "q_mismatch_inductor_vs_fp64",
                                       "q_mismatch_eager_vs_fp64", "tie_row_ours", "tie_row_torch_round",
                                       "near_half_ours_vs_torch", "zero_row_ok", "mode", "kernel_launched",
                                       "ours_mean", "inductor_mean", "out_strides_match_inductor", "error")
                  if kk in r}
            if ex:
                extras.append(f"- `{r['case']}` {r.get('dtype', '')}: " + json.dumps(ex, default = str))
        if extras:
            L.append("\n<details><summary>extra checks</summary>\n\n" + "\n".join(extras) + "\n\n</details>\n")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", os.path.join(out_dir, "results.json"), "and report.md")


# ── main ───────────────────────────────────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default = os.path.join(WS, "wt_verify_h3"), help = "worktree holding " + H3_REL)
    ap.add_argument("--bias-tree", default = None, help = "worktree holding " + BIAS_REL + " (default: --tree, else "
                    "wt_verify_bias)")
    ap.add_argument("--kernels", default = "all", help = "comma list of " + ",".join(ALL_KERNELS))
    ap.add_argument("--gpu", default = "auto")
    ap.add_argument("--iters", type = int, default = 50)
    ap.add_argument("--big", action = "store_true", help = "add the numel > 2^31 cases")
    ap.add_argument("--quick", action = "store_true", help = "a few cases per kernel (smoke)")
    ap.add_argument("--out-dir", default = os.path.join(WS, "outputs", "triton_vs_inductor"))
    ap.add_argument("--log-dir", default = os.path.join(WS, "logs", "triton_vs_inductor"))
    ap.add_argument("--report-only", action = "store_true")
    args = ap.parse_args()

    if args.report_only:
        build_report(args.out_dir)
        return
    gpu = pick_gpu(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"using physical GPU {gpu}", flush = True)
    os.makedirs(os.path.join(args.out_dir, "parts"), exist_ok = True)
    os.makedirs(args.log_dir, exist_ok = True)

    import torch
    import triton

    try:
        # this venv's bitsandbytes cannot initialise; Diffusers only needs to believe it is absent
        import diffusers.utils.import_utils as _iu

        _iu._bitsandbytes_available = False
    except Exception:  # noqa: BLE001
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kernels = ALL_KERNELS if args.kernels == "all" else [k.strip() for k in args.kernels.split(",")]
    h3 = bias_mod = None
    h3_path = os.path.join(args.tree, H3_REL)
    if any(k != "nvfp4_bias_add" for k in kernels):
        h3 = load_module(h3_path, "h3vae_under_test")
        assert h3._kernels() is not None, "Triton kernels unavailable"
    if "nvfp4_bias_add" in kernels:
        bt = args.bias_tree
        if bt is None:
            bt = args.tree if os.path.exists(os.path.join(args.tree, BIAS_REL)) else os.path.join(WS, "wt_verify_bias")
        bias_mod = load_module(os.path.join(bt, BIAS_REL), "nvfp4_bias_under_test")
    hz = Harness(args, h3, bias_mod)
    dispatch = {
        "gn_stats": lambda: hz.k_gn_silu_pad(stats_only = True),
        "gn_silu_pad": lambda: hz.k_gn_silu_pad(),
        "add_residual": hz.k_add_residual,
        "add_rmsnorm": hz.k_add_rmsnorm,
        "qk_norm_rope": hz.k_qk_norm_rope,
        "swiglu": hz.k_swiglu,
        "quant_rows": lambda: hz.k_int8("quant_rows"),
        "dequant_epilogue": lambda: hz.k_int8("dequant_epilogue"),
        "nvfp4_bias_add": hz.k_bias,
    }
    for k in kernels:
        t0 = time.time()
        start = len(hz.rows)
        dispatch[k]()
        tree = args.tree if k != "nvfp4_bias_add" else os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(bias_mod.__file__)))))
        meta = {"tree": tree, "head": git_head(tree), "torch": torch.__version__, "triton": triton.__version__,
                "gpu": torch.cuda.get_device_name(0), "physical_gpu": gpu, "big": args.big, "quick": args.quick,
                "iters": args.iters, "seconds": round(time.time() - t0, 1)}
        # the representative generated code per kernel: the first real-shape case
        kdir = os.path.join(args.log_dir, k)
        if os.path.isdir(kdir):
            files = sorted(f for f in os.listdir(kdir) if f.endswith(".py") and "emulate" not in f)
            if files:
                first = next((r for r in hz.rows[start:] if r.get("inductor_kernels")), None)
                pick = f"{first['case']}__{first['dtype']}.py" if first else files[0]
                src = os.path.join(kdir, pick if pick in files else files[0])
                with open(src) as f_in, open(os.path.join(args.log_dir, f"{k}.py"), "w") as f_out:
                    f_out.write(f"# representative: {os.path.basename(src)}; every case is under {kdir}/\n")
                    f_out.write(f_in.read())
        with open(os.path.join(args.out_dir, "parts", f"{k}.json"), "w") as f:
            json.dump({"kernel": k, "meta": meta, "rows": hz.rows[start:]}, f, indent = 1, default = str)
    build_report(args.out_dir)


if __name__ == "__main__":
    main()
