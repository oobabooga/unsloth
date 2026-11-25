#!/usr/bin/env python3
"""Item 7: AITER Triton Conv3D (aiter #5952/#8215 + gfx1151 configs #6973) vs MIOpen for the Wan 2.2 TI2V-5B VAE.

  aiter_conv3d.py --aiter-dir <A12_WORK>/aiter --mode micro --arm miopen|aiter --out F
  aiter_conv3d.py --aiter-dir <A12_WORK>/aiter --mode decode --arm miopen|aiter --vae-dir D --out F [--save V.pt]

micro   every distinct decoder Conv3d call of a 704x1280x49-frame decode (latent [1, 48, 13, 44, 80]), as diffusers'
        WanCausalConv3d issues it: input already causally padded (cache frames + F.pad), conv padding 0, bf16, random
        tensors (fixed seed). Per shape: median ms (CUDA events), and max abs / rel L2 vs an fp32 F.conv3d on the same
        inputs, plus an output sha. ``calls`` = how often a decode issues that shape (weights for an estimated total).
decode  full AutoencoderKLWan.decode of a fixed-seed latent at 704x1280x49 (1 warm + --reps timed), every
        WanCausalConv3d routed to aiter's conv3d (aiter arm) or stock (miopen arm); engagement counted (aiter calls,
        fallbacks with their errors). Saves the video fp16 for PSNR between arms.
The aiter Triton package is imported through stub ``aiter`` / ``aiter.ops`` / ``aiter.ops.triton`` packages, so no
HIP build or aiter install is involved. Writes one JSON; exits 0."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time

# (weight [Cout, Cin, kt, kh, kw], padded input [1, Cin, T, H, W], calls per 704x1280x49 decode); captured from
# diffusers 0.40 AutoencoderKLWan(Wan-AI/Wan2.2-TI2V-5B-Diffusers vae/config.json) on the meta device
SHAPES = [
    ((1024, 1024, 3, 3, 3), (1, 1024, 3, 46, 82), 130), ((1024, 1024, 3, 3, 3), (1, 1024, 4, 90, 162), 72),
    ((512, 512, 3, 3, 3), (1, 512, 6, 178, 322), 60), ((256, 256, 3, 3, 3), (1, 256, 6, 354, 642), 60),
    ((1024, 48, 3, 3, 3), (1, 48, 3, 46, 82), 13), ((2048, 1024, 3, 1, 1), (1, 1024, 3, 44, 80), 12),
    ((2048, 1024, 3, 1, 1), (1, 1024, 4, 88, 160), 12), ((512, 1024, 1, 1, 1), (1, 1024, 4, 176, 320), 12),
    ((512, 1024, 3, 3, 3), (1, 1024, 6, 178, 322), 12), ((256, 512, 1, 1, 1), (1, 512, 4, 352, 640), 12),
    ((256, 512, 3, 3, 3), (1, 512, 6, 354, 642), 12), ((12, 256, 3, 3, 3), (1, 256, 6, 354, 642), 12),
    ((1024, 1024, 3, 3, 3), (1, 1024, 3, 90, 162), 6), ((512, 512, 3, 3, 3), (1, 512, 3, 178, 322), 5),
    ((256, 256, 3, 3, 3), (1, 256, 3, 354, 642), 5), ((48, 48, 1, 1, 1), (1, 48, 13, 44, 80), 1),
    ((512, 1024, 1, 1, 1), (1, 1024, 1, 176, 320), 1), ((512, 1024, 3, 3, 3), (1, 1024, 3, 178, 322), 1),
    ((256, 512, 1, 1, 1), (1, 512, 1, 352, 640), 1), ((256, 512, 3, 3, 3), (1, 512, 3, 354, 642), 1),
    ((12, 256, 3, 3, 3), (1, 256, 3, 354, 642), 1),
]
LATENT = (1, 48, 13, 44, 80)  # 1280x704, 49 frames


def import_conv3d(aiter_dir: str, cpu_interp: bool = False):
    import importlib.machinery
    import importlib.util
    root = os.path.abspath(aiter_dir)
    for name, sub in (("aiter", "aiter"), ("aiter.ops", "aiter/ops"), ("aiter.ops.triton", "aiter/ops/triton")):
        spec = importlib.machinery.ModuleSpec(name, None, is_package = True)
        spec.submodule_search_locations = [os.path.join(root, sub)]
        sys.modules[name] = importlib.util.module_from_spec(spec)
    from aiter.ops.triton.conv import conv3d as C
    if cpu_interp:  # local TRITON_INTERPRET=1 smoke only: pretend to be a gfx1151 wave32 device
        from aiter.ops.triton.conv import _launch, _prepack, _utils
        from aiter.ops.triton.utils._triton import arch_info
        import contextlib

        import torch
        arch_info._CACHED_ARCH = "gfx1151"
        torch.cuda.device = lambda d: contextlib.nullcontext()
        for m in (C, _utils, _launch, _prepack):
            if hasattr(m, "_is_amd_wave32"):
                m._is_amd_wave32 = lambda: True
            if hasattr(m, "_validate_conv_devices"):
                m._validate_conv_devices = lambda *a, **k: None
    return C


def sha(t) -> str:
    import torch
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def micro(a, obs, dev):
    import torch
    import torch.nn.functional as F
    C = import_conv3d(a.aiter_dir, a.cpu_interp) if a.arm == "aiter" else None
    shapes = SHAPES if not a.tiny else [((32, 16, 3, 3, 3), (1, 16, 3, 10, 12), 1), ((16, 32, 1, 1, 1), (1, 32, 2, 8, 8), 1)]
    rows = []
    for i, (ws, xs, calls) in enumerate(shapes):
        g = torch.Generator(device = dev).manual_seed(100 + i)
        x = torch.randn(xs, generator = g, device = dev).to(torch.bfloat16)
        w = (torch.randn(ws, generator = g, device = dev) * (1.0 / (ws[1] * ws[2] * ws[3] * ws[4]) ** 0.5)).to(torch.bfloat16)
        b = (torch.randn(ws[0], generator = g, device = dev) * 0.1).to(torch.bfloat16)
        row = {"w": list(ws), "x": list(xs), "calls": calls}
        if C is not None:
            row["route"] = C._resolve_route(ws[2], ws[3], ws[4], (1, 1, 1), (1, 1, 1), xs[0], xs[1], xs[2], xs[3],
                                            xs[4], ws[0], "ncdhw", (0, 0, 0)).name

        def fn():
            return C.conv3d(x, w, b) if C is not None else F.conv3d(x, w, b)
        try:
            with torch.no_grad():
                y = fn()
                for _ in range(a.warm):
                    fn()
                if dev == "cuda":
                    torch.cuda.synchronize()
                    ms = []
                    for _ in range(a.reps):
                        e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                        e0.record()
                        fn()
                        e1.record()
                        e1.synchronize()
                        ms.append(e0.elapsed_time(e1))
                    row["ms"], row["med_ms"] = ms, st.median(ms)
                ref = F.conv3d(x.float(), w.float(), b.float())
                d = (y.float() - ref)
                row["max_abs"] = float(d.abs().max())
                row["rel_l2"] = float(torch.linalg.vector_norm(d) / torch.linalg.vector_norm(ref))
                row["out_sha"] = sha(y)
                row["shape_ok"] = list(y.shape) == list(ref.shape)
                del ref, d, y
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)[:300]
        rows.append(row)
        obs["rows"] = rows
        del x, w, b
        if dev == "cuda":
            torch.cuda.empty_cache()
    ok = [r for r in rows if r.get("med_ms")]
    obs["est_decode_conv_ms"] = sum(r["med_ms"] * r["calls"] for r in ok) if len(ok) == len(rows) else None


def decode(a, obs):
    import torch
    import diffusers
    from diffusers.models.autoencoders import autoencoder_kl_wan as W
    obs["diffusers"] = diffusers.__version__
    eng = {"aiter": 0, "fallback": 0, "errors": []}
    if a.arm == "aiter":
        import torch.nn.functional as F
        C = import_conv3d(a.aiter_dir)
        orig = W.WanCausalConv3d.forward

        def fwd(self, x, cache_x = None):
            padding = list(self._padding)
            if cache_x is not None and self._padding[4] > 0:
                x = torch.cat([cache_x.to(x.device), x], dim = 2)
                padding[4] -= cache_x.shape[2]
            x = F.pad(x, padding)
            if x.dtype in (torch.bfloat16, torch.float16) and self.groups == 1 and self.padding_mode == "zeros":
                try:
                    y = C.conv3d(x, self.weight, self.bias, stride = self.stride, padding = self.padding,
                                 dilation = self.dilation)
                    eng["aiter"] += 1
                    return y
                except Exception as exc:  # noqa: BLE001 -- counted: a silent fallback would void the arm
                    if len(eng["errors"]) < 5:
                        eng["errors"].append(repr(exc)[:200])
            eng["fallback"] += 1
            return torch.nn.Conv3d.forward(self, x)
        W.WanCausalConv3d.forward = fwd
        obs["patched_from"] = getattr(orig, "__qualname__", str(orig))
    vae = diffusers.AutoencoderKLWan.from_pretrained(a.vae_dir, subfolder = "vae", torch_dtype = torch.bfloat16)
    vae = vae.to("cuda").eval()
    g = torch.Generator(device = "cuda").manual_seed(7)
    z = torch.randn(LATENT, generator = g, device = "cuda", dtype = torch.float32).to(torch.bfloat16)
    obs["latent"] = list(LATENT)
    with torch.no_grad():
        t = time.perf_counter()
        vae.decode(z).sample
        torch.cuda.synchronize()
        obs["first_s"] = time.perf_counter() - t
        obs["engagement_first"] = dict(eng, errors = list(eng["errors"]))
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            y = vae.decode(z).sample
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    obs["s"], obs["med_s"] = ts, st.median(ts)
    obs["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    obs["engagement"] = eng
    obs["out_shape"] = list(y.shape)
    if a.save:
        torch.save(y.float().clamp(-1, 1).to(torch.float16).contiguous().cpu(), a.save)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aiter-dir", required = True)
    ap.add_argument("--mode", required = True, choices = ("micro", "decode"))
    ap.add_argument("--arm", required = True, choices = ("miopen", "aiter"))
    ap.add_argument("--out", required = True)
    ap.add_argument("--vae-dir")
    ap.add_argument("--save")
    ap.add_argument("--reps", type = int, default = None)
    ap.add_argument("--warm", type = int, default = 2)
    ap.add_argument("--tiny", action = "store_true", help = "two tiny shapes (local smoke)")
    ap.add_argument("--cpu-interp", action = "store_true", help = "TRITON_INTERPRET=1 on CPU (local smoke)")
    a = ap.parse_args()
    if a.reps is None:
        a.reps = 5 if a.mode == "micro" else 2
    obs: dict = {"mode": a.mode, "arm": a.arm}
    t0 = time.time()
    try:
        import torch
        import triton
        dev = "cpu" if a.cpu_interp else "cuda"
        obs.update(torch = torch.__version__, hip = torch.version.hip, triton = triton.__version__,
                   arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if dev == "cuda" else None,
                   miopen_find_mode = os.environ.get("MIOPEN_FIND_MODE"))
        if a.mode == "micro":
            micro(a, obs, dev)
        else:
            decode(a, obs)
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-2000:]
    obs["seconds"] = round(time.time() - t0, 1)
    open(a.out, "w", encoding = "utf-8").write(json.dumps(obs, default = str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
