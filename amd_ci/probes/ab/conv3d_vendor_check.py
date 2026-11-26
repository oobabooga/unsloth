#!/usr/bin/env python3
"""GPU check of Studio's vendored ROCm Triton conv3d (core/inference/rocm_triton_conv3d + diffusion_rocm_triton_conv3d).

  conv3d_vendor_check.py --backend <tree>/studio/backend --mode micro --arm stock|triton --out J
  conv3d_vendor_check.py --backend <tree>/studio/backend --mode decode|ltxdecode --arm stock|triton --vae-dir D --out J
                         [--save T.pt] [--latent 1,48,13,44,80] [--tiling] [--reps 2]

micro      every distinct Wan 2.2 TI2V-5B decoder conv of a 704x1280x49 decode (input already causally padded, conv
           padding 0, bf16, fixed seed), the vendored conv3d (triton arm) or F.conv3d (stock arm): median ms (CUDA
           events), route, max abs / rel L2 vs fp32 F.conv3d, output sha; est_decode_conv_ms = sum(ms * calls).
decode     diffusers AutoencoderKLWan.decode of a fixed-seed latent (default 704x1280x49) with Studio's router
           installed (triton arm: UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D=1; stock arm: =0, install returns 0), 1 cold +
           --reps timed decodes; router COUNTS, total Conv3d calls (forward pre-hooks), coverage = routed / total,
           wall, torch.cuda.max_memory_allocated, the router's self-check state.
ltxdecode  the same for the LTX-2.5 VAE (AutoencoderKLLTX2Video, Lightricks/LTX-2.5-Diffusers vae/; default latent
           [1, 128, 7, 22, 40] = 704x1280x49).
Each arm is its own process. Writes JSON to --out only (stdout may carry import banners); never reads a token."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time

# (weight [Cout, Cin, kt, kh, kw], padded input [1, Cin, T, H, W], calls per 704x1280x49 decode); diffusers 0.40
# AutoencoderKLWan(Wan-AI/Wan2.2-TI2V-5B-Diffusers vae/config.json), captured on the meta device
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
LATENTS = {"decode": (1, 48, 13, 44, 80), "ltxdecode": (1, 128, 7, 22, 40)}
ENV = "UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D"


def sha(t) -> str:
    import torch
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def micro(a, obs):
    import torch
    import torch.nn.functional as F
    C = None
    if a.arm == "triton":
        from core.inference.rocm_triton_conv3d import conv3d as C
    rows = []
    for i, (ws, xs, calls) in enumerate(SHAPES):
        g = torch.Generator(device = "cuda").manual_seed(100 + i)
        x = torch.randn(xs, generator = g, device = "cuda").to(torch.bfloat16)
        w = (torch.randn(ws, generator = g, device = "cuda") * (1.0 / (ws[1] * ws[2] * ws[3] * ws[4]) ** 0.5)
             ).to(torch.bfloat16)
        b = (torch.randn(ws[0], generator = g, device = "cuda") * 0.1).to(torch.bfloat16)
        row = {"w": list(ws), "x": list(xs), "calls": calls}
        if C is not None:
            row["route"] = C._resolve_route(ws[2], ws[3], ws[4], (1, 1, 1), (1, 1, 1), xs[0], xs[1], xs[2], xs[3],
                                            xs[4], ws[0], "ncdhw", padding = (0, 0, 0)).name

        def fn():
            return C.conv3d(x, w, b, (1, 1, 1), (0, 0, 0), (1, 1, 1)) if C is not None else F.conv3d(x, w, b)
        try:
            with torch.no_grad():
                y = fn()
                for _ in range(a.warm):
                    fn()
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
                d = y.float() - ref
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
        torch.cuda.empty_cache()
    ok = [r for r in rows if r.get("med_ms")]
    obs["est_decode_conv_ms"] = sum(r["med_ms"] * r["calls"] for r in ok) if len(ok) == len(rows) else None


def decode(a, obs):
    import logging

    import diffusers
    import torch
    from core.inference import diffusion_rocm_triton_conv3d as R
    obs["diffusers"] = diffusers.__version__
    obs["router_file"] = R.__file__
    cls = diffusers.AutoencoderKLWan if a.mode == "decode" else diffusers.AutoencoderKLLTX2Video
    # Studio's own ROCm setting (diffusion_speed._enable_miopen_immediate): no Find trial launches
    mi = getattr(torch.backends, "miopen", None)
    if mi is not None and hasattr(mi, "immediate") and not os.environ.get("MIOPEN_FIND_MODE"):
        mi.immediate = True
    obs["miopen_immediate"] = bool(getattr(mi, "immediate", False))
    vae = cls.from_pretrained(a.vae_dir, subfolder = "vae", torch_dtype = torch.bfloat16).to("cuda").eval()
    if a.tiling and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    obs["vae_class"] = type(vae).__name__
    total = {"conv3d_calls": 0}

    def pre(_m, _args):
        total["conv3d_calls"] += 1
    n_conv = 0
    for m in vae.modules():
        if isinstance(m, torch.nn.Conv3d):
            m.register_forward_pre_hook(pre)
            n_conv += 1
    obs["conv3d_modules"] = n_conv
    log = logging.getLogger("conv3d_vendor_check")
    logging.basicConfig(level = logging.INFO, stream = sys.stderr)
    obs["wrapped"] = R.install(vae, log)
    latent = tuple(int(v) for v in a.latent.split(",")) if a.latent else LATENTS[a.mode]
    obs["latent"] = list(latent)
    g = torch.Generator(device = "cuda").manual_seed(7)
    z = torch.randn(latent, generator = g, device = "cuda", dtype = torch.float32).to(torch.bfloat16)
    with torch.no_grad():
        R.reset_counts()
        total["conv3d_calls"] = 0
        torch.cuda.synchronize()
        t = time.perf_counter()
        vae.decode(z).sample
        torch.cuda.synchronize()
        obs["first_s"] = time.perf_counter() - t
        obs["counts_first"] = dict(R.COUNTS, conv3d_calls = total["conv3d_calls"])
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(a.reps):
            R.reset_counts()
            total["conv3d_calls"] = 0
            torch.cuda.synchronize()
            t = time.perf_counter()
            y = vae.decode(z).sample
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    obs["s"], obs["med_s"] = ts, st.median(ts)
    obs["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    obs["counts"] = dict(R.COUNTS, conv3d_calls = total["conv3d_calls"])
    obs["coverage"] = R.COUNTS["routed"] / total["conv3d_calls"] if total["conv3d_calls"] else None
    obs["router_state"] = {k: v for k, v in R._STATE.items() if k != "mod"}
    obs["out_shape"] = list(y.shape)
    obs["out_sha"] = sha(y)
    if a.save:
        torch.save(y.float().clamp(-1, 1).to(torch.float16).contiguous().cpu(), a.save)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--mode", required = True, choices = ("micro", "decode", "ltxdecode"))
    ap.add_argument("--arm", required = True, choices = ("stock", "triton"))
    ap.add_argument("--out", required = True)
    ap.add_argument("--vae-dir")
    ap.add_argument("--save")
    ap.add_argument("--latent")
    ap.add_argument("--tiling", action = "store_true")
    ap.add_argument("--reps", type = int, default = None)
    ap.add_argument("--warm", type = int, default = 2)
    a = ap.parse_args()
    if a.reps is None:
        a.reps = 5 if a.mode == "micro" else 2
    for k in [k for k in os.environ if "TOKEN" in k]:
        os.environ.pop(k, None)
    os.environ[ENV] = "1" if a.arm == "triton" else "0"
    sys.path.insert(0, os.path.abspath(a.backend))
    obs: dict = {"mode": a.mode, "arm": a.arm, "backend": os.path.abspath(a.backend)}
    t0 = time.time()
    try:
        import torch
        import triton
        obs.update(torch = torch.__version__, hip = torch.version.hip, triton = triton.__version__,
                   arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
                   platform = sys.platform, miopen_find_mode = os.environ.get("MIOPEN_FIND_MODE"))
        if a.mode == "micro":
            micro(a, obs)
        else:
            if not a.vae_dir:
                raise SystemExit("--vae-dir is required for decode modes")
            decode(a, obs)
    except Exception as exc:  # noqa: BLE001
        import traceback
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-2000:]
    obs["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding = "utf-8") as fh:
        fh.write(json.dumps(obs, default = str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
