#!/usr/bin/env python3
"""One fresh process of the PR 19873 measurement (Wan 2.2 TI2V-5B VAE decode, bf16, gfx1151).

  worker.py --backend <tree>/studio/backend --vae-dir D --mode smoke|warm|cold|acc --out J [--frames 81] [--save P]

The arm is chosen by the tree (--backend) plus the environment the orchestrator sets; this file never sets the
router env. Studio's router (core.inference.diffusion_rocm_triton_conv3d.install) is installed on the real VAE when
the tree has it (absent at the merge base: routed is 0 by construction there).

smoke  latent (1,48,3,16,16): two decodes; wrapped count, router COUNTS, self-check state, routes.
warm   for 480x832 then 704x1280 (--frames): one warm-up decode, then one timed decode with the peak-memory counter
       reset; wall, max_memory_allocated / reserved, COUNTS, total Conv3d calls, route histogram.
cold   TRITON_CACHE_DIR must be empty: decode 480p then 720p once each; per decode wall and the compiled variants
       per Triton kernel (sum of len(device_caches[dev][0])) plus .hsaco files in the cache dir.
acc    no router: every distinct (weight shape, input shape, stride, padding) Conv3d call of a 480p and a 720p
       decode (latent T=3, which already yields every chunk shape) is recomputed as fp32 F.conv3d (reference),
       bf16 F.conv3d (MIOpen), the vendored conv3d with ALLOW_WINOGRAD False (direct) and True (Winograd);
       rel L2 vs the reference for each, on the real layer weights and real activations.
JSON to --out only. Never reads or passes a token."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

RES = {"480": (480, 832), "720": (704, 1280)}
if os.environ.get("C3D_TEST_SMALL") == "1":  # local plumbing test only
    RES = {"480": (64, 96), "720": (96, 128)}
PKG = "core.inference.rocm_triton_conv3d"
DEV = os.environ.get("C3D_DEVICE", "cuda")  # "cpu" only for the local plumbing test


def sync(torch):
    if DEV == "cuda":
        torch.cuda.synchronize()


def latent(h, w, frames):
    return (1, 48, (frames - 1) // 4 + 1, h // 16, w // 16)


def variants():
    try:
        from triton.runtime.jit import JITFunction
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)[:200]}
    out = {}
    for name, m in list(sys.modules.items()):
        if not name.startswith(PKG) or m is None:
            continue
        for v in list(vars(m).values()):
            fn = v
            for _ in range(5):
                if isinstance(fn, JITFunction):
                    break
                fn = getattr(fn, "fn", None)
            if not isinstance(fn, JITFunction):
                continue
            n = 0
            for dc in (getattr(fn, "device_caches", None) or {}).values():
                c = dc[0] if isinstance(dc, (tuple, list)) else dc
                if isinstance(c, dict):
                    n += len(c)
            key = f"{fn.__module__.split('.')[-1]}.{fn.__name__}"
            out[key] = max(out.get(key, 0), n)
    return {k: v for k, v in sorted(out.items()) if v}


def cache_files():
    d = os.environ.get("TRITON_CACHE_DIR")
    if not d or not os.path.isdir(d):
        return {"dir": d, "hsaco": 0, "files": 0}
    n = h = 0
    for _root, _dirs, files in os.walk(d):
        for f in files:
            n += 1
            h += f.endswith(".hsaco")
    return {"dir": d, "hsaco": h, "files": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required = True)
    ap.add_argument("--vae-dir", required = True)
    ap.add_argument("--mode", required = True, choices = ("smoke", "warm", "cold", "acc"))
    ap.add_argument("--out", required = True)
    ap.add_argument("--frames", type = int, default = 81)
    ap.add_argument("--save", default = "")
    a = ap.parse_args()
    for k in [k for k in os.environ if "TOKEN" in k.upper()]:
        os.environ.pop(k, None)
    sys.path.insert(0, os.path.abspath(a.backend))
    obs: dict = {"mode": a.mode, "backend": os.path.abspath(a.backend), "frames": a.frames,
                 "env": {k: v for k, v in os.environ.items() if k.startswith("UNSLOTH_DIFFUSION") or k == "TRITON_CACHE_DIR"}}
    t0 = time.time()

    def save():
        obs["seconds"] = round(time.time() - t0, 1)
        with open(a.out, "w", encoding = "utf-8") as fh:
            fh.write(json.dumps(obs, default = str))
    try:
        run(a, obs, save)
    except Exception as exc:  # noqa: BLE001
        obs["error"] = repr(exc)[:400]
        obs["tb"] = traceback.format_exc()[-2500:]
    save()
    return 0


def run(a, obs, save):
    import logging

    import diffusers
    import torch
    import torch.nn.functional as F
    logging.basicConfig(level = logging.INFO, stream = sys.stderr)
    log = logging.getLogger("c3d_worker")
    obs.update(torch = torch.__version__, hip = torch.version.hip, diffusers = diffusers.__version__,
               arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if DEV == "cuda" else DEV)
    try:
        import triton
        obs["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        obs["triton"] = repr(exc)[:200]
    mi = getattr(torch.backends, "miopen", None)  # Studio's ROCm setting, identical in every arm
    if mi is not None and hasattr(mi, "immediate") and not os.environ.get("MIOPEN_FIND_MODE"):
        mi.immediate = True
    obs["miopen_immediate"] = bool(getattr(mi, "immediate", False))
    if a.mode == "cold":
        obs["cache_at_start"] = cache_files()
    t = time.time()
    vae = diffusers.AutoencoderKLWan.from_pretrained(a.vae_dir, subfolder = "vae", torch_dtype = torch.bfloat16)
    vae = vae.to(DEV).eval()
    obs["load_s"] = round(time.time() - t, 2)
    obs["param_dtypes"] = sorted({str(p.dtype) for p in vae.parameters()})
    calls = {"n": 0}

    def pre(_m, _args):
        calls["n"] += 1
    convs = [m for m in vae.modules() if isinstance(m, torch.nn.Conv3d)]
    for m in convs:
        m.register_forward_pre_hook(pre)
    obs["conv3d_modules"] = len(convs)
    try:
        from core.inference import diffusion_rocm_triton_conv3d as R
        obs["router_file"] = R.__file__
    except Exception as exc:  # noqa: BLE001 - the merge base has no router
        R = None
        obs["router_file"] = None
        obs["router_import"] = repr(exc)[:200]
    if R is not None and os.environ.get("C3D_TEST_SMALL") == "1":  # local plumbing test on CUDA only
        import core.inference.rocm_bf16 as RB
        RB.is_rocm_torch = lambda _t: True
        R._device_arch = lambda _t, _i: "gfx1151"
    routes: dict = {}
    if a.mode == "acc":
        return accuracy(a, obs, save, torch, F, vae, convs)
    obs["wrapped"] = R.install(vae, log) if R is not None else 0
    if R is not None and hasattr(R, "_route_name"):
        orig = R._route_name

        def counted(*args, **kw):
            r = orig(*args, **kw)
            routes[r.name] = routes.get(r.name, 0) + 1
            return r
        R._route_name = counted
    if R is not None and R._STATE.get("mod") is not None:
        obs["allow_winograd"] = getattr(R._STATE["mod"], "ALLOW_WINOGRAD", "absent (always on)")

    def decode(shape, peak = False):
        g = torch.Generator(device = DEV).manual_seed(7)
        z = torch.randn(shape, generator = g, device = DEV, dtype = torch.float32).to(torch.bfloat16)
        if R is not None:
            R.reset_counts()
        routes.clear()
        calls["n"] = 0
        sync(torch)
        if peak and DEV == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        with torch.no_grad():
            y = vae.decode(z).sample
        sync(torch)
        rec = {"latent": list(shape), "s": round(time.perf_counter() - t, 3),
               "counts": dict(R.COUNTS) if R is not None else {"routed": 0},
               "conv3d_calls": calls["n"], "routes": dict(routes), "out_shape": list(y.shape)}
        if peak and DEV == "cuda":
            rec["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            rec["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)
        rec["finite"] = bool(torch.isfinite(y).all())
        return rec, y

    def router_state():
        if R is None:
            return None
        return {str(k): v for k, v in R._STATE.items() if k != "mod"}

    if a.mode == "smoke":
        obs["decodes"] = []
        for _ in range(2):
            rec, _y = decode((1, 48, 3, 16, 16))
            obs["decodes"].append(rec)
        obs["router_state"] = router_state()
    elif a.mode == "warm":
        obs["res"] = {}
        for tag, (h, w) in RES.items():
            shape = latent(h, w, a.frames)
            first, _y = decode(shape)
            del _y
            rec, y = decode(shape, peak = True)
            rec["warmup_s"] = first["s"]
            obs["res"][tag] = rec
            if a.save and tag == "720":
                small = y[:, :, ::4, ::4, ::4].float().clamp(-1, 1).to(torch.float16).contiguous().cpu()
                torch.save(small, a.save)
            del y
            save()
        obs["router_state"] = router_state()
    elif a.mode == "cold":
        obs["variants_before"] = variants()
        obs["cold"] = {}
        for tag, (h, w) in RES.items():
            rec, y = decode(latent(h, w, a.frames))
            del y
            rec["variants_after"] = variants()
            rec["variants_total"] = sum(v for v in rec["variants_after"].values() if isinstance(v, int))
            rec["cache"] = cache_files()
            obs["cold"][tag] = rec
            save()
        obs["router_state"] = router_state()


def accuracy(a, obs, save, torch, F, vae, convs):
    from core.inference.rocm_triton_conv3d import conv3d as mod
    obs["kernel_file"] = mod.__file__
    seen: set = set()
    rows: list = []
    names = {m: n for n, m in vae.named_modules()}

    def rel(y, ref):
        return float(torch.linalg.vector_norm(y.float() - ref) / torch.linalg.vector_norm(ref))

    def check(m, x, w, b):
        st, pad, dil = tuple(m.stride), tuple(m.padding), tuple(m.dilation)
        key = (tuple(w.shape), tuple(x.shape), st, pad, dil)
        if key in seen or x.dtype != torch.bfloat16 or m.groups != 1 or m.padding_mode != "zeros":
            return
        seen.add(key)
        row = {"module": names.get(m, "?"), "w": list(w.shape), "x": list(x.shape), "stride": st, "padding": pad}
        try:
            with torch.no_grad():
                ref = F.conv3d(x.float(), w.float(), None if b is None else b.float(), st, pad, dil)
                row["miopen"] = rel(F.conv3d(x, w, b, st, pad, dil), ref)
                N, C, D, H, W = x.shape
                K, _c, T, Rr, S = w.shape
                for arm, allow in (("direct", False), ("winograd", True)):
                    mod.ALLOW_WINOGRAD = allow
                    row[f"route_{arm}"] = mod._resolve_route(T, Rr, S, st, dil, N, C, D, H, W, K, "ncdhw",
                                                             padding = pad).name
                    y = mod.conv3d(x, w, b, st, pad, dil)
                    row[arm] = rel(y, ref)
                    del y
                mod.ALLOW_WINOGRAD = False
                del ref
            for arm in ("direct", "winograd"):
                row[f"ratio_{arm}"] = row[arm] / row["miopen"] if row["miopen"] else None
        except Exception as exc:  # noqa: BLE001
            row["error"] = repr(exc)[:300]
        rows.append(row)
        if DEV == "cuda":
            torch.cuda.empty_cache()
    def wrap(m, orig):
        # at _conv_forward, the level Studio's router wraps: the input is already causally padded (WanCausalConv3d)
        def _conv_forward(input, weight, bias):
            check(m, input, weight, bias)
            return orig(input, weight, bias)
        return _conv_forward
    for m in convs:
        m._conv_forward = wrap(m, m._conv_forward)
    obs["rows"] = rows
    for tag, (h, w) in RES.items():
        g = torch.Generator(device = DEV).manual_seed(7)
        z = torch.randn(latent(h, w, 9), generator = g, device = DEV, dtype = torch.float32).to(torch.bfloat16)
        with torch.no_grad():
            vae.decode(z).sample
        sync(torch)
        obs.setdefault("decoded", []).append(tag)
        save()


if __name__ == "__main__":
    sys.exit(main())
