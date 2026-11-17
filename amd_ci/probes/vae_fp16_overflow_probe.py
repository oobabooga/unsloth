#!/usr/bin/env python3
"""Probe: unsloth #18016 / #18077, Studio's ROCm VAE attention on the REAL Z-Image-Turbo (FLUX.1) VAE, one checkout.

Observes only (judged by criteria/vae_fp16_overflow.py). Uses the checkout's own
``core.inference.diffusion_vae_attn_chunked`` (C), plus the PR #18077 as-submitted module (shipped next to this file,
loaded into the same package) as the extra arm ``pr_orig``.

  fp16 decode   the real VAE in fp16 (the dtype Studio picks on ROCm cards without native bf16, gfx103x), Studio's
                attention patch installed (``C.install``), decoding real latents (a Z-Image render encoded by the same
                VAE) at 1024 and 512 px: NaN count, half-math engagement, numerics vs an fp32 reference; arms default
                and kill (UNSLOTH_VAE_ATTN_HALF_MATH=0, the reporter's workaround).
  bf16 decode   the same VAE in bf16 (gfx11 / gfx1151 default), 1024 px, timed: half (default), half_aa (A/A), kill,
                pr_orig (the #18077 as-submitted module, default env). PSNR / LPIPS vs the fp32 reference.
  bf16 attn     q / k / v [1, 1, 16384, 512] through the mode: half, half_aa, kill, pr_orig (timing only).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
HERE = Path(__file__).resolve().parent


def _write(path: Path, obs: dict) -> None:
    path.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")


def _sha(t) -> str:
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--reps", type = int, default = 7)
    ap.add_argument("--warmup", type = int, default = 2)
    ap.add_argument("--vae-dir", default = "")
    ap.add_argument("--device", default = "cuda:0", help = "cpu: harness smoke test only (no timing evidence)")
    args = ap.parse_args()

    t0 = time.time()
    obs: dict = {"state": args.state, "fp16": [], "bf16": {}, "attn": {}}
    sys.path.insert(0, str(args.checkout / "studio" / "backend"))
    try:
        import torch
        import torch.nn.functional as F

        if args.device != "cpu":
            props = torch.cuda.get_device_properties(0)
            obs["arch"] = (getattr(props, "gcnArchName", "") or "").split(":")[0]
            obs["device_name"] = torch.cuda.get_device_name(0)
        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        obs["platform"] = sys.platform
        from core.inference import diffusion_vae_attn_chunked as C

        obs["module_file"] = C.__file__
        obs["module_sha256"] = hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location(
            "core.inference.pr18077_orig_vae_attn", HERE / "pr18077_orig_vae_attn_chunked.py"
        )
        P = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = P
        spec.loader.exec_module(P)
        obs["pr_orig_has_linux_branch"] = hasattr(P, "_non_windows_rocm_untrusted_head")
        import diffusers
        from diffusers import AutoencoderKL

        obs["diffusers"] = diffusers.__version__
        vae_dir = args.vae_dir
        if not vae_dir:
            from huggingface_hub import snapshot_download

            root = os.environ.get("RUNNER_TEMP") or str(HERE)
            vae_dir = str(Path(snapshot_download(
                "Tongyi-MAI/Z-Image-Turbo", allow_patterns = ["vae/*"], local_dir = str(Path(root) / "zimage_vae"),
            )) / "vae")
        lats = torch.load(HERE / "zimage_latents.pt")
    except Exception as e:  # noqa: BLE001
        obs["setup_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        _write(args.out, obs)
        return 0

    dev = torch.device(args.device)
    if dev.type == "cpu":  # smoke test: no-op CUDA calls, wall-clock events, a fake ROCm math-only GPU
        class _Ev:
            def __init__(self, **_):
                self.t = 0.0

            def record(self):
                self.t = time.perf_counter()

            def elapsed_time(self, other):
                return (other.t - self.t) * 1e3

        torch.cuda.synchronize = lambda *a: None
        torch.cuda.reset_peak_memory_stats = lambda *a: None
        torch.cuda.memory_allocated = lambda *a: 0
        torch.cuda.max_memory_allocated = lambda *a: 0
        torch.cuda.empty_cache = lambda: None
        torch.cuda.Event = _Ev
        torch.version.hip = "smoke"
        for M in (C, P):
            M.fused_sdpa_available = lambda q, k, v: False
            real = M._half_math_call
            M._half_math_call = (lambda r: lambda q: r(type("Q", (), {"device": type("D", (), {"type": "cuda"})(),
                                                                       "dtype": q.dtype})()))(real)
        lats = {"512": lats["512"], "1024": lats["512"]}  # smoke: small decode only
    env = C.VAE_ATTN_HALF_MATH_ENV
    target = type("T", (), {"backend": "rocm", "device": "cuda"})()

    # engagement counters of every mode the patched forwards create
    modes: list = []
    for M in (C, P):
        real_mode = M.chunked_attention_mode

        def counting(*a, _real = real_mode, **kw):
            m = _real(*a, **kw)
            modes.append(m)
            return m

        M.chunked_attention_mode = counting

    def drain() -> dict:
        ms = list(modes)
        modes.clear()
        return {"half": sum(m.half for m in ms), "chunked": sum(m.chunked for m in ms), "calls": len(ms)}

    def run_arm(name: str, fn):
        if name == "kill":
            os.environ[env] = "0"
        else:
            os.environ.pop(env, None)
        try:
            return fn()
        finally:
            os.environ.pop(env, None)

    def timed(arms: dict, reps: int, warmup: int) -> dict:
        res = {n: {"times_ms": [], "peak_mib": 0.0} for n in arms}
        for _ in range(warmup):
            for n, fn in arms.items():
                run_arm(n, fn)
        torch.cuda.synchronize()
        names = list(arms)
        for rep in range(reps):
            order = names[rep % len(names):] + names[:rep % len(names)]
            for n in order:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base_alloc = torch.cuda.memory_allocated()
                s, e = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                s.record()
                out = run_arm(n, arms[n])
                e.record()
                torch.cuda.synchronize()
                res[n]["times_ms"].append(round(s.elapsed_time(e), 4))
                res[n]["peak_mib"] = max(res[n]["peak_mib"],
                                         round((torch.cuda.max_memory_allocated() - base_alloc) / 2**20, 1))
                del out
        for n in res:
            ts = res[n]["times_ms"]
            res[n]["median_ms"] = round(statistics.median(ts), 4)
            res[n]["spread_rel"] = round((max(ts) - min(ts)) / statistics.median(ts), 4)
        return res

    img = lambda t: (t.float() / 2 + 0.5).clamp(0, 1)  # noqa: E731
    lp: dict = {}

    def lpips_of(a, b):
        if "net" not in lp:
            try:
                import lpips

                lp["net"] = lpips.LPIPS(net = "alex", verbose = False).to(dev).eval()
            except Exception as e:  # noqa: BLE001
                lp["net"] = None
                obs["lpips_error"] = f"{type(e).__name__}: {e}"[:300]
        if lp["net"] is None or not bool(torch.isfinite(a).all()):
            return None
        with torch.no_grad():
            return round(float(lp["net"](img(a) * 2 - 1, img(b) * 2 - 1).item()), 6)

    def numerics(out, ref) -> dict:
        d = (out.float() - ref.float()).abs()
        finite = bool(torch.isfinite(out).all())
        r = {"nan": int(torch.isnan(out).sum()), "finite": finite,
             "max_abs": float(d.max()) if finite else None}
        if finite:
            mse = float(((img(out) - img(ref)) ** 2).mean())
            r["psnr_db"] = round(10 * math.log10(1.0 / mse), 2) if mse > 0 else float("inf")
            r["lpips"] = lpips_of(out, ref)
        return r

    try:
        vae32 = AutoencoderKL.from_pretrained(vae_dir, torch_dtype = torch.float32).eval().to(dev)
        refs = {}
        with torch.no_grad():
            for s, z in lats.items():
                refs[s] = vae32.decode(z.to(dev)).sample
    except Exception as e:  # noqa: BLE001
        obs["setup_error"] = f"vae: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        _write(args.out, obs)
        return 0

    # ---- fp16 decode (gfx103x dtype): the #18016 NaN
    try:
        vae16 = copy.deepcopy(vae32).to(torch.float16)
        obs["fp16_installed"] = C.install(vae16, target)
        for s in ("1024", "512"):
            z = lats[s].to(dev, torch.float16)
            for arm in ("default", "kill"):
                cell: dict = {"size": int(s), "arm": arm}
                try:
                    drain()
                    with torch.no_grad():
                        out = run_arm(arm, lambda: vae16.decode(z).sample)
                    torch.cuda.synchronize()
                    cell["engagement"] = drain()
                    cell["vs_fp32"] = numerics(out, refs[s])
                    cell["sha256"] = _sha(out)
                    del out
                except Exception as e:  # noqa: BLE001
                    cell["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}"
                obs["fp16"].append(cell)
                print(f"fp16 {s} {arm}: {cell.get('vs_fp32')} {cell.get('engagement')} {cell.get('error', '')}",
                      flush = True)
        del vae16
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        obs["fp16_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    _write(args.out, obs)

    # ---- bf16 decode (gfx1151 default), timed
    b = obs["bf16"]
    try:
        vbf = copy.deepcopy(vae32).to(torch.bfloat16)
        vpr = copy.deepcopy(vbf)
        b["installed"] = C.install(vbf, target)
        b["installed_pr_orig"] = P.install(vpr, target)
        z = lats["1024"].to(dev, torch.bfloat16)

        def dec(v):
            def fn():
                with torch.no_grad():
                    return v.decode(z).sample
            return fn

        arms = {"half": dec(vbf), "half_aa": dec(vbf), "kill": dec(vbf), "pr_orig": dec(vpr)}
        drain()
        outs, eng = {}, {}
        for n, fn in arms.items():
            outs[n] = run_arm(n, fn)
            eng[n] = drain()
        torch.cuda.synchronize()
        b["engagement"] = eng
        b["sha256"] = {n: _sha(o) for n, o in outs.items()}
        b["half_aa_bitwise"] = bool(torch.equal(outs["half"], outs["half_aa"]))
        b["vs_fp32"] = {n: numerics(o, refs["1024"]) for n, o in outs.items()}
        del outs
        torch.cuda.empty_cache()
        b["timing"] = timed(arms, args.reps, args.warmup)
        drain()
        del vbf, vpr
    except Exception as e:  # noqa: BLE001
        b["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    _write(args.out, obs)
    print(f"bf16 decode: {b.get('timing', {}).get('half', {}).get('median_ms')} kill "
          f"{b.get('timing', {}).get('kill', {}).get('median_ms')} pr_orig "
          f"{b.get('timing', {}).get('pr_orig', {}).get('median_ms')} {b.get('error', '')}", flush = True)

    # ---- bf16 attention op at the 1024 px mid-block shape
    a = obs["attn"]
    try:
        g = torch.Generator(device = dev).manual_seed(16384)
        q, k, v = (torch.randn(1, 1, 16384, 512, device = dev, generator = g).to(torch.bfloat16) for _ in range(3))
        a["fused_sdpa_available"] = bool(C.fused_sdpa_available(q, k, v))

        def att(M, name):
            def fn():
                m = M.chunked_attention_mode()
                with m, torch.no_grad():
                    out = F.scaled_dot_product_attention(q, k, v)
                a.setdefault("engagement", {})[name] = {"half": m.half, "chunked": m.chunked}
                return out
            return fn

        arms = {"half": att(C, "half"), "half_aa": att(C, "half_aa"), "kill": att(C, "kill"),
                "pr_orig": att(P, "pr_orig")}
        a["timing"] = timed(arms, args.reps, args.warmup)
        drain()
    except Exception as e:  # noqa: BLE001
        a["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    obs["elapsed_s"] = round(time.time() - t0, 1)
    _write(args.out, obs)
    print(f"attn: {a.get('timing', {}).get('half', {}).get('median_ms')} kill "
          f"{a.get('timing', {}).get('kill', {}).get('median_ms')} pr_orig "
          f"{a.get('timing', {}).get('pr_orig', {}).get('median_ms')} {a.get('error', '')}", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
