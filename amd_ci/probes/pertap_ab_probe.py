#!/usr/bin/env python3
"""LTX-2.3 VAE decode, base vs head (Windows ROCm per-tap Conv3d) vs head + kill switch, every arm a fresh process.

Arms decode one saved N(0,1) latent per clip through Studio's decode route (MIOpen immediate, enable_tiling, LTX wide
tiles, untiled-when-fits); head also runs the PR's installer exactly as video.py does. Records warm (median of 3 after
the first), first, peak, how many Conv3d calls took the per-tap path, which tree each arm imported; PSNR vs CPU fp32
and vs base. Observes only; writes JSON via --out.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

LTX_SHAPES = {
    "s512x288x97": (1, 128, 13, 9, 16),
    "m1280x704x49": (1, 128, 7, 22, 40),
    "l1920x1088x121": (1, 128, 16, 34, 60),
}


def err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:400]}"


def compare(x, ref) -> dict:
    x, ref = x.float(), ref.float()
    if x.shape != ref.shape:
        return {"shape_mismatch": [list(x.shape), list(ref.shape)]}
    d = (x - ref).abs()
    mse = float(((x.clamp(-1, 1) - ref.clamp(-1, 1)) / 2).pow(2).mean())
    return {"max_abs": round(float(d.max()), 5), "mean_abs": round(float(d.mean()), 6),
            "psnr_db": round(10 * math.log10(1 / max(mse, 1e-20)), 2), "bit_identical": bool(d.max() == 0)}


def _sync(dev):
    import torch

    if dev == "cuda":
        torch.cuda.synchronize()


def _time_calls(fn, dev, n):
    ts, out = [], None
    for _ in range(n):
        _sync(dev)
        t0 = time.perf_counter()
        out = fn()
        _sync(dev)
        ts.append(time.perf_counter() - t0)
    return ts, out


def _timing(ts):
    warm = ts[1:]
    return {"first_s": round(ts[0], 3), "warm_median_s": round(statistics.median(warm), 4) if warm else None,
            "warm_spread_pct": round(100 * (max(warm) - min(warm)) / statistics.median(warm), 2) if len(warm) > 1 else None,
            "all_s": [round(t, 4) for t in ts]}


def _logger(lines):
    import logging

    lg = logging.getLogger("pertap_ab_probe")
    lg.setLevel(logging.INFO)

    class H(logging.Handler):
        def emit(self, rec):
            lines.append(rec.getMessage()[:300])

    lg.addHandler(H())
    return lg


def _backend(checkout):
    p = str(Path(checkout) / "studio" / "backend")
    if p not in sys.path:
        sys.path.insert(0, p)


def _tree_info(r):
    import core.inference.video as V

    r["imported_video_py"] = V.__file__
    try:
        import core.inference.video_vae_rocm_pertap as P

        r["pertap_module"] = P.__file__
        r["pertap_platform_applies"] = bool(P.platform_applies(__import__("torch")))
        r["pertap_env"] = os.environ.get(P.PERTAP_ENV)
        return P
    except ImportError as exc:
        r["pertap_module"] = f"absent ({exc})"
        return None


def child_gpucheck(checkout, out_path):
    """Head only: the patched Conv3d forced onto the per-tap path vs the stock op on the GPU (LTX geometries, bf16 +
    fp32), then the cliff conv (Cin=Cout=128, 3x3x3, 123 frames at 272x480: input > 2 GiB) through the real gate."""
    import torch
    import torch.nn.functional as F

    r = {"arm": "gpucheck"}
    try:
        _backend(checkout)
        P = _tree_info(r)
        from core.inference.diffusion_speed import _enable_miopen_immediate

        r["immediate"] = bool(_enable_miopen_immediate(_logger([])))
        g = torch.Generator(device = "cpu").manual_seed(1)
        checks = {}
        saved = P.ENGAGE_BYTES
        P.ENGAGE_BYTES = 0
        for dtype in (torch.float32, torch.bfloat16):
            for name, (cin, cout, k, pad, T) in {"k3_tpad0": (64, 96, (3, 3, 3), (0, 1, 1), 9),
                                                  "k1": (64, 32, (1, 1, 1), (0, 0, 0), 5),
                                                  "k3_pad_all": (32, 32, (3, 3, 3), (1, 1, 1), 6)}.items():
                conv = torch.nn.Conv3d(cin, cout, k, padding = pad).to("cuda", dtype)
                x = torch.randn(1, cin, T, 40, 48, generator = g).to("cuda", dtype)
                with torch.no_grad():
                    ref64 = F.conv3d(x.double().cpu(), conv.weight.double().cpu(), conv.bias.double().cpu(), 1, pad)
                    stock = conv(x).double().cpu()
                    conv.forward = types.MethodType(P._forward, conv)
                    y = conv(x).double().cpu()
                checks[f"{name}/{str(dtype)[6:]}"] = {
                    "pertap_calls": int(getattr(conv, "_unsloth_rocm_pertap_calls", 0)),
                    "pertap_max_err_vs_fp64": float((y - ref64).abs().max()),
                    "stock_max_err_vs_fp64": float((stock - ref64).abs().max()),
                    "ref_absmax": float(ref64.abs().max())}
        P.ENGAGE_BYTES = saved
        r["checks"] = checks
        conv = torch.nn.Conv3d(128, 128, 3, padding = (0, 1, 1)).to("cuda", torch.bfloat16)
        x = torch.randn(1, 128, 123, 272, 480, device = "cuda", dtype = torch.bfloat16)
        r["cliff_input_bytes"] = x.numel() * 2
        with torch.no_grad():
            ts, ref = _time_calls(lambda: conv(x), "cuda", 3)
            r["cliff_stock_ms"] = [round(1000 * t, 1) for t in ts]
            conv.forward = types.MethodType(P._forward, conv)
            ts, y = _time_calls(lambda: conv(x), "cuda", 3)
            r["cliff_pertap_ms"] = [round(1000 * t, 1) for t in ts]
            r["cliff_pertap_calls"] = int(getattr(conv, "_unsloth_rocm_pertap_calls", 0))
            r["cliff_vs_stock"] = compare(y[:, :, ::30].cpu(), ref[:, :, ::30].cpu())  # frames 0, 30, ..., 120
        r["ok"] = True
    except Exception as exc:  # noqa: BLE001
        r["ok"] = False
        r["error"] = err(exc)
        r["tail"] = traceback.format_exc().splitlines()[-12:]
    out_path.write_text(json.dumps(r, indent = 1), encoding = "utf-8")


def child_ltx(arm, shape_key, checkout, out_path):
    import torch

    r = {"arm": arm, "shape": shape_key, "checkout": checkout}
    work = out_path.parent
    try:
        lines = []
        lg = _logger(lines)
        dev = "cpu" if arm == "cpu_ref" else "cuda"
        dtype = torch.float32 if dev == "cpu" else torch.bfloat16
        _backend(checkout)
        P = _tree_info(r)
        if dev == "cuda":
            from core.inference.diffusion_speed import _enable_miopen_immediate

            r["immediate"] = bool(_enable_miopen_immediate(lg))
        from core.inference.video_ltx2 import load_ltx23_vae

        t0 = time.perf_counter()
        vae = load_ltx23_vae({}, variant = "distilled", torch_dtype = dtype, hf_token = None).to(dev).eval()
        r["load_s"] = round(time.perf_counter() - t0, 1)
        z = torch.load(work / f"ltx_latent_{shape_key}.pt").to(dev, dtype)
        if dev == "cuda":
            vae.enable_tiling()
            from core.inference.video_ltx2_vae_tiles import install as install_tiles
            from core.inference.video_vae_untiled import install_untiled_decode

            r["wide_tiles"] = bool(install_tiles(vae, lg))
            pipe = types.SimpleNamespace(vae = vae)
            r["untiled_when_fits"] = bool(install_untiled_decode(pipe, "ltx-2", logger = lg))
            if P is not None:  # what video.py runs right after the frame pad fix
                r["pertap_installed_convs"] = int(P.install_on_pipe(pipe, types.SimpleNamespace(device = "cuda"), logger = lg))
            decode = pipe.vae.decode
        else:
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            vae.use_tiling = False
            decode = vae.decode
        n = 1 if dev == "cpu" else 4
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            ts, vid = _time_calls(lambda: decode(z, return_dict = False)[0], dev, n)
        r.update(_timing(ts))
        if dev == "cuda":
            r["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            r["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        if P is not None:
            total = int(P.pertap_calls(vae))
            r["pertap_calls_total"] = total
            r["pertap_calls_per_decode"] = total / n
            r["pertap_engaged_convs"] = sorted(name for name, m in vae.decoder.named_modules()
                                               if getattr(m, "_unsloth_rocm_pertap_calls", 0))
        r["last_tile"] = list(vae.__dict__.get("_unsloth_last_decode_tile") or [])[:3]
        r["out_shape"] = list(vid.shape)
        r["finite"] = bool(torch.isfinite(vid).all())
        r["studio_log"] = lines[:16]
        torch.save(vid.to(torch.float16).cpu(), work / f"ltx_{shape_key}_{arm}.pt")
        r["ok"] = True
    except Exception as exc:  # noqa: BLE001
        r["ok"] = False
        r["error"] = err(exc)
        r["tail"] = traceback.format_exc().splitlines()[-12:]
    out_path.write_text(json.dumps(r, indent = 1), encoding = "utf-8")


def run_arm(work, args, timeout_s, tag, env = None):
    jp, log = work / f"{tag}.json", work / f"{tag}.log"
    t0 = time.time()
    try:
        with open(log, "w", encoding = "utf-8") as fh:
            rc = subprocess.run([sys.executable, "-u", __file__, *args, "--res", str(jp)], stdout = fh,
                                stderr = subprocess.STDOUT, timeout = timeout_s,
                                env = {**os.environ, **(env or {})}).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    row = json.loads(jp.read_text(encoding = "utf-8")) if jp.exists() else {"ok": False, "error": "no record"}
    row["rc"], row["wall_s"] = rc, round(time.time() - t0, 1)
    if not row.get("ok"):
        try:
            row["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace").splitlines()[-15:]
        except OSError:
            pass
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: rc={rc} ok={row.get('ok')} wall={row['wall_s']}s "
          f"first={row.get('first_s')} warm={row.get('warm_median_s')} calls={row.get('pertap_calls_total')}", flush = True)
    return row


ARM_TREE = {"base": "base", "base_aa": "base", "head": "head", "head_aa": "head", "head_kill": "head", "cpu_ref": "base"}
KILL = {"UNSLOTH_VIDEO_VAE_ROCM_PERTAP": "0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--head")
    ap.add_argument("--out")
    ap.add_argument("--work")
    ap.add_argument("--shapes", default = "s512x288x97,m1280x704x49,l1920x1088x121")
    ap.add_argument("--arms", default = "base,head,head_kill,base_aa,head_aa")
    ap.add_argument("--cpu-ref", default = "s512x288x97,m1280x704x49")
    ap.add_argument("--cpu-ref-last", default = "")
    ap.add_argument("--gpucheck", action = "store_true")
    ap.add_argument("--child", nargs = 3)
    ap.add_argument("--child-gpucheck")
    ap.add_argument("--res")
    a = ap.parse_args()
    os.environ.pop("HF_TOKEN", None)
    if a.child:
        child_ltx(a.child[0], a.child[1], a.child[2], Path(a.res))
        return 0
    if a.child_gpucheck:
        child_gpucheck(a.child_gpucheck, Path(a.res))
        return 0
    import torch

    work = Path(a.work)
    work.mkdir(parents = True, exist_ok = True)
    tree = {"base": a.base, "head": a.head}
    out = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": sys.platform, "torch": torch.__version__,
           "hip": torch.version.hip, "hf_token_present": bool(os.environ.get("HF_TOKEN"))}
    try:
        out["device"] = torch.cuda.get_device_name(0)
        out["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)
        import diffusers

        out["diffusers"] = diffusers.__version__
    except Exception as exc:  # noqa: BLE001
        out["env_error"] = err(exc)

    def flush():
        Path(a.out).write_text(json.dumps(out, indent = 1), encoding = "utf-8")

    flush()
    rows = out.setdefault("rows", {})
    if a.gpucheck:
        rows["gpucheck"] = run_arm(work, ["--child-gpucheck", a.head], 1800, "gpucheck")
        flush()
    shapes = [s for s in a.shapes.split(",") if s]
    for sk in shapes:
        g = torch.Generator(device = "cpu").manual_seed(0)
        torch.save(torch.randn(*LTX_SHAPES[sk], generator = g), work / f"ltx_latent_{sk}.pt")
    cpu_first = [s for s in a.cpu_ref.split(",") if s in shapes]
    cpu_last = [s for s in a.cpu_ref_last.split(",") if s in shapes]
    arms = [x for x in a.arms.split(",") if x]
    for sk in shapes:
        if sk in cpu_first:
            rows[f"{sk}/cpu_ref"] = run_arm(work, ["--child", "cpu_ref", sk, tree["base"]], 3600, f"ltx_{sk}_cpu_ref")
            flush()
        for arm in arms:
            rows[f"{sk}/{arm}"] = run_arm(work, ["--child", arm, sk, tree[ARM_TREE[arm]]], 3000, f"ltx_{sk}_{arm}",
                                          KILL if arm == "head_kill" else None)
            flush()
    for sk in cpu_last:
        rows[f"{sk}/cpu_ref"] = run_arm(work, ["--child", "cpu_ref", sk, tree["base"]], 3600, f"ltx_{sk}_cpu_ref")
        flush()
    q = {}
    for sk in shapes:
        try:
            refs = {}
            for name in ("base", "cpu_ref"):
                p = work / f"ltx_{sk}_{name}.pt"
                if p.exists():
                    refs[name] = torch.load(p).float()
            for arm in arms:
                p = work / f"ltx_{sk}_{arm}.pt"
                if not p.exists():
                    continue
                x = torch.load(p).float()
                e = {}
                if "base" in refs and arm != "base":
                    e["vs_base"] = compare(x, refs["base"])
                if "cpu_ref" in refs:
                    e["vs_cpu_fp32"] = compare(x, refs["cpu_ref"])
                q[f"{sk}/{arm}"] = e
                del x
            refs.clear()
        except Exception as exc:  # noqa: BLE001
            q[f"{sk}/compare_error"] = err(exc)
        out["quality"] = q
        flush()
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
