#!/usr/bin/env python3
"""AMD gfx1151 VAE ideas probe (observes only; every arm in a fresh child process).

Linux suite (Qwen-Image VAE, Qwen/Qwen-Image subfolder vae, ~0.25 GB):
  #5 decode of one saved latent at 1664x928: Studio full-frame (bf16, MIOpen immediate, use_tiling False) vs
     Studio's tiled route (diffusion_vae_tiling.install + enable_tiling); plus a Find-mode full-frame diagnostic.
  #2 encode at 1024^2 and 1280^2 as Studio training does (fp32 VAE, fp32 pixels, T=1, untiled, torch default
     MIOpen = Find, cudnn.benchmark off) vs cudnn.enabled=False vs MIOpen immediate.
Windows suite (LTX-2.3 video VAE, unsloth/LTX-2.3-GGUF vae/ltx-2.3-22b-distilled_video_vae.safetensors):
  #6 Studio's decode route (enable_tiling + ltx2 wide tiles + untiled-when-fits, MIOpen immediate) vs the same route
     with every Conv3d decomposed into per-temporal-tap Conv2d calls; stock 256 / 512 px tile replay (#5845).
CPU fp32 references for quality. Writes JSON via --out only.
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

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"

QWEN_REPO = "Qwen/Qwen-Image"
LTX_REPO = "unsloth/LTX-2.3-GGUF"
LTX_FILE = "vae/ltx-2.3-22b-distilled_video_vae.safetensors"
# name -> latent shape (B, C, F, h, w); output = 32x spatial, 8x temporal (+1)
LTX_SHAPES = {
    "s512x288x97": (1, 128, 13, 9, 16),
    "m1280x704x49": (1, 128, 7, 22, 40),
    "l1920x1088x121": (1, 128, 16, 34, 60),
}


def err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:400]}"


def _psnr(x, ref) -> float:
    mse = float(((x.float().clamp(-1, 1) - ref.float().clamp(-1, 1)) / 2).pow(2).mean())
    return round(10 * math.log10(1 / max(mse, 1e-20)), 2)


def compare(x, ref) -> dict:
    x, ref = x.float(), ref.float()
    if x.shape != ref.shape:
        return {"shape_mismatch": [list(x.shape), list(ref.shape)]}
    d = (x - ref).abs()
    out = {"max_abs": round(float(d.max()), 5), "mean_abs": round(float(d.mean()), 6), "psnr_db": _psnr(x, ref)}
    out["rel_l2"] = float((x - ref).norm() / ref.norm().clamp_min(1e-12))
    return out


def _load_img(name: str):
    import numpy as np
    import torch
    from PIL import Image

    a = np.asarray(Image.open(ASSETS / f"{name}.png").convert("RGB"), dtype = np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).contiguous()  # 1,3,H,W in [-1,1]


def _sync(dev: str) -> None:
    import torch

    if dev == "cuda":
        torch.cuda.synchronize()


def _time_calls(fn, dev: str, n: int):
    ts, out = [], None
    for _ in range(n):
        _sync(dev)
        t0 = time.perf_counter()
        out = fn()
        _sync(dev)
        ts.append(time.perf_counter() - t0)
    return ts, out


def _timing(ts: list) -> dict:
    warm = ts[1:]
    return {"first_s": round(ts[0], 3), "warm_median_s": round(statistics.median(warm), 4) if warm else None,
            "warm_min_s": round(min(warm), 4) if warm else None,
            "warm_spread_pct": round(100 * (max(warm) - min(warm)) / statistics.median(warm), 2) if len(warm) > 1 else None,
            "all_s": [round(t, 4) for t in ts]}


def _studio_logger(lines: list):
    import logging

    lg = logging.getLogger("amd_vae_probe")
    lg.setLevel(logging.INFO)

    class H(logging.Handler):
        def emit(self, rec):
            lines.append(rec.getMessage()[:300])

    lg.addHandler(H())
    return lg


def _backend(checkout: str) -> None:
    p = str(Path(checkout) / "studio" / "backend")
    if p not in sys.path:
        sys.path.insert(0, p)


# ----------------------------------------------------------------- per-tap Conv3d -> Conv2d
def pertap_conv3d_forward(self, x):
    """Conv3d as a sum over temporal taps of Conv2d calls (time folded into batch), fp32 accumulation per output
    frame chunk, bias once. Falls back to the native op for cases it does not cover."""
    import torch
    import torch.nn.functional as F

    if self.groups != 1 or self.padding_mode != "zeros" or isinstance(self.padding, str) or x.dim() != 5:
        return self._amd_native_forward(x)
    kt, kh, kw = self.kernel_size
    st, sh, sw = self.stride
    dt, dh, dw = self.dilation
    pt, ph, pw = self.padding
    if pt:
        x = F.pad(x, (0, 0, 0, 0, pt, pt))
    B, C, T, H, W = x.shape
    tout = (T - dt * (kt - 1) - 1) // st + 1
    xt = x.transpose(1, 2)
    if not xt.is_contiguous():
        xt = xt.contiguous()
    ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    O = self.out_channels
    out = torch.empty((B, tout, O, ho, wo), device = x.device, dtype = x.dtype)
    per_frame = B * O * ho * wo * 4
    chunk = max(1, min(tout, int((256 << 20) // max(per_frame, 1))))
    w = self.weight
    for t0 in range(0, tout, chunk):
        t1 = min(tout, t0 + chunk)
        n = t1 - t0
        acc = None
        for k in range(kt):
            s = t0 * st + k * dt
            sl = xt[:, s: s + st * (n - 1) + 1: st]
            y = F.conv2d(sl.reshape(B * n, C, H, W), w[:, :, k], None, (sh, sw), (ph, pw), (dh, dw))
            if acc is None:
                acc = y.float()
            else:
                acc.add_(y)
        if self.bias is not None:
            acc.add_(self.bias.float().view(1, -1, 1, 1))
        out[:, t0:t1] = acc.view(B, n, O, ho, wo).to(x.dtype)
    return out.transpose(1, 2)


def install_pertap(module) -> int:
    import torch

    n = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Conv3d) and not hasattr(m, "_amd_native_forward"):
            m._amd_native_forward = m.forward
            m.forward = types.MethodType(pertap_conv3d_forward, m)
            n += 1
    return n


def pertap_selfcheck(dev: str, dtype) -> dict:
    """Per-tap vs native on random tensors for the conv geometries the LTX decoder uses."""
    import torch

    g = torch.Generator(device = "cpu").manual_seed(1)
    res = {}
    for name, (cin, cout, k, stride, pad, T) in {
        "k3s1p1": (64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), 7),
        "k3_tpad0": (64, 96, (3, 3, 3), (1, 1, 1), (0, 1, 1), 9),
        "k1": (64, 32, (1, 1, 1), (1, 1, 1), (0, 0, 0), 5),
    }.items():
        conv = torch.nn.Conv3d(cin, cout, k, stride, pad).to(dev, dtype)
        with torch.no_grad():
            x = torch.randn(1, cin, T, 20, 24, generator = g).to(dev, dtype)
            ref = conv(x)
            install_pertap(conv)
            y = conv(x)
        res[name] = {"max_abs": float((y.float() - ref.float()).abs().max()),
                     "ref_absmax": float(ref.float().abs().max())}
    return res


def layer_profile(vae, z, dev: str) -> list:
    """Every decoder Conv3d call shape of one untiled decode, then native vs per-tap time per unique (conv, shape)."""
    import torch

    seen: dict = {}
    hooks = []
    for name, m in vae.decoder.named_modules():
        if isinstance(m, torch.nn.Conv3d):
            def pre(mod, args, name = name):
                key = (name, tuple(args[0].shape))
                seen[key] = seen.get(key, 0) + 1
            hooks.append(m.register_forward_pre_hook(pre))
    with torch.no_grad():
        vae.use_tiling = False
        vae.decode(z[:, :, :min(z.shape[2], 16)], return_dict = False)
    for h in hooks:
        h.remove()
    mods = dict(vae.decoder.named_modules())
    rows = []
    for (name, shape), calls in seen.items():
        m = mods[name]
        x = torch.randn(shape, device = dev, dtype = m.weight.dtype)
        res = {"conv": name, "in_shape": list(shape), "calls": calls, "k": list(m.kernel_size),
               "cout": m.out_channels, "stride": list(m.stride)}
        with torch.no_grad():
            for label, fn in (("native", lambda: m._amd_native_forward(x) if hasattr(m, "_amd_native_forward") else m(x)),
                              ("pertap", lambda: pertap_conv3d_forward(m, x))):
                if not hasattr(m, "_amd_native_forward"):
                    m._amd_native_forward = m.forward
                ts, y = _time_calls(fn, dev, 3)
                res[f"{label}_ms"] = round(1000 * min(ts[1:]), 2)
                res[f"{label}_first_ms"] = round(1000 * ts[0], 2)
                del y
        res["speedup"] = round(res["native_ms"] / max(res["pertap_ms"], 1e-6), 3)
        rows.append(res)
        print("layerprof", json.dumps(res), flush = True)
        del x
        torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------- children
def child_qwen(arm: str, out_path: Path, checkout: str) -> None:
    import torch

    r: dict = {"arm": arm}
    work = out_path.parent
    try:
        import diffusers

        dev = "cpu" if arm == "cpu_ref" else "cuda"
        if arm.startswith(("dec_", "enc_imm")):
            torch.backends.miopen.immediate = True  # Studio inference sets this (diffusion_speed._enable_miopen_immediate)
        if arm.startswith("enc_nocudnn"):
            torch.backends.cudnn.enabled = False
        if arm.startswith("dec_full_find"):
            torch.backends.miopen.immediate = False
        r["miopen_immediate"] = bool(getattr(torch.backends.miopen, "immediate", None))
        r["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
        r["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        dtype = torch.bfloat16 if arm.startswith("dec_") else torch.float32
        vae = diffusers.AutoencoderKLQwenImage.from_pretrained(QWEN_REPO, subfolder = "vae", torch_dtype = dtype,
                                                               token = False).to(dev).eval()
        r["vae_class"] = type(vae).__name__
        with torch.no_grad():
            if arm == "cpu_ref":
                torch.set_num_threads(max(1, os.cpu_count() or 1))
                x = _load_img("q1664x928").unsqueeze(2)
                t0 = time.perf_counter()
                z = vae.encode(x).latent_dist.mean
                r["cpu_encode_1664_s"] = round(time.perf_counter() - t0, 1)
                torch.save(z, work / "qwen_latent_1664x928.pt")
                t0 = time.perf_counter()
                torch.save(vae.decode(z).sample.squeeze(2), work / "qwen_dec_cpu.pt")
                r["cpu_decode_s"] = round(time.perf_counter() - t0, 1)
                for size in ("q1024", "q1280"):
                    t0 = time.perf_counter()
                    d = vae.encode(_load_img(size).unsqueeze(2)).latent_dist
                    torch.save({"mean": d.mean, "logvar": d.logvar}, work / f"qwen_enc_cpu_{size}.pt")
                    r[f"cpu_encode_{size}_s"] = round(time.perf_counter() - t0, 1)
            elif arm.startswith("dec_"):
                lines: list = []
                lg = _studio_logger(lines)
                _backend(checkout)
                z = torch.load(work / "qwen_latent_1664x928.pt").to(dev, dtype)
                tiled = arm.startswith("dec_tiled")
                if tiled:
                    from core.inference import diffusion_vae_tiling as T

                    r["studio_tiling_install"] = bool(T.install(vae, lg))
                    vae.enable_tiling()
                else:
                    vae.use_tiling = False
                torch.cuda.reset_peak_memory_stats()
                ts, img = _time_calls(lambda: vae.decode(z).sample, dev, 6)
                r.update(_timing(ts))
                r["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
                r["use_tiling"] = bool(vae.use_tiling)
                r["tile_geometry"] = {k: getattr(vae, k, None) for k in ("tile_sample_min_height", "tile_sample_min_width",
                                                                         "tile_sample_stride_height", "tile_sample_stride_width")}
                r["last_tile"] = list(vae.__dict__.get("_unsloth_last_decode_tile") or [])[:3]
                r["studio_log"] = lines[:12]
                img = img.squeeze(2).float().cpu()
                torch.save(img, work / f"qwen_{arm}.pt")
            elif arm.startswith("enc_"):
                size = arm.rsplit("_", 1)[-1]
                x = _load_img(size).unsqueeze(2).to(dev, torch.float32)
                torch.cuda.reset_peak_memory_stats()
                ts, d = _time_calls(lambda: vae.encode(x).latent_dist, dev, 9)
                r.update(_timing(ts))
                r["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
                torch.save({"mean": d.mean.float().cpu(), "logvar": d.logvar.float().cpu()}, work / f"qwen_{arm}.pt")
        r["ok"] = True
    except Exception as exc:  # noqa: BLE001
        r["ok"] = False
        r["error"] = err(exc)
        r["tail"] = traceback.format_exc().splitlines()[-10:]
    out_path.write_text(json.dumps(r, indent = 1), encoding = "utf-8")


def _ltx_vae(checkout: str, dtype, lg):
    _backend(checkout)
    from core.inference.video_ltx2 import load_ltx23_vae

    return load_ltx23_vae({}, variant = "distilled", torch_dtype = dtype, hf_token = None)


def child_ltx(arm: str, shape_key: str, out_path: Path, checkout: str) -> None:
    import torch

    r: dict = {"arm": arm, "shape": shape_key}
    work = out_path.parent
    try:
        lines: list = []
        lg = _studio_logger(lines)
        dev = "cpu" if arm == "cpu_ref" else "cuda"
        dtype = torch.float32 if dev == "cpu" else torch.bfloat16
        if dev == "cuda":
            _backend(checkout)
            if arm.endswith("_find"):
                torch.backends.miopen.immediate = False
            else:
                from core.inference.diffusion_speed import _enable_miopen_immediate

                r["studio_immediate_call"] = bool(_enable_miopen_immediate(lg))
            if arm.startswith("nocudnn"):
                torch.backends.cudnn.enabled = False
        r["miopen_immediate"] = bool(getattr(torch.backends.miopen, "immediate", None))
        r["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
        t0 = time.perf_counter()
        vae = _ltx_vae(checkout, dtype, lg).to(dev).eval()
        r["load_s"] = round(time.perf_counter() - t0, 1)
        zp = work / f"ltx_latent_{shape_key}.pt"
        z = torch.load(zp).to(dev, dtype)
        if arm == "layerprof":
            r["layers"] = layer_profile(vae, z, dev)
            r["ok"] = True
            out_path.write_text(json.dumps(r, indent = 1), encoding = "utf-8")
            return
        if arm.startswith("pertap"):
            r["pertap_selfcheck"] = pertap_selfcheck(dev, dtype)
            r["pertap_convs"] = install_pertap(vae.decoder)
        if arm.startswith(("studio", "pertap", "nocudnn")):
            vae.enable_tiling()
            from core.inference.video_ltx2_vae_tiles import install as install_tiles
            from core.inference.video_vae_untiled import install_untiled_decode

            r["wide_tiles"] = bool(install_tiles(vae, lg))
            pipe = types.SimpleNamespace(vae = vae)
            r["untiled_when_fits"] = bool(install_untiled_decode(pipe, "ltx-2", logger = lg))
            decode = pipe.vae.decode
        elif arm.startswith(("tile256", "tile512")):
            px = 256 if arm.startswith("tile256") else 512
            vae.enable_tiling(tile_sample_min_height = px, tile_sample_min_width = px,
                              tile_sample_stride_height = px * 3 // 4, tile_sample_stride_width = px * 3 // 4)
            decode = vae.decode
        else:  # cpu_ref: untiled fp32
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            vae.use_tiling = False
            decode = vae.decode
        n = 1 if dev == "cpu" else int(os.environ.get("AMD_VAE_REPS", "4"))
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
            try:
                r["mem_get_info_gib_before"] = [round(v / 2**30, 2) for v in torch.cuda.mem_get_info()]
            except Exception:  # noqa: BLE001
                pass
        with torch.no_grad():
            ts, vid = _time_calls(lambda: decode(z, return_dict = False)[0], dev, n)
        r.update(_timing(ts))
        if dev == "cuda":
            r["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            r["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        r["out_shape"] = list(vid.shape)
        r["finite"] = bool(torch.isfinite(vid).all())
        r["studio_log"] = lines[:16]
        torch.save(vid.to(torch.float16).cpu(), work / f"ltx_{shape_key}_{arm}.pt")
        r["ok"] = True
    except Exception as exc:  # noqa: BLE001
        r["ok"] = False
        r["error"] = err(exc)
        r["tail"] = traceback.format_exc().splitlines()[-10:]
    out_path.write_text(json.dumps(r, indent = 1), encoding = "utf-8")


# ----------------------------------------------------------------- parent
def run_arm(work: Path, args: list, timeout_s: float, tag: str) -> dict:
    jp = work / f"{tag}.json"
    log = work / f"{tag}.log"
    t0 = time.time()
    try:
        with open(log, "w", encoding = "utf-8") as fh:
            rc = subprocess.run([sys.executable, "-u", __file__, *args, str(jp)], stdout = fh,
                                stderr = subprocess.STDOUT, timeout = timeout_s).returncode
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
          f"first={row.get('first_s')} warm={row.get('warm_median_s')}", flush = True)
    return row


def suite_linux(work: Path, checkout: str, out: dict, flush) -> None:
    import torch

    rows = out.setdefault("qwen", {})
    rows["cpu_ref"] = run_arm(work, ["--child-qwen", "cpu_ref", "--checkout", checkout], 3600, "qwen_cpu_ref")
    flush()
    dec_arms = ["dec_full", "dec_tiled", "dec_full_aa", "dec_tiled_aa", "dec_full_find"]
    enc_arms = [f"{a}_{s}" for s in ("q1024", "q1280") for a in ("enc_default", "enc_nocudnn", "enc_imm",
                                                                    "enc_default_aa", "enc_nocudnn_aa")]
    for a in dec_arms + enc_arms:
        rows[a] = run_arm(work, ["--child-qwen", a, "--checkout", checkout], 2400, f"qwen_{a}")
        flush()
    q: dict = {}
    try:
        cref = torch.load(work / "qwen_dec_cpu.pt")
        full = torch.load(work / "qwen_dec_full.pt") if (work / "qwen_dec_full.pt").exists() else None
        for a in dec_arms:
            p = work / f"qwen_{a}.pt"
            if p.exists():
                x = torch.load(p)
                q[a] = {"vs_cpu_fp32": compare(x, cref)}
                if full is not None:
                    q[a]["vs_dec_full"] = compare(x, full)
        for s in ("q1024", "q1280"):
            cr = torch.load(work / f"qwen_enc_cpu_{s}.pt")
            dref = work / f"qwen_enc_default_{s}.pt"
            d0 = torch.load(dref) if dref.exists() else None
            for a in ("enc_default", "enc_nocudnn", "enc_imm", "enc_default_aa", "enc_nocudnn_aa"):
                p = work / f"qwen_{a}_{s}.pt"
                if p.exists():
                    x = torch.load(p)
                    e = {"mean_vs_cpu": compare(x["mean"], cr["mean"]), "logvar_vs_cpu": compare(x["logvar"], cr["logvar"])}
                    if d0 is not None:
                        e["mean_vs_default"] = compare(x["mean"], d0["mean"])
                        e["logvar_vs_default"] = compare(x["logvar"], d0["logvar"])
                    q[f"{a}_{s}"] = e
    except Exception as exc:  # noqa: BLE001
        q["compare_error"] = err(exc)
    out["qwen_quality"] = q


def suite_ltx_large(work: Path, checkout: str, out: dict, flush, with_ab: bool) -> None:
    import torch

    sk = "l1920x1088x121"
    g = torch.Generator(device = "cpu").manual_seed(0)
    torch.save(torch.randn(*LTX_SHAPES[sk], generator = g), work / f"ltx_latent_{sk}.pt")
    rows = out.setdefault("ltx", {})
    arms = ["layerprof"] + (["studio", "pertap", "studio_aa", "pertap_aa"] if with_ab else [])
    for arm in arms:
        rows[f"{sk}/{arm}"] = run_arm(work, ["--child-ltx", arm, sk, "--checkout", checkout], 3600, f"ltx_{sk}_{arm}")
        flush()
    if with_ab:
        try:
            ref = torch.load(work / f"ltx_{sk}_studio.pt").float()
            out["ltx_quality"] = {f"{sk}/{a}": {"vs_studio": compare(torch.load(work / f"ltx_{sk}_{a}.pt").float(), ref)}
                                  for a in ("studio_aa", "pertap", "pertap_aa") if (work / f"ltx_{sk}_{a}.pt").exists()}
        except Exception as exc:  # noqa: BLE001
            out["ltx_quality"] = {"compare_error": err(exc)}


def suite_windows(work: Path, checkout: str, out: dict, flush, shapes: list) -> None:
    import torch

    rows = out.setdefault("ltx", {})
    for sk in shapes:
        g = torch.Generator(device = "cpu").manual_seed(0)
        torch.save(torch.randn(*LTX_SHAPES[sk], generator = g), work / f"ltx_latent_{sk}.pt")
    plan = []
    for sk in shapes:
        if sk != "l1920x1088x121":
            plan.append(("cpu_ref", sk, 3600))
        plan += [("studio", sk, 2400), ("pertap", sk, 2400), ("studio_aa", sk, 2400), ("pertap_aa", sk, 2400)]
    # #5845 replay on the medium shape, Find-mode arms last (a TDR there must not cost the other arms)
    plan += [("tile256", "m1280x704x49", 1800), ("tile512", "m1280x704x49", 1800),
             ("nocudnn", "m1280x704x49", 2400),
             ("tile256_find", "m1280x704x49", 1800), ("tile512_find", "m1280x704x49", 1800)]
    for arm, sk, to in plan:
        rows[f"{sk}/{arm}"] = run_arm(work, ["--child-ltx", arm, sk, "--checkout", checkout], to, f"ltx_{sk}_{arm}")
        flush()
    q: dict = {}
    for sk in shapes:
        try:
            st = work / f"ltx_{sk}_studio.pt"
            cp = work / f"ltx_{sk}_cpu_ref.pt"
            ref_s = torch.load(st).float() if st.exists() else None
            ref_c = torch.load(cp).float() if cp.exists() else None
            for arm in ("studio", "studio_aa", "pertap", "pertap_aa", "tile256", "tile512", "nocudnn",
                        "tile256_find", "tile512_find"):
                p = work / f"ltx_{sk}_{arm}.pt"
                if not p.exists():
                    continue
                x = torch.load(p).float()
                e = {}
                if ref_s is not None and arm != "studio":
                    e["vs_studio"] = compare(x, ref_s)
                if ref_c is not None:
                    e["vs_cpu_fp32"] = compare(x, ref_c)
                q[f"{sk}/{arm}"] = e
        except Exception as exc:  # noqa: BLE001
            q[f"{sk}/compare_error"] = err(exc)
    out["ltx_quality"] = q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices = ("linux", "windows", "windows_prof"))
    ap.add_argument("--checkout")
    ap.add_argument("--out")
    ap.add_argument("--work")
    ap.add_argument("--shapes", default = "s512x288x97,m1280x704x49,l1920x1088x121")
    ap.add_argument("--child-qwen", nargs = 1)
    ap.add_argument("--child-ltx", nargs = 2)
    ap.add_argument("rest", nargs = "*")
    a = ap.parse_args()
    if a.child_qwen:
        child_qwen(a.child_qwen[0], Path(a.rest[-1]), a.checkout)
        return 0
    if a.child_ltx:
        child_ltx(a.child_ltx[0], a.child_ltx[1], Path(a.rest[-1]), a.checkout)
        return 0
    work = Path(a.work)
    work.mkdir(parents = True, exist_ok = True)
    out: dict = {"suite": a.suite, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import torch

        out["torch"] = torch.__version__
        out["hip"] = torch.version.hip
        out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        out["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if torch.cuda.is_available() else None
        import diffusers

        out["diffusers"] = diffusers.__version__
    except Exception as exc:  # noqa: BLE001
        out["env_error"] = err(exc)

    def flush() -> None:
        Path(a.out).write_text(json.dumps(out, indent = 1), encoding = "utf-8")

    flush()
    if a.suite == "linux":
        suite_linux(work, a.checkout, out, flush)
        suite_ltx_large(work, a.checkout, out, flush, with_ab = True)
    elif a.suite == "windows_prof":
        suite_ltx_large(work, a.checkout, out, flush, with_ab = False)
    else:
        suite_windows(work, a.checkout, out, flush, [s for s in a.shapes.split(",") if s])
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
