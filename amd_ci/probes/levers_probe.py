#!/usr/bin/env python3
"""Observe the vendor-blog AMD levers on this gfx1151 host (no verdict; JSON to --out, never stdout).

  python levers_probe.py --checkout UNSLOTH_TREE --out obs.json [--work DIR] [--skip p2]

P1 / V1  SDPA flash through the CK backend (torch.backends.cuda.preferred_rocm_fa_library("ck")) vs AOTriton:
         does CK engage on this wheel, per-call ms at Wan2.2-5B / FLUX.3 klein shapes, rel L2 vs math SDPA.
V6       preferred BLAS library, and per-GEMM ms for untabled DiT shapes (Z-Image, Wan 5B) under hipBLASLt vs rocBLAS.
P2 / V2  MIOpen find-db for VAE decode: fresh process per arm, fixed latents, decode ms (median of 3) and output diff:
         imm      immediate mode (Studio's default), empty user db
         imm_aa   the same again (A/A)
         find     MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=4 into the user db (one-off tuning cost)
         imm_db   immediate mode reading that user db (the shipped-db lever)
         FindDb / immediate hits from MIOPEN_LOG_LEVEL=5 on the imm_db arm.
VAE      Studio's Triton VAE kernels with the ROCm gate lifted in-process: GroupNorm+SiLU parity / ms at VAE shapes,
         then FLUX.2 / Wan2.2 VAE decode fused vs stock (fresh process per arm, A/A stock), diff + LPIPS.
MAPPING  4 GiB host->device copy from an mmap opened ACCESS_COPY (Studio's native cache today) vs ACCESS_READ:
         GiB/s cold / warm and RSS / RssAnon / RssFile after the copy and after close (Linux; psutil on Windows).
P3 / V3  FLUX.3 QK-norm+RoPE Triton prologue: Studio's own per-key bitwise probe + timed race (`_check`) on ROCm,
         run directly (the module's ROCm gate is what is in question).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path


def err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:400]}"


def cuda_ms(fn, iters = 20, repeats = 3):
    import torch

    fn()
    torch.cuda.synchronize()
    meds = []
    for _ in range(repeats):
        ts = []
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b))
        meds.append(statistics.median(ts))
    return {"ms": round(statistics.median(meds), 4), "repeats_ms": [round(x, 4) for x in meds]}


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


# ----------------------------------------------------------------------------------------------------------- P1
def p1(out: dict) -> None:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    r: dict = {}
    lib = getattr(torch.backends.cuda, "preferred_rocm_fa_library", None)
    r["api"] = lib is not None
    if lib is None:
        out["p1"] = r
        return
    r["before"] = str(lib())
    r["aotriton_version"] = str(getattr(torch._C, "_aotriton_version", lambda: None)()) \
        if hasattr(torch._C, "_aotriton_version") else None
    g = torch.Generator(device = "cpu").manual_seed(0)

    def qkv(shape):
        return [torch.randn(*shape, generator = g).to("cuda", torch.bfloat16) for _ in range(3)]

    def flash(q, k, v):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v)

    small = qkv((1, 24, 1024, 128))
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(*[t.float() for t in small]).to(torch.bfloat16)
    arms = {}
    for name in ("aotriton", "ck"):
        a: dict = {}
        try:
            lib(name)
            a["set_to"] = str(lib())
            a["rel_l2_vs_math"] = rel_l2(flash(*small), ref)
            a["ok"] = True
        except Exception as exc:  # noqa: BLE001 - CK not built into this wheel is the expected failure
            a["ok"] = False
            a["error"] = err(exc)
        arms[name] = a
    shapes = [(1, 24, 27280, 128), (2, 24, 4096, 128), (1, 24, 4608, 128)]
    for shp in shapes:
        q, k, v = qkv(shp)
        iters = 5 if shp[2] > 20000 else 20
        for name in ("aotriton", "ck"):
            if not arms[name].get("ok"):
                continue
            try:
                lib(name)
                arms[name][f"ms_{'x'.join(map(str, shp))}"] = cuda_ms(lambda: flash(q, k, v), iters = iters)
            except Exception as exc:  # noqa: BLE001
                arms[name][f"err_{'x'.join(map(str, shp))}"] = err(exc)
        if arms["aotriton"].get("ok") and arms["ck"].get("ok"):
            lib("aotriton")
            o1 = flash(q, k, v)
            lib("ck")
            o2 = flash(q, k, v)
            arms["ck"][f"rel_l2_vs_aotriton_{'x'.join(map(str, shp))}"] = rel_l2(o2, o1)
        del q, k, v
    try:
        lib("default")
    except Exception:  # noqa: BLE001
        pass
    r["arms"] = arms
    out["p1"] = r


# ----------------------------------------------------------------------------------------------------------- V6
def v6(out: dict) -> None:
    import torch
    import torch.nn.functional as F

    r: dict = {}
    pref = getattr(torch.backends.cuda, "preferred_blas_library", None)
    r["default"] = str(pref()) if pref else None
    r["env"] = os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT")
    # (tokens, in, out): Z-Image-Turbo 1024^2 (dim 3840, ffn 10240), Wan2.2-5B 1280x704x21 (dim 3072, ffn 14336)
    shapes = [(4128, 3840, 3840), (4128, 3840, 10240), (4128, 10240, 3840),
              (5280, 3072, 3072), (5280, 3072, 14336), (5280, 14336, 3072)]
    g = torch.Generator(device = "cpu").manual_seed(1)
    res = []
    for n, k, m in shapes:
        x = torch.randn(n, k, generator = g).to("cuda", torch.bfloat16)
        w = (torch.randn(m, k, generator = g) / k ** 0.5).to("cuda", torch.bfloat16)
        row = {"shape": f"{n}x{k}->{m}"}
        outs = {}
        for lib in ("cublas", "hipblaslt"):
            try:
                pref(lib)
                row[lib] = cuda_ms(lambda: F.linear(x, w))["ms"]
                outs[lib] = F.linear(x, w)
            except Exception as exc:  # noqa: BLE001
                row[lib] = None
                row[f"{lib}_error"] = err(exc)
        if len(outs) == 2:
            row["rel_l2"] = rel_l2(outs["hipblaslt"], outs["cublas"])
        res.append(row)
        del x, w, outs
    try:
        pref(r["default"] or "default")
    except Exception:  # noqa: BLE001
        pass
    r["gemms"] = res
    out["v6"] = r


# ----------------------------------------------------------------------------------------------------------- P3
def p3(out: dict, checkout: Path) -> None:
    import torch

    sys.path.insert(0, str(checkout / "studio" / "backend"))
    r: dict = {}
    try:
        from core.inference import diffusion_flux3_qk_rope as Q

        r["wanted_on_this_host"] = bool(Q.wanted("cuda"))
        for wd in (torch.bfloat16, torch.float32):
            key = f"bf16_w{str(wd).split('.')[-1]}"
            try:
                with torch.no_grad():
                    r[key] = Q._check(torch.device("cuda", 0), torch.bfloat16, wd, 128, 1e-6)
            except Exception as exc:  # noqa: BLE001
                r[key] = {"error": err(exc), "tail": traceback.format_exc().splitlines()[-6:]}
    except Exception as exc:  # noqa: BLE001
        r["error"] = err(exc)
        r["tail"] = traceback.format_exc().splitlines()[-8:]
    out["p3"] = r


# ----------------------------------------------------------------------------------------------------------- P2
VAES = {
    "flux2_vae_1024": ("unsloth/flux-3-image-klein", "AutoencoderKLFlux2", (1, 32, 128, 128)),
    "wan22_vae_480p33": ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "AutoencoderKLWan", (1, 48, 9, 30, 52)),
}


def p2_child(vae_key: str, mode: str, out_path: Path) -> int:
    """One arm in a fresh process: load the VAE, decode fixed latents 3x (after one warm decode), save the output."""
    import torch

    r: dict = {"vae": vae_key, "mode": mode, "env": {k: os.environ.get(k) for k in
                                                  ("MIOPEN_FIND_MODE", "MIOPEN_FIND_ENFORCE", "MIOPEN_USER_DB_PATH")}}
    try:
        import diffusers

        repo, cls, shape = VAES[vae_key]
        if mode.startswith("imm") or mode in ("stock", "fused"):
            torch.backends.miopen.immediate = True
        r["immediate"] = bool(getattr(torch.backends.miopen, "immediate", None))
        dev, dt = ("cpu", torch.float32) if mode == "cpu" else ("cuda", torch.bfloat16)
        vae = getattr(diffusers, cls).from_pretrained(repo, subfolder = "vae", torch_dtype = dt,
                                                      token = False).to(dev).eval()
        if mode == "fused":
            sys.path.insert(0, os.environ["LEVERS_STUDIO_BACKEND"])
            from core.inference import diffusion_vae_fused as M

            M.is_rocm_torch = lambda _t: False  # the gate under test: everything else in runtime_ok still applies
            r["runtime_ok"] = bool(M.runtime_ok())
            fn = M.install_wan_vae if cls == "AutoencoderKLWan" else M.install_group_norm_vae
            r["patched"] = int(fn(vae, encoder = False))
        # cpu arm: fp32 decode of the SAME bf16-rounded latents, the reference that says which GPU arm is right
        z = torch.randn(*shape, generator = torch.Generator(device = "cpu").manual_seed(0)).to(torch.bfloat16)
        z = z.to(dev, dt)
        ts = []
        with torch.no_grad():
            for i in range(1 if dev == "cpu" else 4):
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                img = vae.decode(z).sample
                if dev == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        r["first_s"] = round(ts[0], 3)
        r["decode_s"] = round(statistics.median(ts[1:] or ts), 4)
        r["decode_all_s"] = [round(t, 4) for t in ts]
        r["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        torch.save(img.float().cpu(), out_path.with_suffix(".pt"))
        r["ok"] = True
    except Exception as exc:  # noqa: BLE001
        r["ok"] = False
        r["error"] = err(exc)
        r["tail"] = traceback.format_exc().splitlines()[-8:]
    out_path.write_text(json.dumps(r, indent = 1), encoding = "utf-8")
    return 0


def p2(out: dict, work: Path, timeout_s: float, out_dir_for_db: Path) -> None:
    import torch

    r: dict = {}
    for vae_key in VAES:
        db = work / f"miopen_db_{vae_key}"
        db.mkdir(parents = True, exist_ok = True)
        empty = work / f"miopen_empty_{vae_key}"
        empty.mkdir(parents = True, exist_ok = True)
        base = {k: v for k, v in os.environ.items() if not k.startswith("MIOPEN_")}
        arms = [("imm", {"MIOPEN_USER_DB_PATH": str(empty)}),
                ("imm_aa", {"MIOPEN_USER_DB_PATH": str(empty)}),
                ("find", {"MIOPEN_USER_DB_PATH": str(db), "MIOPEN_FIND_MODE": "1", "MIOPEN_FIND_ENFORCE": "4"}),
                ("imm_db", {"MIOPEN_USER_DB_PATH": str(db), "MIOPEN_LOG_LEVEL": "5"}),
                ("cpu", {"MIOPEN_USER_DB_PATH": str(empty)})]
        rows = {}
        for mode, env in arms:
            jp = work / f"p2_{vae_key}_{mode}.json"
            log = work / f"p2_{vae_key}_{mode}.log"
            t0 = time.time()
            try:
                with open(log, "w", encoding = "utf-8") as fh:
                    rc = subprocess.run([sys.executable, __file__, "--p2-child", vae_key, mode, str(jp)],
                                        env = {**base, **env}, stdout = fh, stderr = subprocess.STDOUT,
                                        timeout = timeout_s).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            row = json.loads(jp.read_text(encoding = "utf-8")) if jp.exists() else {"ok": False}
            row["rc"], row["wall_s"] = rc, round(time.time() - t0, 1)
            if mode == "imm_db":
                text = log.read_text(encoding = "utf-8", errors = "replace") if log.exists() else ""
                row["log_findb_lines"] = sum(1 for ln in text.splitlines() if "FindDb" in ln or "Find-db" in ln
                                             or "find_db" in ln.lower())
                row["log_immediate_lines"] = sum(1 for ln in text.splitlines() if "Immediate" in ln)
                row["log_tail"] = [ln for ln in text.splitlines() if "FindDb" in ln or "Immediate" in ln][:8]
                try:
                    log.unlink()  # LOG_LEVEL=5 logs are large
                except OSError:
                    pass
            rows[mode] = row
        try:
            ref = torch.load(work / f"p2_{vae_key}_imm.pt")
            for mode in ("imm_aa", "find", "imm_db"):
                p = work / f"p2_{vae_key}_{mode}.pt"
                if p.exists():
                    x = torch.load(p)
                    d = (x - ref).abs()
                    mse = float(((x.clamp(-1, 1) - ref.clamp(-1, 1)) / 2).pow(2).mean())
                    rows[mode]["vs_imm"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                                            "psnr_db": round(10 * __import__("math").log10(1 / max(mse, 1e-20)), 2),
                                            "lpips": lpips_of(x, ref)}
            cp = work / f"p2_{vae_key}_cpu.pt"
            if cp.exists():
                cref = torch.load(cp)
                for mode in ("imm", "imm_aa", "find", "imm_db"):
                    p = work / f"p2_{vae_key}_{mode}.pt"
                    if p.exists():
                        x = torch.load(p)
                        mse = float(((x.clamp(-1, 1) - cref.clamp(-1, 1)) / 2).pow(2).mean())
                        rows[mode]["vs_cpu_fp32"] = {
                            "max_abs": float((x - cref).abs().max()),
                            "psnr_db": round(10 * __import__("math").log10(1 / max(mse, 1e-20)), 2),
                            "lpips": lpips_of(x, cref)}
        except Exception as exc:  # noqa: BLE001
            rows["compare_error"] = err(exc)
        r[vae_key] = rows
        r[vae_key]["db_files"] = sorted(p.name for p in db.rglob("*") if p.is_file())
        keep = out_dir_for_db / vae_key  # the tuned db travels with the artifact (it is the lever to ship)
        keep.mkdir(parents = True, exist_ok = True)
        for f in db.rglob("*"):
            if f.is_file() and f.stat().st_size < 64 << 20:
                (keep / f.name).write_bytes(f.read_bytes())
    out["p2"] = r


def lpips_of(x, ref):
    try:
        import lpips
        import torch

        net = lpips.LPIPS(net = "alex", verbose = False)
        if x.dim() == 5:  # video [B, C, T, H, W]: frames as a batch
            x = x[0].permute(1, 0, 2, 3)
            ref = ref[0].permute(1, 0, 2, 3)
        with torch.no_grad():
            return float(net(x.clamp(-1, 1), ref.clamp(-1, 1)).mean())
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {err(exc)}"



# ------------------------------------------------------------------------------------------- VAE fused (repos row 5)
def gn_kernels(out: dict, checkout: Path) -> None:
    """Studio's Triton GroupNorm(+SiLU) vs nn.GroupNorm on VAE shapes, gate lifted in-process: parity and ms."""
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(checkout / "studio" / "backend"))
    from core.inference import diffusion_vae_fused as M

    M.is_rocm_torch = lambda _t: False
    r: dict = {"runtime_ok": bool(M.runtime_ok())}
    rows = []
    for c, h, w in ((512, 128, 128), (512, 256, 256), (256, 512, 512), (128, 1024, 1024)):
        norm = torch.nn.GroupNorm(32, c, eps = 1e-6).to("cuda", torch.bfloat16)
        with torch.no_grad():
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.2, 0.2)
        x = (torch.randn(1, c, h, w, generator = torch.Generator(device = "cpu").manual_seed(c + h)) * 2 + 0.3).to(
            "cuda", torch.bfloat16)
        row = {"shape": f"{c}x{h}x{w}"}
        try:
            with torch.no_grad():
                ref = F.silu(norm(x))
                got = M.group_norm_act(x, norm, True)
                ref32 = F.silu(F.group_norm(x.float(), 32, norm.weight.float(), norm.bias.float(), 1e-6))
                row["bitwise_vs_stock"] = bool(torch.equal(got.contiguous(), ref.contiguous()))
                row["max_abs_vs_stock"] = float((got.float() - ref.float()).abs().max())
                row["rel_l2_vs_fp32"] = rel_l2(got, ref32)
                row["stock_rel_l2_vs_fp32"] = rel_l2(ref, ref32)
                row["stock_ms"] = cuda_ms(lambda: F.silu(norm(x)), iters = 10)["ms"]
                row["fused_ms"] = cuda_ms(lambda: M.group_norm_act(x, norm, True), iters = 10)["ms"]
        except Exception as exc:  # noqa: BLE001
            row["error"] = err(exc)
        rows.append(row)
    r["group_norm_silu"] = rows
    out["vae_gn"] = r


def vae_fused(out: dict, work: Path, timeout_s: float, checkout: Path) -> None:
    """Decode with the fused VAE kernels (gate lifted) vs stock, fresh process per arm, same latents."""
    import torch

    r: dict = {}
    base = {k: v for k, v in os.environ.items() if not k.startswith("MIOPEN_")}
    base["LEVERS_STUDIO_BACKEND"] = str(checkout / "studio" / "backend")
    for vae_key in VAES:
        rows = {}
        for mode in ("stock", "fused", "stock_aa"):
            m = "stock" if mode == "stock_aa" else mode
            jp = work / f"vf_{vae_key}_{mode}.json"
            t0 = time.time()
            try:
                with open(work / f"vf_{vae_key}_{mode}.log", "w", encoding = "utf-8") as fh:
                    rc = subprocess.run([sys.executable, __file__, "--p2-child", vae_key, m, str(jp)], env = base,
                                        stdout = fh, stderr = subprocess.STDOUT, timeout = timeout_s).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            row = json.loads(jp.read_text(encoding = "utf-8")) if jp.exists() else {"ok": False}
            row["rc"], row["wall_s"] = rc, round(time.time() - t0, 1)
            pt = jp.with_suffix(".pt")
            if pt.exists():
                pt.rename(work / f"vf_{vae_key}_{mode}.out.pt")
            rows[mode] = row
        try:
            ref = torch.load(work / f"vf_{vae_key}_stock.out.pt")
            for mode in ("fused", "stock_aa"):
                p = work / f"vf_{vae_key}_{mode}.out.pt"
                if p.exists():
                    x = torch.load(p)
                    rows[mode]["vs_stock"] = {"max_abs": float((x - ref).abs().max()),
                                              "mean_abs": float((x - ref).abs().mean()), "lpips": lpips_of(x, ref)}
        except Exception as exc:  # noqa: BLE001
            rows["compare_error"] = err(exc)
        r[vae_key] = rows
    out["vae_fused"] = r


# ---------------------------------------------------------------------------------- mapping mode (repos row 10)
def _mem() -> dict:
    out = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.split(":")[0] in ("VmRSS", "RssAnon", "RssFile", "RssShmem"):
                out[line.split(":")[0]] = int(line.split()[1]) // 1024  # MiB
    except OSError:
        try:
            import psutil

            mi = psutil.Process().memory_info()
            out["rss"] = mi.rss >> 20
            out["private"] = getattr(mi, "private", 0) >> 20
        except Exception:  # noqa: BLE001
            pass
    return out


def mm_child(path: str, mode: str, out_path: Path) -> int:
    import mmap

    import torch

    r: dict = {"mode": mode, "before": _mem()}
    try:
        size = os.path.getsize(path)
        chunk = 256 << 20
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_COPY if mode == "copy" else mmap.ACCESS_READ)
        import warnings

        warnings.simplefilter("ignore")  # frombuffer on a read-only map warns once; reads are what is measured
        for rep in ("cold", "warm"):
            dev = []
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for off in range(0, size, chunk):
                n = min(chunk, size - off)
                dev.append(torch.frombuffer(mm, dtype = torch.uint8, count = n, offset = off).to("cuda"))
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            r[f"{rep}_s"] = round(dt, 3)
            r[f"{rep}_gib_s"] = round(size / dt / 2 ** 30, 2)
            r[f"after_{rep}"] = _mem()
            del dev
            torch.cuda.empty_cache()
        del mm
        import gc

        gc.collect()
        r["after_close"] = _mem()
        r["ok"] = True
    except Exception as exc:  # noqa: BLE001
        r["ok"] = False
        r["error"] = err(exc)
    out_path.write_text(json.dumps(r, indent = 1), encoding = "utf-8")
    return 0


def mapping(out: dict, work: Path) -> None:
    """Host->device copy speed and process memory for ACCESS_COPY (Studio today) vs ACCESS_READ, fresh process each,
    over a 4 GiB file (both arms read the same file; the first arm also pays the cold page cache, so the order is
    read, copy, read and the warm figures are compared)."""
    path = work / "mm_4g.bin"
    if not path.exists() or path.stat().st_size != 4 << 30:
        with open(path, "wb") as f:
            block = os.urandom(64 << 20)
            for _ in range(64):
                f.write(block)
    rows = []
    for mode in ("read", "copy", "read", "copy"):
        jp = work / f"mm_{len(rows)}_{mode}.json"
        subprocess.run([sys.executable, __file__, "--mm-child", str(path), mode, str(jp)], timeout = 1200)
        rows.append(json.loads(jp.read_text(encoding = "utf-8")) if jp.exists() else {"mode": mode, "ok": False})
    out["mapping"] = {"file_gib": 4, "runs": rows}
    try:
        path.unlink()
    except OSError:
        pass

def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--p2-child":
        return p2_child(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    if len(sys.argv) > 1 and sys.argv[1] == "--mm-child":
        return mm_child(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--work", type = Path, default = None)
    ap.add_argument("--skip", default = "")
    ap.add_argument("--p2-timeout-s", type = float, default = 2400)
    a = ap.parse_args()
    work = a.work or a.out.parent / "levers_work"
    work.mkdir(parents = True, exist_ok = True)
    skip = set(filter(None, a.skip.split(",")))
    obs: dict = {"host": {"platform": platform.platform(), "python": sys.version.split()[0]}, "errors": {}}

    def flush():
        a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")

    try:
        import torch

        p = torch.cuda.get_device_properties(0)
        obs["host"].update(torch = torch.__version__, hip = getattr(torch.version, "hip", None),
                           arch = str(getattr(p, "gcnArchName", "")), device = p.name,
                           miopen_immediate_attr = hasattr(getattr(torch.backends, "miopen", None), "immediate"))
    except Exception as exc:  # noqa: BLE001
        obs["errors"]["host"] = err(exc)
    flush()
    for name, fn in (("p1", lambda: p1(obs)), ("v6", lambda: v6(obs)), ("p3", lambda: p3(obs, a.checkout)),
                     ("vae_gn", lambda: gn_kernels(obs, a.checkout)),
                     ("vae_fused", lambda: vae_fused(obs, work, a.p2_timeout_s, a.checkout)),
                     ("mapping", lambda: mapping(obs, work)),
                     ("p2", lambda: p2(obs, work, a.p2_timeout_s, a.out.parent / "miopen_db"))):
        if name in skip:
            obs[name] = {"_skipped": True}
            continue
        t0 = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - one section failing must not hide the others
            obs["errors"][name] = err(exc)
            obs.setdefault(name, {})["tail"] = traceback.format_exc().splitlines()[-8:]
        obs.setdefault(name, {})["_elapsed_s"] = round(time.time() - t0, 1)
        flush()
    for pt in work.glob("*.pt"):
        try:
            pt.unlink()
        except OSError:
            pass
    flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
