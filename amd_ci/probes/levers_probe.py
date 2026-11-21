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
        if mode.startswith("imm"):
            torch.backends.miopen.immediate = True
        r["immediate"] = bool(getattr(torch.backends.miopen, "immediate", None))
        vae = getattr(diffusers, cls).from_pretrained(repo, subfolder = "vae", torch_dtype = torch.bfloat16,
                                                      token = False).to("cuda").eval()
        z = torch.randn(*shape, generator = torch.Generator(device = "cpu").manual_seed(0)).to("cuda", torch.bfloat16)
        ts = []
        with torch.no_grad():
            for i in range(4):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                img = vae.decode(z).sample
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
        r["first_s"] = round(ts[0], 3)
        r["decode_s"] = round(statistics.median(ts[1:]), 4)
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


def p2(out: dict, work: Path, timeout_s: float) -> None:
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
                ("imm_db", {"MIOPEN_USER_DB_PATH": str(db), "MIOPEN_LOG_LEVEL": "5"})]
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
        except Exception as exc:  # noqa: BLE001
            rows["compare_error"] = err(exc)
        r[vae_key] = rows
        r[vae_key]["db_files"] = sorted(p.name for p in db.rglob("*") if p.is_file())
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


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--p2-child":
        return p2_child(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
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
                     ("p2", lambda: p2(obs, work, a.p2_timeout_s))):
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
