#!/usr/bin/env python3
"""Probe (gfx1151): FLUX.2 VAE whole-map strip decode (diffusion_vae_strips.full_map_decode) at 1024 and 2048 px,
upsample fold off / on, fresh process per arm, same latents.

Per arm: decode seconds (median of 3 after one warm decode), peak allocated during the decodes above what was allocated
before them (torch.cuda.max_memory_allocated after reset_peak_memory_stats), the tree's own estimate
full_map_decode_mib(vae, z), fold calls and conv_transpose2d calls (whole map = 1 per fold, strips = more).
Arms: off (UNSLOTH_VAE_UPSAMPLE_FOLD=0) and auto (unset: the fold arms on gfx1151), ABBA; on a tree that has
_fold_full_max also "whole" (its fold limit forced back to the whole map, the base behaviour, as an in-tree control).
Outputs go to $AMD_CI_WORK/vaefold/<state>_<res>_<arm>.pt; the head compares its outputs with the base's (max abs, PSNR).
Observes only (criteria/vae_fold_peak.py judges). JSON to --out, never stdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = "unsloth/flux-3-image-klein"
RES = (1024, 2048)


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent = 2, default = str), encoding = "utf-8")


def child(res: int, arm: str, backend: str, out_json: Path, out_pt: Path) -> int:
    r: dict = {"res": res, "arm": arm, "fold_env": os.environ.get("UNSLOTH_VAE_UPSAMPLE_FOLD")}
    try:
        import torch
        import torch.nn.functional as F

        sys.path.insert(0, backend)
        import diffusers
        from core.inference import diffusion_vae_strips as S

        r["module"] = S.__file__
        r["has_full_max"] = hasattr(S, "_fold_full_max")
        if arm == "whole":
            if not r["has_full_max"]:
                raise RuntimeError("whole arm needs _fold_full_max")
            S._fold_full_max = lambda _x: None
        seen = [0]
        real = F.conv_transpose2d

        def counted(*a, **k):
            seen[0] += 1
            return real(*a, **k)

        F.conv_transpose2d = counted
        vae = diffusers.AutoencoderKLFlux2.from_pretrained(
            REPO, subfolder = "vae", torch_dtype = torch.bfloat16, token = False
        ).to("cuda").eval()
        lat = res // 8
        z = torch.randn(1, 32, lat, lat, generator = torch.Generator(device = "cpu").manual_seed(0))
        z = z.to(torch.bfloat16).to("cuda")
        r["estimate_mib"] = float(S.full_map_decode_mib(vae, z))
        r["fold_on"] = bool(S._fold_on(z))
        f0 = int(S._FOLD_CALLS[0])
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        ts, img = [], None
        with torch.no_grad():
            for i in range(4):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = S.full_map_decode(vae, z)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
                if i == 0:
                    img = out.float().cpu()
                del out
        peak = torch.cuda.max_memory_allocated()
        r["before_mib"] = round(before / 2**20, 1)
        r["peak_total_mib"] = round(peak / 2**20, 1)
        r["peak_delta_mib"] = round((peak - before) / 2**20, 1)
        r["fold_calls"] = int(S._FOLD_CALLS[0]) - f0
        r["conv_transpose_calls"] = seen[0]
        r["first_s"] = round(ts[0], 4)
        r["decode_s"] = round(statistics.median(ts[1:]), 4)
        r["decode_all_s"] = [round(t, 4) for t in ts]
        r["md5"] = hashlib.md5(img.numpy().tobytes()).hexdigest()
        r["finite"] = bool(torch.isfinite(img).all())
        torch.save(img, out_pt)
        r["ok"] = True
    except Exception as exc:  # noqa: BLE001
        r["ok"] = False
        r["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        r["tail"] = traceback.format_exc().splitlines()[-10:]
    _write(out_json, r)
    return 0


def _cmp(a: Path, b: Path) -> dict:
    import torch

    x, y = torch.load(a), torch.load(b)
    if x.shape != y.shape:
        return {"error": f"shape {tuple(x.shape)} vs {tuple(y.shape)}"}
    d = (x.clamp(-1, 1) - y.clamp(-1, 1)).double()
    mse = float((d * d).mean())
    return {"max_abs": float(d.abs().max()), "psnr_db": 99.0 if mse == 0 else round(10 * math.log10(4.0 / mse), 2),
            "identical": bool(torch.equal(x, y))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state")
    ap.add_argument("--checkout")
    ap.add_argument("--out", type = Path)
    ap.add_argument("--child", nargs = 5, default = None)
    ap.add_argument("--timeout", type = float, default = 1200)
    args = ap.parse_args()
    if args.child:
        res, arm, backend, oj, op = args.child
        return child(int(res), arm, backend, Path(oj), Path(op))

    import torch

    res: dict = {"state": args.state, "checkout": args.checkout, "torch": torch.__version__,
                 "hip": getattr(torch.version, "hip", None), "platform": sys.platform, "cuda": torch.cuda.is_available(),
                 "arms": {}}
    if torch.cuda.is_available():
        res["arch"] = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or "")
    _write(args.out, res)
    if not torch.cuda.is_available():
        return 0
    work = Path(os.environ.get("AMD_CI_WORK") or args.out.parent) / "vaefold"
    work.mkdir(parents = True, exist_ok = True)
    backend = str(Path(args.checkout).resolve() / "studio" / "backend")
    has_full_max = "_fold_full_max" in (Path(backend) / "core" / "inference" / "diffusion_vae_strips.py").read_text(
        encoding = "utf-8")
    env0 = {k: v for k, v in os.environ.items() if k != "UNSLOTH_VAE_UPSAMPLE_FOLD" and k != "CI"}
    try:
        for rz in RES:
            order = ["off", "auto", "auto", "off"]
            if has_full_max and rz == 1024:
                order = ["off", "auto", "whole", "whole", "auto", "off"]
            for i, arm in enumerate(order):
                name = f"{rz}_{arm}_{i}"
                oj, op = work / f"{args.state}_{name}.json", work / f"{args.state}_{name}.pt"
                env = dict(env0)
                if arm == "off":
                    env["UNSLOTH_VAE_UPSAMPLE_FOLD"] = "0"
                t0 = time.time()
                lg = work / f"{args.state}_{name}.log"
                try:
                    with open(lg, "w", encoding = "utf-8") as fh:
                        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(rz), arm,
                                             backend, str(oj), str(op)], env = env, stdout = fh,
                                            stderr = subprocess.STDOUT, timeout = args.timeout).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
                row = json.loads(oj.read_text(encoding = "utf-8")) if oj.exists() else {"ok": False}
                row["rc"], row["wall_s"] = rc, round(time.time() - t0, 1)
                if not row.get("ok"):
                    row["log_tail"] = lg.read_text(encoding = "utf-8", errors = "replace")[-2000:] if lg.exists() else None
                res["arms"][name] = row
                _write(args.out, res)
        # in-state comparisons: fold vs no fold, A/A
        cmp: dict = {}
        for rz in RES:
            first = {}
            for name in res["arms"]:
                if name.startswith(f"{rz}_"):
                    arm = name.split("_")[1]
                    first.setdefault(arm, name)
            p = lambda n: work / f"{args.state}_{n}.pt"
            if "off" in first and "auto" in first:
                cmp[f"{rz}_auto_vs_off"] = _cmp(p(first["auto"]), p(first["off"]))
            if "whole" in first and "auto" in first:
                cmp[f"{rz}_auto_vs_whole"] = _cmp(p(first["auto"]), p(first["whole"]))
            # vs the base state's renders of the same arm (written when the base probe ran first)
            if args.state != "base":
                for arm, n in first.items():
                    bn = "auto" if arm == "whole" else arm
                    cands = sorted(work.glob(f"base_{rz}_{bn}_*.pt"))
                    if cands:
                        cmp[f"{rz}_{arm}_vs_base_{bn}"] = _cmp(p(n), cands[0])
        res["compare"] = cmp
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
        res["tb"] = traceback.format_exc()[-3000:]
    _write(args.out, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
