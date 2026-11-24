#!/usr/bin/env python3
"""AMD coverage Nov 24 (M05 + M10). Observes only; numbers are judged in the report.

  img  --arms FILE   one fresh process per arm (probes/m24/img_arm.py): Studio load + renders, timing, peak, PNGs
  vae  --caps DIR    M05: FLUX.2-family VAE decodes of real latents captured from a Studio FLUX.2 klein render
                     (plus the Nov 21 random latent) on the GPU in fp32 / bf16, stock and Studio-routed, against a CPU
                     fp32 reference (and CPU bf16), with per-module local-error hooks; LPIPS / PSNR / max abs.
Public repos only; every token variable is stripped from the children."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")


def child_env(extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in TOKEN_VARS + ("PYTHONPATH",)}
    env.update({"PYTHONUNBUFFERED": "1", "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"})
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


def run(cmd: list, log: Path, env: dict, timeout: float, cwd: str | None = None):
    t0 = time.time()
    with open(log, "w", encoding = "utf-8") as fh:
        try:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env, timeout = timeout, cwd = cwd).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    return rc, round(time.time() - t0, 1)


def tail(p: Path, n: int = 30) -> list:
    try:
        return p.read_text(encoding = "utf-8", errors = "replace").splitlines()[-n:]
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------------------------------------------- img
def cmd_img(a) -> dict:
    arms = json.loads(Path(a.arms).read_text(encoding = "utf-8"))
    work = Path(a.work)
    work.mkdir(parents = True, exist_ok = True)
    res: dict = {"arms": {}, "order": []}
    for arm in arms:
        tag = arm["tag"]
        ind = work / "ind" / tag
        shutil.rmtree(ind, ignore_errors = True)
        env = child_env({"TORCHINDUCTOR_CACHE_DIR": str(ind), **(arm.get("env") or {})})
        cmd = [sys.executable, "-u", str(HERE / "m24" / "img_arm.py"), "--root", a.backend, "--model", arm["model"],
               "--tag", tag, "--out", str(Path(a.out_dir)), "--load", json.dumps(arm.get("load") or {}),
               "--gen", json.dumps(arm.get("gen") or {}), "--renders", str(arm.get("renders", 2)),
               "--prompt-schedule", ",".join(str(x) for x in arm.get("prompt_schedule") or [])]
        if arm.get("save_renders"):
            cmd += ["--save-renders", ",".join(str(x) for x in arm["save_renders"])]
        if arm.get("noise_fp32"):
            cmd.append("--noise-fp32")
        if arm.get("capture_vae"):
            cmd.append("--capture-vae")
        log = Path(a.out_dir) / f"{tag}.log"
        Path(a.out_dir).mkdir(parents = True, exist_ok = True)
        rc, wall = run(cmd, log, env, arm.get("timeout", a.arm_timeout), cwd = str(work))
        rp = Path(a.out_dir) / tag / "record.json"
        r = json.loads(rp.read_text(encoding = "utf-8")) if rp.is_file() else {"verdict": "missing"}
        r["rc"], r["arm_s"], r["spec"] = rc, wall, arm
        if r.get("verdict") != "ok":
            r["log_tail"] = tail(log, 60)
        res["arms"][tag] = r
        res["order"].append(tag)
        print(f"ARM {tag} rc={rc} {r.get('verdict')} {wall}s "
              f"sps={[x.get('s_per_step') for x in r.get('renders', [])]}", flush = True)
        Path(a.out).write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")
    return res


# ---------------------------------------------------------------------------------------------------------------- vae
VAE_REPOS = {
    "flux2": "unsloth/FLUX.2-klein-4B",          # the FLUX.2 klein VAE (byte-identical to black-forest-labs')
    "flux3": "unsloth/flux-3-image-klein",       # FLUX.3 klein's VAE file (the Nov 21 test's)
    "moss": "unsloth/MOSS-Image-SFT-FP8",        # MOSS-Image small-decoder FLUX.2 VAE
}


def lpips_net():
    try:
        import lpips

        return lpips.LPIPS(net = "alex", verbose = False)
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc!r}"[:300]


def metrics(x, ref, net) -> dict:
    import torch

    x, ref = x.float(), ref.float()
    mse = float(((x.clamp(-1, 1) - ref.clamp(-1, 1)) / 2).pow(2).mean())
    out = {"max_abs": float((x - ref).abs().max()), "mean_abs": float((x - ref).abs().mean()),
           "psnr_db": round(10 * math.log10(1 / max(mse, 1e-20)), 2), "finite": bool(torch.isfinite(x).all())}
    if isinstance(net, str):
        out["lpips"] = net
    else:
        with torch.no_grad():
            out["lpips"] = float(net(x.clamp(-1, 1), ref.clamp(-1, 1)).mean())
    return out


def cmd_vae(a) -> dict:
    import torch

    caps = sorted(Path(a.caps).glob("vae_cap_r*_0.pt"))
    work = Path(a.work)
    work.mkdir(parents = True, exist_ok = True)
    od = Path(a.out_dir)
    od.mkdir(parents = True, exist_ok = True)
    res: dict = {"caps": [c.name for c in caps], "arms": {}, "scores": {}}
    lat = [[f"real{i}", str(c)] for i, c in enumerate(caps[:2])]
    for name, p in lat:
        d = torch.load(p, map_location = "cpu")
        res.setdefault("latent_info", {})[name] = {"shape": list(d["z"].shape), "dtype_at_decode": d.get("z_dtype"),
                                                   "std": float(d["z"].std()), "absmax": float(d["z"].abs().max()),
                                                   "prompt": d.get("prompt"), "model": d.get("model")}
    rand = [["rand0", "rand:0"]]
    real0 = lat[:1]
    small = a.small
    allv = lat + rand
    G = {"MIOPEN_DEBUG_CONV_GEMM": "0"}
    plan = [  # (vae key, mode, tag suffix, latents, extra env, local_err)
        ("flux2", "cpu_fp32", "", allv, {}, False),
        ("flux2", "cpu_bf16", "", real0 + rand, {}, False),
        ("flux2", "gpu_raw_bf16", "", allv, {}, False),
        ("flux2", "gpu_raw_fp32", "", allv, {}, False),
        ("flux2", "gpu_studio_bf16", "", allv, {}, False),
        ("flux2", "gpu_studio_fp32", "", allv, {}, False),
        ("flux2", "gpu_math_bf16", "", allv, {}, False),
        ("flux2", "gpu_studio_bf16", "_nogemm", allv, G, False),
        ("flux2", "gpu_raw_bf16", "_nogemm", allv, G, False),
        ("flux2", "gpu_raw_bf16", "_local", real0 + rand, {}, True),
        ("flux2", "gpu_studio_bf16", "_local", real0 + rand, {}, True),
        ("flux2", "gpu_studio_fp32", "_local", real0, {}, True),
        ("flux3", "cpu_fp32", "", real0 + rand, {}, False),
        ("flux3", "gpu_raw_bf16", "", real0 + rand, {}, False),
        ("flux3", "gpu_studio_bf16", "", real0 + rand, {}, False),
        ("moss", "cpu_fp32", "", real0 + rand, {}, False),
        ("moss", "gpu_raw_bf16", "", real0 + rand, {}, False),
        ("moss", "gpu_studio_bf16", "", real0 + rand, {}, False),
    ]
    if small:  # plumbing probe: one real latent, a handful of arms
        plan = [p for p in plan if p[0] == "flux2" and p[2] in ("", "_local") and p[1] in
                ("cpu_fp32", "gpu_raw_bf16", "gpu_studio_bf16")]
        plan = [(k, m, s, real0 if le else real0 + rand, e, le) for k, m, s, _, e, le in plan]
        plan = [p for p in plan if not (p[5] and p[1] == "gpu_raw_bf16")]
    empty_db = work / "miopen_user_db_empty"
    empty_db.mkdir(parents = True, exist_ok = True)
    for key, mode, suf, lats, extra, le in plan:
        if not lats:
            continue
        tag = f"{key}_{mode}{suf}"
        env = child_env({"MIOPEN_USER_DB_PATH": str(empty_db), **extra})
        env = {k: v for k, v in env.items() if not (k.startswith("MIOPEN_DEBUG") and k not in extra)}
        cmd = [sys.executable, "-u", str(HERE / "m24" / "vae_arm.py"), "--repo", VAE_REPOS[key], "--mode", mode,
               "--tag", tag, "--latents", json.dumps(lats), "--out", str(work / "dec"), "--studio-backend", a.backend]
        if le:
            cmd.append("--local-err")
        rc, wall = run(cmd, od / f"{tag}.log", env, a.arm_timeout)
        jp = work / "dec" / f"{tag}.json"
        r = json.loads(jp.read_text(encoding = "utf-8")) if jp.is_file() else {"ok": False, "missing": True}
        r["rc"], r["wall_s"], r["env_set"], r["vae_key"] = rc, wall, extra, key
        if not r.get("ok"):
            r["log_tail"] = tail(od / f"{tag}.log", 40)
        res["arms"][tag] = r
        print(f"VAE {tag} rc={rc} ok={r.get('ok')} {wall}s", flush = True)
        Path(a.out).write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")
    net = lpips_net()
    res["lpips_net"] = "alex" if not isinstance(net, str) else net
    dec = work / "dec"
    for tag, r in res["arms"].items():
        key = r.get("vae_key")
        for lname in (r.get("latents") or {}):
            ref_p = dec / f"{key}_cpu_fp32__{lname}.pt"
            x_p = dec / f"{tag}__{lname}.pt"
            if tag == f"{key}_cpu_fp32" or not ref_p.is_file() or not x_p.is_file():
                continue
            res["scores"][f"{tag}|{lname}"] = metrics(torch.load(x_p), torch.load(ref_p), net)
    # Studio's own in-render decode (captured from the klein render) vs the CPU fp32 decode of the same latent
    for name, p in lat:
        ref_p = dec / f"flux2_cpu_fp32__{name}.pt"
        if ref_p.is_file():
            so = torch.load(p, map_location = "cpu")["studio_out"]
            res["scores"][f"studio_inrender|{name}"] = metrics(so, torch.load(ref_p), net)
    # keep a compact copy of every decode for offline rechecks (fp16)
    keep = od / "decodes_fp16"
    keep.mkdir(parents = True, exist_ok = True)
    for p in dec.glob("*.pt"):
        torch.save(torch.load(p).half(), keep / p.name)
    Path(a.out).write_text(json.dumps(res, indent = 1, default = str), encoding = "utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices = ("img", "vae"))
    ap.add_argument("--backend", required = True, help = "studio/backend of the tree under test")
    ap.add_argument("--out", required = True, help = "observation JSON")
    ap.add_argument("--out-dir", required = True, help = "artifact dir (records, PNGs, logs)")
    ap.add_argument("--work", required = True)
    ap.add_argument("--arms", default = "")
    ap.add_argument("--caps", default = "")
    ap.add_argument("--small", action = "store_true")
    ap.add_argument("--arm-timeout", type = float, default = 5400)
    a = ap.parse_args()
    for k in TOKEN_VARS:
        os.environ.pop(k, None)
    (cmd_img if a.cmd == "img" else cmd_vae)(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
