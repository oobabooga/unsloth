#!/usr/bin/env python3
"""Probe (WINDOWS leg): PR 19873 (vendored Triton conv3d for the Wan VAE on ROCm gfx11), Wan 2.2 TI2V-5B VAE decode, bf16.
Windows variant of amd_overlay_19873/probes/c3d_probe.py: arms A / C / D only (no old head), head tree = the PR
checkout itself (sha checked by the criteria), fewer repeats: warm A1 A2 C1 D1 C2 D2 A3, cold A1 C1 C2 D1.

Observes only; criteria/c3d_measure.py judges. Runs from the head step (other states record a skip): every arm is a
fresh process of probes/c3d/worker.py on one tree:
  A = merge base (sibling `base` checkout, MIOpen; no router there)
  B = old head 541f66e1a6 (git worktree from the head checkout; Winograd on by default)
  C = new head, router env unset (direct kernels only)
  D = new head, UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D_WINOGRAD=1
Order: VAE fetch (unsloth/Wan2.2-TI2V-5B-FP8 vae/ only, token=False), smoke A B C D (tiny latent; gates the rest),
warm A1 A2 A3 (A/A spread first) then C1 D1 D2 C2 C3 D3 then A4 (drift control), acc (per-conv error, one
process), cold B1 C1 C2 B2 (each with an empty TRITON_CACHE_DIR). No token is read or passed.

  python c3d_probe.py --state head --checkout states/head --out obs_head.json [--frames 81] [--items ...]
"""
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
WORKER = HERE / "c3d" / "worker.py"
OLD_HEAD = "541f66e1a6"
# C / D pinned to the fix commit (the PR head moved on to a memory-only _prepack.py change)
NEW_HEAD = "15c1a7ff963570f91facbb0af6bf7942d949ba8b"
VAE_REPO = "unsloth/Wan2.2-TI2V-5B-FP8"
WINO = "UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D_WINOGRAD"
ARM_ENV = {"A": {}, "C": {}, "D": {WINO: "1"}}
SECRET = ("TOKEN", "SECRET", "PASSWORD", "WANDB_API_KEY")
WARM_PLAN = ["A1", "A2", "C1", "D1", "C2", "D2", "A3"]
COLD_PLAN = ["A1", "C1", "C2", "D1"]


def child_env(heavy: Path, extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in SECRET) and not k.startswith("UNSLOTH_DIFFUSION_ROCM")
           and k not in ("PYTHONPATH", "TRITON_CACHE_DIR")}
    env.update({"HF_HOME": str(heavy / "hf"), "PYTHONUNBUFFERED": "1", "UNSLOTH_DISABLE_AUTO_UPDATES": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONIOENCODING": "utf-8",
                "TRITON_CACHE_DIR": str(heavy / "triton_warm")})
    env.update(extra)
    return env


def git(*args, timeout = 900):
    r = subprocess.run(["git", *map(str, args)], capture_output = True, text = True, encoding = "utf-8",
                       errors = "replace", timeout = timeout,
                       env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in SECRET)})
    return r.returncode, (r.stdout + r.stderr).strip()


def worker(ctx, tree: Path, mode: str, tag: str, env_extra: dict, timeout: int, extra = ()) -> dict:
    out = ctx["work"] / f"{mode}_{tag}.json"
    log = ctx["work"] / f"{mode}_{tag}.log"
    cmd = [sys.executable, "-u", str(WORKER), "--backend", str(tree / "studio" / "backend"), "--vae-dir",
           str(ctx["vae"]), "--mode", mode, "--out", str(out), "--frames", str(ctx["frames"]), *extra]
    t = time.time()
    try:
        with open(log, "w", encoding = "utf-8") as fh:
            rc = subprocess.run(cmd, cwd = tree / "studio" / "backend", stdout = fh, stderr = subprocess.STDOUT,
                                env = child_env(ctx["heavy"], env_extra), timeout = timeout).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    rec: dict = {"rc": rc, "proc_s": round(time.time() - t, 1), "tree": tree.name, "env_extra": env_extra}
    try:
        rec.update(json.loads(out.read_text(encoding = "utf-8")))
    except Exception as exc:  # noqa: BLE001
        rec["missing"] = repr(exc)[:200]
    text = log.read_text(encoding = "utf-8", errors = "replace") if log.is_file() else ""
    rec["log_tail"] = text[-2000:]
    rec["router_lines"] = [x[:400] for x in text.splitlines() if "rocm_triton_conv3d" in x][:20]
    return rec


def psnr(a: Path, b: Path):
    import torch
    if not (a.is_file() and b.is_file()):
        return None
    x, y = torch.load(a).float(), torch.load(b).float()
    if x.shape != y.shape:
        return None
    mse = float(((x - y) ** 2).mean())
    return float("inf") if mse == 0 else round(10 * math.log10(4.0 / mse), 2)


def fetch_vae(ctx, obs):
    d = ctx["heavy"] / "wan_vae"
    code = ("import sys, time\nfrom huggingface_hub import snapshot_download\n"
            "for i in range(5):\n"
            "    try:\n"
            "        snapshot_download(sys.argv[1], allow_patterns=['vae/*', 'model_index.json'], local_dir=sys.argv[2],"
            " token=False, max_workers=8); break\n"
            "    except Exception as e:\n"
            "        print('retry', i, repr(e)[:300], flush=True); time.sleep(20)\n"
            "else:\n    raise SystemExit(3)\n")
    t = time.time()
    r = subprocess.run([sys.executable, "-c", code, VAE_REPO, str(d)], capture_output = True, text = True,
                       encoding = "utf-8", errors = "replace", timeout = 3600, env = child_env(ctx["heavy"], {}))
    info = {"repo": VAE_REPO, "rc": r.returncode, "s": round(time.time() - t, 1), "tail": (r.stdout + r.stderr)[-800:]}
    try:
        info["config"] = json.loads((d / "vae" / "config.json").read_text(encoding = "utf-8"))
        st = d / "vae" / "diffusion_pytorch_model.safetensors"
        with open(st, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            hdr = json.loads(fh.read(n))
        info["stored_dtypes"] = sorted({v["dtype"] for k, v in hdr.items() if k != "__metadata__"})
        info["bytes"] = st.stat().st_size
    except Exception as exc:  # noqa: BLE001
        info["inspect_error"] = repr(exc)[:300]
    obs["vae_fetch"] = info
    return d if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--frames", type = int, default = 81)
    ap.add_argument("--items", default = "smoke,warm,acc,cold")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    obs: dict = {"state": a.state, "args": vars(a)}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    try:
        import torch
        obs["env"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                      "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
                      "python": sys.version.split()[0]}
        import triton
        obs["env"]["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)[:500]
    if a.state != "head":
        obs["skipped"] = "measured from the head step only (base and old-head trees run from there)"
        save()
        return 0
    head = Path(a.checkout).resolve()
    work = out.parent / "c3dwork"
    work.mkdir(parents = True, exist_ok = True)
    # big files (weights, Triton caches, outputs, the old-head tree) stay out of the uploaded out/ dir
    heavy = Path(os.environ.get("AMD_CI_WORK") or work.parent.parent) / "c3dheavy"
    heavy.mkdir(parents = True, exist_ok = True)
    base = head.parent / "base"
    trees = {"A": base, "C": head, "D": head}
    obs["shas"] = {k: git("-C", p, "rev-parse", "HEAD")[1] for k, p in (("base", base), ("head", head))}
    obs["head_log"] = git("-C", head, "log", "--oneline", "-4")[1]
    obs["head_has_fix"] = git("-C", head, "merge-base", "--is-ancestor", NEW_HEAD, "HEAD")[0] == 0
    obs["platform"] = sys.platform
    save()
    ctx = {"work": work, "heavy": heavy, "frames": a.frames}
    ctx["vae"] = fetch_vae(ctx, obs)
    save()
    if ctx["vae"] is None:
        obs["fatal"] = "VAE fetch failed"
        save()
        return 0
    items = [x for x in a.items.split(",") if x]
    t_all = time.time()
    if "smoke" in items:
        obs["smoke"] = {}
        for arm in ("A", "C", "D"):
            obs["smoke"][arm] = worker(ctx, trees[arm], "smoke", arm, ARM_ENV[arm], 1500)
            save()
        sm = obs["smoke"]

        def routed(arm):
            ds = sm[arm].get("decodes") or []
            return sum((d.get("counts") or {}).get("routed", 0) for d in ds)
        obs["smoke_routed"] = {arm: routed(arm) for arm in sm}
        obs["smoke_ok"] = bool(all(sm[x].get("rc") == 0 and not sm[x].get("error") and sm[x].get("decodes")
                                   for x in sm) and obs["smoke_routed"]["C"] > 0 and obs["smoke_routed"]["A"] == 0)
        save()
        if not obs["smoke_ok"]:
            obs["fatal"] = "smoke failed: big items skipped"
            save()
            return 0
    if "warm" in items:
        obs["warm"] = {}
        for tag in WARM_PLAN:
            arm = tag[0]
            sv = heavy / f"y720_{tag}.pt"
            obs["warm"][tag] = worker(ctx, trees[arm], "warm", tag, ARM_ENV[arm], 3600,
                                      ["--save", str(sv)] if tag in ("A1", "A2", "C1", "D1") else [])
            obs["warm"][tag]["arm"] = arm
            save()
        obs["psnr720_vs_A1"] = {t: psnr(heavy / "y720_A1.pt", heavy / f"y720_{t}.pt") for t in ("A2", "C1", "D1")}
        save()
    if "acc" in items:
        obs["acc"] = worker(ctx, head, "acc", "CD", {}, 3600)
        save()
    if "cold" in items:
        obs["cold"] = {}
        for tag in COLD_PLAN:
            arm = tag[0]
            cache = heavy / f"triton_cold_{tag}"
            shutil.rmtree(cache, ignore_errors = True)
            cache.mkdir(parents = True)
            obs["cold"][tag] = worker(ctx, trees[arm], "cold", tag, dict(ARM_ENV[arm], TRITON_CACHE_DIR = str(cache)),
                                      4800)
            obs["cold"][tag]["arm"] = arm
            shutil.rmtree(cache, ignore_errors = True)
            save()
    obs["items_s"] = round(time.time() - t_all, 1)
    for sub in ("wan_vae", "hf", "triton_warm"):
        shutil.rmtree(heavy / sub, ignore_errors = True)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
