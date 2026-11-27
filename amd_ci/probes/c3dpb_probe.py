#!/usr/bin/env python3
"""Probe: Studio ROCm conv3d prebuild (branch studio-rocm-conv3d-prebuild = main + patch), Wan 2.2 TI2V-5B VAE
decode, bf16, gfx1151, Linux and Windows (same file).

Observes only; criteria/c3dpb.py judges. Runs from the head step (other states record a skip); every arm is a fresh
process of probes/c3dpb/worker.py:
  A = base (main): Triton conv3d routed, compiled inline on a cold cache
  B = head (main + patch): cold calls on MIOpen until the prebuild child has built the variant
  M = base with UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D=0 (MIOpen reference)
Order: VAE fetch (token=False), smoke A B (tiny latent, each on its own empty Triton cache, which then serves as
that arm's warm cache), coldflow A B (each on a fresh empty cache: first 480p decode right after load, wait for the
child, second 480p, first 720p), warm M1 A1 B1 M2 A2 B2 (M at 480 only; full-output sha256 per timed decode).
No token is read or passed.

  python c3dpb_probe.py --state head --checkout states/head --out obs_head.json [--frames 81] [--items ...]
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
WORKER = HERE / "c3dpb" / "worker.py"
VAE_REPO = "unsloth/Wan2.2-TI2V-5B-FP8"
ARM_ENV = {"A": {}, "B": {}, "M": {"UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D": "0"}}
SECRET = ("TOKEN", "SECRET", "PASSWORD", "WANDB_API_KEY")
WARM_PLAN = ["M1", "A1", "B1", "M2", "A2", "B2"]


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
    ap.add_argument("--items", default = "smoke,cold,warm")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    obs: dict = {"state": a.state, "args": vars(a), "os": sys.platform}

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
        obs["skipped"] = "measured from the head step only (the base tree runs from there)"
        save()
        return 0
    head = Path(a.checkout).resolve()
    base = head.parent / "base"
    work = out.parent / "c3dpbwork"
    work.mkdir(parents = True, exist_ok = True)
    heavy = Path(os.environ.get("AMD_CI_WORK") or work.parent.parent) / "c3dpbheavy"
    heavy.mkdir(parents = True, exist_ok = True)
    trees = {"A": base, "B": head, "M": base}
    obs["shas"] = {k: git("-C", p, "rev-parse", "HEAD")[1] for k, p in (("base", base), ("head", head))}
    obs["head_log"] = git("-C", head, "log", "--oneline", "-3")[1]
    obs["head_has_prebuild"] = "def prebuild(" in (head / "studio" / "backend" / "core" / "inference" /
                                                  "diffusion_rocm_triton_conv3d.py").read_text(encoding = "utf-8")
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

    def fresh(name):
        d = heavy / name
        shutil.rmtree(d, ignore_errors = True)
        d.mkdir(parents = True)
        return d
    warm_cache = {"A": heavy / "triton_A", "B": heavy / "triton_B", "M": heavy / "triton_M"}
    if "smoke" in items:
        obs["smoke"] = {}
        for arm in ("A", "B"):
            fresh(warm_cache[arm].name)
            obs["smoke"][arm] = worker(ctx, trees[arm], "smoke", arm,
                                       dict(ARM_ENV[arm], TRITON_CACHE_DIR = str(warm_cache[arm])), 4800)
            save()
        sm = obs["smoke"]

        def routed(arm, i):
            ds = sm[arm].get("decodes") or []
            return (ds[i].get("counts") or {}).get("routed", 0) if len(ds) > i else None
        obs["smoke_routed"] = {arm: [routed(arm, 0), routed(arm, 1)] for arm in sm}
        ok_run = all(sm[x].get("rc") == 0 and not sm[x].get("error") and len(sm[x].get("decodes") or []) == 2
                     for x in sm)
        obs["smoke_ok"] = bool(ok_run and (obs["smoke_routed"]["A"][1] or 0) > 0
                               and (obs["smoke_routed"]["B"][1] or 0) > 0 and sm["B"].get("has_prebuild"))
        save()
        if not obs["smoke_ok"]:
            obs["fatal"] = "smoke failed: big items skipped"
            save()
            return 0
    if "cold" in items:
        obs["cold"] = {}
        for arm in ("A", "B"):
            cache = fresh(f"triton_cold_{arm}")
            obs["cold"][arm] = worker(ctx, trees[arm], "coldflow", arm,
                                      dict(ARM_ENV[arm], TRITON_CACHE_DIR = str(cache)), 6000)
            shutil.rmtree(cache, ignore_errors = True)
            save()
    if "warm" in items:
        warm_cache["M"].mkdir(parents = True, exist_ok = True)
        obs["warm"] = {}
        for tag in WARM_PLAN:
            arm = tag[0]
            extra = ["--res", "480"] if arm == "M" else []
            obs["warm"][tag] = worker(ctx, trees[arm], "warm", tag,
                                      dict(ARM_ENV[arm], TRITON_CACHE_DIR = str(warm_cache[arm])), 4800, extra)
            obs["warm"][tag]["arm"] = arm
            save()
    obs["items_s"] = round(time.time() - t_all, 1)
    for sub in ("wan_vae", "hf", "triton_A", "triton_B", "triton_M"):
        shutil.rmtree(heavy / sub, ignore_errors = True)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
