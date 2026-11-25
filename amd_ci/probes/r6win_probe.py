#!/usr/bin/env python3
"""Probe (R6-A n0#1): GPU_RESOURCE_CACHE_SIZE=64 vs unset on Windows gfx1151, Studio renders. Observes only.

Head state only (the base state records a skip; nothing differs between trees). FLUX.3 klein (public, token-free
fetch into the job's HF cache), Studio defaults, fresh process per arm, 5 back-to-back renders at mixed sizes per
process (job 1 = first render after load), arms ordered off on on off off on on off off on (5 + 5).
Per arm: per-job wall / denoise / VAE seconds (r6win/arm.py, a copy of wintiers/arm.py with mixed sizes) and the
PNG sha256 per job (the setting must not change pixels).

  python r6win_probe.py --state head --checkout states/head --out obs_head.json [--model unsloth/flux-3-image-klein]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from i8auto_probe import arm_env, fetch  # noqa: E402

TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")
ORDER = ["off", "on", "on", "off", "off", "on", "on", "off", "off", "on"]
SIZES = [[1024, 1024], [768, 1344], [1344, 768], [1024, 1024], [832, 1216]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--model", default = "unsloth/flux-3-image-klein")
    ap.add_argument("--prompts", type = int, default = 5)
    a = ap.parse_args()
    for k in TOKEN_VARS:
        os.environ.pop(k, None)
    out = Path(a.out).resolve()
    obs: dict = {"state": a.state, "platform": sys.platform, "model": a.model, "order": ORDER, "sizes": SIZES,
                 "parent_env_GPU_RESOURCE_CACHE_SIZE": os.environ.get("GPU_RESOURCE_CACHE_SIZE")}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    try:
        import torch
        obs["torch"], obs["hip"] = torch.__version__, getattr(torch.version, "hip", None)
        obs["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        obs["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if obs["gpu"] else None
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)[:500]
    if a.state != "head":
        obs["skipped"] = "measured on the head state only (arms differ by one env var)"
        save()
        return 0
    checkout = Path(a.checkout).resolve()
    backend = checkout / "studio" / "backend"
    work = out.parent / "r6win_arms"
    work.mkdir(parents = True, exist_ok = True)
    hf_home = out.parent / "hf_home"
    flog: list = []
    obs["fetch"] = fetch(a.model, ["*.png", "*.jpg", "variants/*"], hf_home, flog)
    save()
    arms = []
    for idx, setting in enumerate(ORDER):
        tag = f"a{idx}_{setting}"
        extra = {"GPU_RESOURCE_CACHE_SIZE": "64"} if setting == "on" else {}
        env = arm_env(hf_home, extra)
        if setting == "off":
            env.pop("GPU_RESOURCE_CACHE_SIZE", None)
        gen = {"sizes": SIZES}
        cmd = [sys.executable, "-u", str(HERE / "r6win" / "arm.py"), "--root", str(backend), "--kind", "image",
               "--model", a.model, "--tag", tag, "--out", str(work), "--load", "{}", "--gen", json.dumps(gen),
               "--prompts", str(a.prompts), "--warmup", "0", "--tf", "none", "--prof-step", "-1"]
        log = work / f"{tag}.log"
        t0 = time.time()
        try:
            with open(log, "w", encoding = "utf-8") as fh:
                rc = subprocess.run(cmd, cwd = backend, stdout = fh, stderr = subprocess.STDOUT, env = env,
                                    timeout = 3600).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        try:
            rec = json.loads((work / tag / "record.json").read_text(encoding = "utf-8"))
        except Exception as exc:  # noqa: BLE001
            rec = {"verdict": "error", "error": f"no record: {exc}"[:300]}
        shas = []
        for i in range(a.prompts):
            p = work / tag / f"r{i}.png"
            shas.append(hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else None)
        text = log.read_text(encoding = "utf-8", errors = "replace") if log.is_file() else ""
        arms.append({"idx": idx, "setting": setting, "rc": rc, "arm_seconds": round(time.time() - t0, 1),
                     "env_value": env.get("GPU_RESOURCE_CACHE_SIZE"), "load_s": rec.get("load_s"),
                     "status": rec.get("status"), "renders": rec.get("renders"), "png_sha": shas,
                     "verdict": rec.get("verdict"), "error": rec.get("error"), "log_tail": text[-2500:]})
        obs["arms"] = arms
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
