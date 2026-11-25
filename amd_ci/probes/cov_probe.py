#!/usr/bin/env python3
"""Probe: gfx1151 coverage renders through Studio's own load + generate (observes only; criteria/cov_renders.py).

Head state only (main; nothing differs between states). Arms from --arms (JSON list), each a fresh process:
  {"tag", "kind": "image"|"video", "model", "load": {...}, "gen": {...}, "renders": N, "png": k, "timeout": s,
   "purge_after": true}
video arms run probes/mq/wan_arm.py (fp32 CPU noise from the seed, the same draw as the NVIDIA reference harness),
image arms probes/mq/img_arm.py. ``purge_after`` deletes the Hugging Face hub cache after the arm (disk).
Public repos only; every token variable is stripped from the children.

  python cov_probe.py --state head --checkout <tree> --out obs.json --arms amd_ci/arms/cov_win.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN", "GH_TOKEN",
              "GITHUB_TOKEN")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--arms", required = True)
    a = ap.parse_args()
    out = Path(a.out)
    obs: dict = {"state": a.state, "arms": {}, "order": []}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    if a.state != "head":
        obs["skipped"] = "renders run on the head tree only (no code differs between states)"
        save()
        return 0
    backend = str(Path(a.checkout).resolve() / "studio" / "backend")
    od = out.parent / "cov"
    work = out.resolve().parents[2] / "cov_work"  # the job root, outside the uploaded out/
    od.mkdir(parents = True, exist_ok = True)
    work.mkdir(parents = True, exist_ok = True)
    hf_home = Path(os.environ.get("HF_HOME") or (work / "hf_home"))
    arms_path = Path(a.arms)
    if not arms_path.is_absolute():
        arms_path = Path(__file__).resolve().parents[1] / "arms" / arms_path.name
    arms = json.loads(arms_path.read_text(encoding = "utf-8"))
    try:
        obs["disk_free_gb_start"] = round(shutil.disk_usage(work).free / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    for spec in arms:
        tag = spec["tag"]
        script = HERE / "mq" / ("img_arm.py" if spec.get("kind") == "image" else "wan_arm.py")
        cmd = [sys.executable, "-u", str(script), "--root", backend, "--model", spec["model"], "--tag", tag,
               "--out", str(od), "--load", json.dumps(spec.get("load") or {}), "--gen", json.dumps(spec.get("gen") or {}),
               "--renders", str(spec.get("renders", 2))]
        if spec.get("kind") != "image":
            cmd += ["--noise-fp32", "--save-png-every", str(spec.get("png", 1))]
        # HF_ENDPOINT: the runner's mirror stalls on uncached shards; the public Hub (no token) does not
        env = {k: v for k, v in os.environ.items() if k not in TOKEN_VARS + ("PYTHONPATH", "HF_ENDPOINT")}
        env.update({"PYTHONUNBUFFERED": "1", "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                    "HF_HOME": str(hf_home), "HF_HUB_DOWNLOAD_TIMEOUT": "300", "HF_HUB_ETAG_TIMEOUT": "60",
                    "TORCHINDUCTOR_CACHE_DIR": str(work / "ind")})
        env.update({k: str(v) for k, v in (spec.get("env") or {}).items()})
        t0 = time.time()
        log = od / f"{tag}.log"
        with open(log, "w", encoding = "utf-8", errors = "replace") as fh:
            try:
                rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env, cwd = str(work),
                                    timeout = spec.get("timeout", 10800)).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        rp = od / tag / "record.json"
        r = json.loads(rp.read_text(encoding = "utf-8")) if rp.is_file() else {"verdict": "missing"}
        r["rc"], r["arm_s"], r["spec"] = rc, round(time.time() - t0, 1), spec
        if r.get("verdict") != "ok":
            try:
                r["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace").splitlines()[-80:]
            except Exception:  # noqa: BLE001
                pass
        try:
            r["disk_free_gb_after"] = round(shutil.disk_usage(work).free / 1e9, 1)
        except Exception:  # noqa: BLE001
            pass
        obs["arms"][tag] = r
        obs["order"].append(tag)
        save()
        if spec.get("purge_after"):
            shutil.rmtree(hf_home / "hub", ignore_errors = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
