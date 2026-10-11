#!/usr/bin/env python3
"""Probe: the Studio SDXL LoRA trainer at one state, in three precisions.

Runs sdxl_train_parity.py three times against this state's checkout, each in a fresh
process (seed, data and model identical): fp32 reference (`--mixed no`), fp16, bf16.
Writes one observations JSON via --out with every arm's per-step loss / grad_norm and
device facts. Observes only; criteria/sdxl_fp16_scaler.py judges.

  python sdxl_fp16_probe.py --state base --checkout <tree> --out obs_base.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ID = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
ARMS = (("fp32", "no"), ("fp16", "fp16"), ("bf16", "bf16"))


def ensure_model(dest: Path) -> Path:
    if (dest / "model_index.json").is_file():
        return dest
    from huggingface_hub import snapshot_download
    snapshot_download(REPO_ID, local_dir = str(dest))
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--model-dir", type = Path, default = None)
    ap.add_argument("--steps", type = int, default = 6)
    ap.add_argument("--res", type = int, default = 1024)
    ap.add_argument("--timeout", type = int, default = 1800)
    args = ap.parse_args()

    work = Path(os.environ.get("AMD_CI_WORK") or args.out.resolve().parent.parent)
    env = dict(os.environ)
    env.setdefault("HF_HOME", str(work / "hf_home"))  # never the runner's $HOME overlay
    env["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["HF_HOME"] = env["HF_HOME"]

    obs: dict = {"state": args.state, "checkout": args.checkout, "steps": args.steps, "res": args.res,
                 "arms": {}}
    try:
        model = ensure_model(args.model_dir or (work / "models" / "tiny-sdxl"))
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"model download failed: {type(e).__name__}: {e}"
        args.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
        return 1
    obs["model"] = str(model)

    for name, mixed in ARMS:
        arm_out = args.out.with_name(f"{args.out.stem}_{name}.json")
        log = args.out.with_name(f"{args.out.stem}_{name}.log")
        arm_out.unlink(missing_ok = True)
        cmd = [args.python, str(HERE / "sdxl_train_parity.py"), "--wt", args.checkout, "--base", str(model),
               "--mixed", mixed, "--steps", str(args.steps), "--res", str(args.res), "--out", str(arm_out)]
        with open(log, "wb") as fh:
            try:
                rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env,
                                    timeout = args.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        arm: dict = {"rc": rc, "log": str(log)}
        if arm_out.is_file():
            try:
                arm.update(json.loads(arm_out.read_text(encoding = "utf-8")))
            except Exception as e:  # noqa: BLE001
                arm["parse_error"] = f"{type(e).__name__}: {e}"
        else:
            arm["missing_output"] = True
            arm["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-1500:]
        obs["arms"][name] = arm

    args.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
