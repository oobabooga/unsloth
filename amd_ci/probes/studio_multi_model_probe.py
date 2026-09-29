#!/usr/bin/env python3
"""Probe: install this checkout's Studio into a private home, then drive two GGUFs through it.

Observes only (install outcome + multi_model_probe checks); criteria/studio_multi_model.py judges.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    tmp = Path(os.environ.get("RUNNER_TEMP") or args.out.parent)
    home = tmp / f"studio_home_{args.state}"
    obs: dict = {"state": args.state, "home": str(home)}
    hf = tmp / f"hf_{args.state}"
    # The runner's shared HF cache is not writable by every job: each state downloads into its own.
    env = {
        **os.environ,
        "UNSLOTH_STUDIO_HOME": str(home),
        "UNSLOTH_SKIP_AUTOSTART": "1",
        "HF_HOME": str(hf),
        "HF_HUB_CACHE": str(hf / "hub"),
        "HF_XET_CACHE": str(hf / "xet"),
        "HUGGINGFACE_HUB_CACHE": str(hf / "hub"),
    }
    t = time.time()
    log = tmp / f"install_{args.state}.log"
    with open(log, "w", encoding = "utf-8") as fh:
        rc = subprocess.run(["bash", "./install.sh", "--local"], cwd = args.checkout, env = env,
                            stdout = fh, stderr = subprocess.STDOUT).returncode
    obs["install_rc"] = rc
    obs["install_s"] = round(time.time() - t, 1)
    obs["install_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
    cli = home / "unsloth_studio" / "bin" / "unsloth"
    obs["cli_exists"] = cli.exists()
    if cli.exists():
        res = tmp / f"probe_{args.state}.json"
        p = subprocess.run([sys.executable, str(HERE / "multi_model_probe.py"), "--bin", str(cli),
                            "--home", str(home), "--out", str(res)],
                           capture_output = True, text = True, encoding = "utf-8", errors = "replace", env = env)
        obs["probe_rc"] = p.returncode
        obs["probe_tail"] = (p.stdout or "")[-4000:]
        if res.exists():
            obs["probe"] = json.loads(res.read_text(encoding = "utf-8"))
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
