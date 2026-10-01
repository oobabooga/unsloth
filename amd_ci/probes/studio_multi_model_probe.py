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


WRAPPER = """#!/usr/bin/env bash
# The HIP device multiplier must fool only Studio's placement probe. Inherited by llama-server it
# shows the child two "GPUs" and llama.cpp splits a model over the phantom pair, which the shim
# cannot serve. Record what Studio asked for, fold phantom ordinals onto the real GPU (as the shim
# does for torch), and run the real binary without the shim.
{
  printf '%s HIP=%s ROCR=%s CUDA=%s ARGS=%s\\n' "$(date +%s)" "${HIP_VISIBLE_DEVICES-unset}" \\
    "${ROCR_VISIBLE_DEVICES-unset}" "${CUDA_VISIBLE_DEVICES-unset}" "$*"
} >> "__LOG__"
fold() { local out="" t; IFS=',' read -ra toks <<< "$1"; for t in "${toks[@]}"; do
  [[ "$t" =~ ^[0-9]+$ ]] && (( t >= __REAL__ )) && t=0
  [[ ",$out," == *",$t,"* ]] || out="${out:+$out,}$t"; done; printf '%s' "$out"; }
for v in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; do
  [ -n "${!v+x}" ] && export "$v=$(fold "${!v}")"
done
unset LD_PRELOAD SHIM_EXTRA_DEVICES
exec "__REAL_BIN__" "$@"
"""


def wrap_llama_server(home: Path, log: Path) -> str:
    """Put WRAPPER in front of every llama-server under the Studio home; returns what it did."""
    real = os.environ.get("AMD_CI_REAL_GPUS", "1")
    done = []
    for binary in home.glob("**/llama-server"):
        if not binary.is_file() or binary.name.endswith(".real"):
            continue
        moved = binary.with_name("llama-server.real")
        binary.rename(moved)
        binary.write_text(WRAPPER.replace("__LOG__", str(log)).replace("__REAL__", real)
                          .replace("__REAL_BIN__", str(moved)), encoding = "utf-8")
        binary.chmod(0o755)
        done.append(str(binary))
    return ",".join(done) or "no llama-server found"


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
    obs["llama_backend_forced"] = os.environ.get("UNSLOTH_LLAMA_CPP_BACKEND")
    obs["spoofed_devices"] = os.environ.get("AMD_CI_SPOOFED_DEVICES")
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
    launches = tmp / f"llama_launches_{args.state}.log"
    if cli.exists() and os.environ.get("AMD_CI_SPOOFED_DEVICES") and os.name != "nt":
        obs["llama_wrapper"] = wrap_llama_server(home, launches)
    if cli.exists():
        res = tmp / f"probe_{args.state}.json"
        p = subprocess.run([sys.executable, str(HERE / "multi_model_probe.py"), "--bin", str(cli),
                            "--home", str(home), "--out", str(res)],
                           capture_output = True, text = True, encoding = "utf-8", errors = "replace", env = env)
        obs["probe_rc"] = p.returncode
        obs["probe_tail"] = (p.stdout or "")[-4000:]
        if res.exists():
            obs["probe"] = json.loads(res.read_text(encoding = "utf-8"))
    if launches.exists():
        obs["llama_launches"] = launches.read_text(encoding = "utf-8", errors = "replace").splitlines()[-20:]
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
