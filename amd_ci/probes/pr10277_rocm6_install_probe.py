#!/usr/bin/env python3
"""Probe (PR 10277): run THIS checkout's real `install.sh --local` inside ROCm 6.x containers.

Observes only; criteria/pr10277_bnb_rocm6.py judges.

Per ROCm image: a standalone clone of the checkout's exact commit is mounted into
`rocm/dev-ubuntu-22.04:<tag>` with the GPU passed through, the real installer runs
there (so ROCm version detection reads a genuine ROCm 6.x userspace), and then
pr10277_measure.py records what the installed environment does: the torch build
the installer chose, which libbitsandbytes it loads, and whether a 4-bit
forward/backward and a tiny Unsloth QLoRA run survive. Each GPU step runs in a
subprocess, so a SIGSEGV is recorded as data rather than killing the probe.

The installer is told the arch is gfx1100 (UNSLOTH_ROCM_GFX_ARCH), the
RX 7900 XTX of issue 10273. On this gfx1151 runner gfx1151 carries its own ROCm 7
kernel floor, so without it both states would pick rocm7.0 and the changed branch
would never run. HSA_OVERRIDE_GFX_VERSION=11.0.0 lets the rocm6.x wheels, which
ship no gfx1151 kernels, run at all.
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


def sh(cmd: list[str], log, timeout: int | None = None) -> int:
    log.write(f"\n$ {' '.join(cmd)}\n".encode())
    log.flush()
    try:
        return subprocess.run(cmd, stdout = log, stderr = subprocess.STDOUT, timeout = timeout).returncode
    except subprocess.TimeoutExpired:
        return 124


def device_gids() -> list[str]:
    gids = set()
    for p in ["/dev/kfd", *[str(x) for x in Path("/dev/dri").glob("renderD*")]]:
        try:
            gids.add(str(os.stat(p).st_gid))
        except OSError:
            pass
    return sorted(gids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--images", default = "rocm/dev-ubuntu-22.04:6.1.2,rocm/dev-ubuntu-22.04:6.3.4")
    ap.add_argument("--gfx", default = "gfx1100")
    ap.add_argument("--hsa-override", default = "11.0.0")
    ap.add_argument("--install-timeout", type = int, default = 5400)
    args = ap.parse_args()

    work = args.out.parent / f"pr10277_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "gfx_declared": args.gfx, "hsa_override": args.hsa_override,
                 "images": {}}
    log = open(work / "probe_host.log", "ab")

    sha = subprocess.run(["git", "-C", str(args.checkout), "rev-parse", "HEAD"],
                         capture_output = True, text = True).stdout.strip()
    obs["commit"] = sha
    # The clone stays OUT of the artifact tree (out/ is uploaded whole).
    src = Path(os.environ.get("AMD_CI_WORK", str(args.out.parent))) / f"pr10277_src_{args.state}"
    if not (src / ".git").exists():
        rc = sh(["git", "clone", "-q", "--no-hardlinks", str(args.checkout), str(src)], log)
        rc = rc or sh(["git", "-C", str(src), "checkout", "-q", sha], log)
        if rc:
            obs["setup_error"] = f"could not clone {sha}"
            args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
            return 0

    gids = device_gids()
    obs["device_gids"] = gids
    for image in [i for i in args.images.split(",") if i]:
        tag = image.rsplit(":", 1)[-1]
        rec: dict = {"image": image}
        idir = work / f"rocm_{tag}"
        idir.mkdir(parents = True, exist_ok = True)
        present = subprocess.run(["docker", "image", "inspect", image],
                                 capture_output = True).returncode == 0
        rec["image_was_present"] = present
        (work / "pulled_images.txt").open("a", encoding = "utf-8").write(
            "" if present else image + "\n")
        t0 = time.time()
        rec["pull_rc"] = 0 if present else sh(["docker", "pull", "-q", image], log, timeout = 1800)
        cmd = ["docker", "run", "--rm", "--network", "host",
               "--device", "/dev/kfd", "--device", "/dev/dri",
               "--security-opt", "seccomp=unconfined", "--ipc", "host",
               *sum([["--group-add", g] for g in gids], []),
               "-e", f"UNSLOTH_ROCM_GFX_ARCH={args.gfx}",
               "-e", f"HSA_OVERRIDE_GFX_VERSION={args.hsa_override}",
               "-e", f"INSTALL_TIMEOUT={args.install_timeout}",
               "-e", f"HOST_UID={os.getuid()}", "-e", f"HOST_GID={os.getgid()}",
               "-v", f"{src}:/src:ro", "-v", f"{idir}:/out",
               "-v", f"{HERE}:/probe:ro",
               image, "bash", "/probe/pr10277_in_container.sh"]
        rec["docker_rc"] = sh(cmd, log, timeout = args.install_timeout + 2400)
        rec["seconds"] = round(time.time() - t0)
        res = idir / "result.json"
        if res.is_file():
            try:
                rec.update(json.loads(res.read_text(encoding = "utf-8")))
            except Exception as e:  # noqa: BLE001
                rec["parse_error"] = f"{type(e).__name__}: {e}"
        else:
            rec["missing_result"] = True
        obs["images"][tag] = rec

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
