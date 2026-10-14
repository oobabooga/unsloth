#!/usr/bin/env python3
"""Probe: which llama.cpp prebuilt does this checkout pick for an RDNA 1 host with ROCm installed?

Observes only. A fake `rocminfo` placed first on PATH reports one GPU agent (default gfx1010,
RX 5700 XT) beside the CPU agent, exactly as ROCm userspace does on such a host; everything else
is the runner's real detection (sysfs, the real RADV Vulkan ICD, no NVIDIA). The checkout's own
studio/install_llama_prebuilt.py --resolve-prebuilt answers against the real published release.

Two arms per checkout:
  default  masks and HSA_OVERRIDE_GFX_VERSION unset
  control  HSA_OVERRIDE_GFX_VERSION=10.3.0 (the user runs the card on ROCm on purpose)

With --run-chosen, a Vulkan asset chosen by the default arm is downloaded and its llama-server
--list-devices run WITHOUT the fake rocminfo, then a tiny GGUF completion with -ngl 99, so the
chosen bundle is shown to drive the real Radeon.

Writes JSON to --out, never stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

FAKE_ROCMINFO = """#!/bin/sh
cat <<'EOF'
ROCk module is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.14
==========
HSA Agents
==========
*******
Agent 1
*******
  Name:                    AMD Ryzen 7 3700X 8-Core Processor
  Uuid:                    CPU-XX
  Marketing Name:          AMD Ryzen 7 3700X 8-Core Processor
  Vendor Name:             CPU
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    __GFX__
  Uuid:                    GPU-XX
  Marketing Name:          AMD Radeon RX 5700 XT
  Vendor Name:             AMD
  Device Type:             GPU
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--__GFX__:xnack-
*** Done ***
EOF
"""

SCRUB = (
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION", "UNSLOTH_LLAMA_CPP_BACKEND",
    "UNSLOTH_FORCE_VULKAN", "UNSLOTH_ROCM_GFX_ARCH", "UNSLOTH_LLAMA_RELEASE_TAG",
)
TINY_GGUF = ("https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/"
             "SmolLM2-135M-Instruct-Q4_K_M.gguf")


def _env(fake_dir: str | None, control: bool) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    if fake_dir:
        env["PATH"] = fake_dir + os.pathsep + env.get("PATH", "")
    if control:
        env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    return env


def _last_json(text: str):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def resolve(checkout: Path, env: dict) -> dict:
    script = checkout / "studio" / "install_llama_prebuilt.py"
    r = subprocess.run(
        [sys.executable, str(script), "--resolve-prebuilt", "--output-format", "json"],
        env = env, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
        timeout = 600,
    )
    return {"rc": r.returncode, "payload": _last_json(r.stdout),
            "stderr_tail": r.stderr[-3000:], "stdout_tail": r.stdout[-1500:]}


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers = {"User-Agent": "amd-ci-probe"})
    with urllib.request.urlopen(req, timeout = 600) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def run_chosen(payload: dict, work: Path) -> dict:
    out: dict = {}
    url = (f"https://github.com/{payload['repo']}/releases/download/"
           f"{payload['release_tag']}/{payload['asset']}")
    out["url"] = url
    archive = work / payload["asset"]
    t = time.time()
    _download(url, archive)
    out["download_s"] = round(time.time() - t, 1)
    root = work / "bundle"
    root.mkdir()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(root)
    servers = sorted(p for p in root.rglob("llama-server") if p.is_file())
    out["llama_server"] = str(servers[0]) if servers else None
    if not servers:
        return out
    server = servers[0]
    bindir = server.parent
    for p in bindir.iterdir():
        if p.is_file():
            p.chmod(p.stat().st_mode | 0o111)
    env = _env(None, False)
    env["LD_LIBRARY_PATH"] = str(bindir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    r = subprocess.run([str(server), "--list-devices"], env = env, capture_output = True,
                       text = True, encoding = "utf-8", errors = "replace", timeout = 120)
    out["list_devices_rc"] = r.returncode
    out["list_devices"] = (r.stdout + "\n" + r.stderr)[-3000:]
    completion = next((bindir / n for n in ("llama-completion", "llama-cli")
                       if (bindir / n).is_file()), None)
    out["completion_binary"] = completion.name if completion else None
    if completion is None:
        return out
    model = work / "tiny.gguf"
    try:
        _download(TINY_GGUF, model)
    except Exception as exc:  # noqa: BLE001
        out["model_error"] = repr(exc)
        return out
    args = [str(completion), "-m", str(model), "-p", "The capital of France is", "-n", "16",
            "-ngl", "99", "--temp", "0"]
    if completion.name == "llama-cli":
        args += ["-st"]
    t = time.time()
    r = subprocess.run(args, env = env, capture_output = True, text = True, encoding = "utf-8",
                       errors = "replace", timeout = 300, stdin = subprocess.DEVNULL)
    out["completion_s"] = round(time.time() - t, 1)
    out["completion_rc"] = r.returncode
    text = r.stdout + "\n" + r.stderr
    out["load_tensors"] = [l.strip() for l in text.splitlines()
                           if re.search(r"load_tensors:.*buffer size", l)][:8]
    out["completion_text"] = r.stdout[-400:]
    out["completion_log_tail"] = text[-2000:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--fake-gfx", default = "gfx1010")
    ap.add_argument("--run-chosen", action = "store_true")
    args = ap.parse_args()

    obs: dict = {"state": args.state, "fake_gfx": args.fake_gfx}
    try:
        work = Path(tempfile.mkdtemp(prefix = f"llama_route_{args.state}_",
                                     dir = os.environ.get("RUNNER_TEMP")))
        fake_dir = work / "fakebin"
        fake_dir.mkdir()
        fake = fake_dir / "rocminfo"
        fake.write_text(FAKE_ROCMINFO.replace("__GFX__", args.fake_gfx), encoding = "utf-8")
        fake.chmod(0o755)
        obs["fake_rocminfo_ok"] = "gfx" in subprocess.run(
            [str(fake)], capture_output = True, text = True).stdout
        obs["real_rocminfo"] = shutil.which("rocminfo", path = os.environ.get("PATH"))
        obs["vulkan_icds"] = sorted(str(p) for d in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d")
                                    if Path(d).is_dir() for p in Path(d).iterdir())
        obs["default"] = resolve(args.checkout, _env(str(fake_dir), False))
        obs["control"] = resolve(args.checkout, _env(str(fake_dir), True))
        payload = (obs["default"] or {}).get("payload") or {}
        if args.run_chosen and payload.get("install_kind", "").endswith("vulkan"):
            try:
                obs["chosen_run"] = run_chosen(payload, work)
            except Exception:  # noqa: BLE001
                obs["chosen_run"] = {"error": traceback.format_exc()[-2000:]}
    except Exception:  # noqa: BLE001
        obs["error"] = traceback.format_exc()[-3000:]
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
