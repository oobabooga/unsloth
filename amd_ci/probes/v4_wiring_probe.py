#!/usr/bin/env python3
"""Probe: Windows + NVIDIA wiring of Studio's SSM kernel install, per state, on a Windows box with
a REAL CUDA 13.0 torch wheel installed (no NVIDIA driver: the box's GPU is AMD).

  urls    - the state's kernel_install.resolve_wheel_url for causal_conv1d / mamba_ssm /
            flash_attn over torch 2.10-2.13 x cp311-cp313 win_amd64 cu13, each URL checked live.
  install - the state's ssm_runtime.ensure_ssm_runtime for Qwen3.5 and Falcon-H1 for real
            (downloads + installs the release wheels into this venv), subprocess / HTTP calls logged.
  import  - a fresh process imports causal_conv1d, causal_conv1d_cuda, mamba_ssm,
            selective_scan_cuda (DLL load against the CUDA torch; no kernel launch possible).
Kernels are uninstalled before and after each state. Observes only; JSON to --out.
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
sys.path.insert(0, str(HERE))
from v4_ondemand_probe import COMMON, STATE, _run, _uninstall  # noqa: E402

URLS = COMMON + r"""
studio = sys.argv[1]
backend = os.path.join(studio, "backend")
sys.path[:0] = [backend, studio]
os.chdir(backend)
from utils import kernel_install as k
from utils.wheel_utils import probe_torch_wheel_env
base = probe_torch_wheel_env(timeout = 300, include_windows = True)
out = {"probe_env": base, "cells": []}
for tv in ("2.10.0", "2.11.0", "2.12.0", "2.13.0"):
    for py in ("cp311", "cp312", "cp313"):
        env = dict(base or {})
        env.update(python_tag = py, abi_tag = py, torch_mm = tv.rsplit(".", 1)[0],
                   torch_version = tv + "+cu130", cuda_version = "13.0", cuda_major = "13",
                   hip_version = "", platform_tag = "win_amd64")
        row = {"torch": tv, "py": py}
        for n in ("causal_conv1d", "mamba_ssm", "flash_attn"):
            try:
                url = k.resolve_wheel_url(n, env)
            except Exception as e:
                url = f"ERR {type(e).__name__}: {e}"
            code = None
            if url and url.startswith("http"):
                try:
                    req = urllib.request.Request(url, method = "HEAD")
                    code = _orig_urlopen(req, timeout = 60).status
                except Exception as e:
                    code = str(getattr(e, "code", type(e).__name__))
            row[n] = {"url": url, "status": code}
        out["cells"].append(row)
_dump("URLS_JSON", out)
"""

INSTALL = COMMON + r"""
studio, model = sys.argv[1], sys.argv[2]
backend = os.path.join(studio, "backend")
sys.path[:0] = [backend, studio]
os.chdir(backend)
out = {"model": model}
t0 = time.time()
msgs = []
try:
    from utils import ssm_runtime as s
    r = s.ensure_ssm_runtime(model, status_cb = msgs.append)
    out["returned"] = r
    out["error"] = None
except BaseException as e:
    import traceback
    out["error"] = f"{type(e).__name__}: {e}"
    out["tb"] = traceback.format_exc()[-1500:]
out["seconds"] = round(time.time() - t0, 2)
out["status"] = msgs
out["calls"] = LOG
_dump("BLOCK_JSON", out)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    args = ap.parse_args()
    studio = str(Path(args.checkout) / "studio")
    py = args.python
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PIP_NO_CACHE_DIR"] = "1"
    work = Path(os.environ.get("V4_SCRATCH") or args.out.parent)
    work.mkdir(parents = True, exist_ok = True)
    env["UV_CACHE_DIR"] = str(work / f"uvcache_{args.state}")
    obs: dict = {"state": args.state}
    obs["torch"] = _run([py, "-c", "import torch, json; print('T ' + json.dumps([torch.__version__, "
                         "torch.version.cuda, torch.cuda.is_available()]))"], env, str(work), 300, "T")
    obs["urls"] = _run([py, "-c", URLS, studio], env, studio, 1800, "URLS_JSON")
    obs["reset_before"] = _uninstall(py, env)
    obs["install"] = {}
    for key, model in (("qwen35", "unsloth/Qwen3.5-0.8B"), ("falconh1", "tiiuae/Falcon-H1-Tiny-R-90M")):
        obs["install"][key] = _run([py, "-c", INSTALL, studio, model], env, studio, 3600, "BLOCK_JSON")
    obs["kernels"] = _run([py, "-c", STATE], env, str(work), 300, "STATE_JSON")
    obs["reset_after"] = _uninstall(py, env)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
