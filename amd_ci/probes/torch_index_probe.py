#!/usr/bin/env python3
"""Probe: which PyTorch index does this checkout's install.sh pick on the REAL host, and what does it print?

Observes only. Lifts get_torch_index_url and its helpers with the checkout's own
tests/sh/test_get_torch_index_url.sh prefix (so the extraction list is the one that
state ships), then calls it with the host's real PATH: real amd-smi, rocminfo, ROCm tree.
Also resolves the repo.radeon.com URL for the host ROCm and records whether it answered.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    obs: dict = {"state": args.state}
    test = args.checkout / "tests" / "sh" / "test_get_torch_index_url.sh"
    install = args.checkout / "install.sh"
    try:
        src = test.read_text(encoding = "utf-8")
        cut = src.index('echo "=== test_get_torch_index_url ===')
        prefix = src[:cut].replace('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
                                   f'SCRIPT_DIR="{test.parent}"')
        driver = prefix + r'''
set +e
PATH="$REAL_PATH" bash -c ". '$_FUNC_FILE'; get_torch_index_url" 2>"$PROBE_ERR" >"$PROBE_OUT"
echo $? >"$PROBE_RC"
'''
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "drv.sh").write_text(driver, encoding = "utf-8")
            env = dict(os.environ, REAL_PATH = os.environ.get("PATH", ""),
                       PROBE_OUT = str(d / "out"), PROBE_ERR = str(d / "err"), PROBE_RC = str(d / "rc"))
            r = subprocess.run(["bash", str(d / "drv.sh")], env = env, capture_output = True,
                               text = True, encoding = "utf-8", timeout = 600)
            obs["driver_rc"] = r.returncode
            obs["driver_tail"] = (r.stdout + r.stderr)[-800:]
            obs["index_url"] = (d / "out").read_text(encoding = "utf-8").strip().splitlines()[-1:] or [""]
            obs["index_url"] = obs["index_url"][0]
            obs["stderr"] = (d / "err").read_text(encoding = "utf-8")[-4000:]
            obs["fn_rc"] = int((d / "rc").read_text(encoding = "utf-8").strip() or -1)
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"

    try:
        v = subprocess.run(["amd-smi", "version"], capture_output = True, text = True,
                           encoding = "utf-8", timeout = 60).stdout
        m = re.search(r"ROCm version: ([0-9.]+)", v)
        obs["host_rocm"] = m.group(1) if m else None
    except Exception as e:  # noqa: BLE001
        obs["host_rocm"] = None
        obs["amd_smi_error"] = f"{type(e).__name__}: {e}"

    # Radeon repo: the host's rocm-rel URL and the HTTP answer, observed directly.
    if obs.get("host_rocm"):
        url = f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{obs['host_rocm']}/"
        r = subprocess.run(["curl", "-fsSL", "--max-time", "20", "-o", os.devnull, url],
                           capture_output = True, text = True, encoding = "utf-8")
        obs["radeon_url"], obs["radeon_curl_rc"] = url, r.returncode
    obs["install_sh_bytes"] = install.stat().st_size if install.exists() else None
    args.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
