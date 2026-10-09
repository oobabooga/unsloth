#!/usr/bin/env python3
"""Probe: numpy's OpenBLAS cost and speed per OPENBLAS_NUM_THREADS on Windows (Studio's #12374 default is 1).

Observes only. A plain venv with numpy + psutil, then scripts/numpy_threads_curve.py's cells: one fresh process per
thread count, committed memory (private bytes) right after `import numpy` and BLAS timings. State independent.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args, _ = ap.parse_known_args()
    work = Path(os.environ.get("AMD_CI_WORK") or args.out.parent) / f"np_{args.state}"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "cpu_count": os.cpu_count()}
    try:
        venv = work / "venv"
        base = getattr(sys, "_base_executable", None) or sys.executable
        subprocess.run([base, "-m", "venv", str(venv)], check = True, timeout = 600)
        py = venv / "Scripts" / "python.exe"
        subprocess.run([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", "numpy", "psutil"],
                       check = True, timeout = 1200)
        r = subprocess.run([str(py), str(HERE / "numpy_threads_curve.py"), str(py)], capture_output = True,
                           text = True, encoding = "utf-8", errors = "replace", timeout = 5400)
        res = next((json.loads(x[7:]) for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
        if res is None:
            obs["probe_error"] = (r.stdout + r.stderr)[-3000:]
        else:
            obs.update(res)
            obs["numpy"] = subprocess.run([str(py), "-c", "import numpy; print(numpy.__version__)"], capture_output = True,
                                          text = True).stdout.strip()
    except BaseException as e:  # noqa: BLE001
        obs["probe_error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
