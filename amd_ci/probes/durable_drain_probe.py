#!/usr/bin/env python3
"""Probe: how fast does the durable chat producer drain a model stream on this host's real SQLite?

Runs probes/durable_drain_cell.py in a checkout's studio/backend, once per cell, each with a fresh
UNSLOTH_STUDIO_HOME. Cells with --latency-ms 0 measure the host's own commits (the question for Windows);
the injected-delay cell is a controlled condition independent of the disk. Observes only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CELLS = [
    # (name, mode, latency_ms, prefill_events, repeats)
    ("real_paced", "paced", 0.0, 0, 3),
    ("real_burst", "burst", 0.0, 0, 3),
    ("real_paced_history", "paced", 0.0, 100_000, 2),
    ("injected30_paced", "paced", 30.0, 0, 1),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 900)
    args = ap.parse_args()

    backend = Path(args.checkout) / "studio" / "backend"
    cell_py = Path(__file__).resolve().parent / "durable_drain_cell.py"
    obs: dict = {"state": args.state, "backend": str(backend), "cells": {}}
    if not backend.is_dir():
        obs["error"] = f"no such directory: {backend}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    root = Path(os.environ.get("AMD_CI_WORK") or tempfile.gettempdir()) / f"drain_{args.state}"
    root.mkdir(parents = True, exist_ok = True)
    for name, mode, latency, prefill, repeats in CELLS:
        runs = []
        for i in range(repeats):
            home = root / f"{name}_{i}"
            shutil.rmtree(home, ignore_errors = True)
            home.mkdir(parents = True)
            out = root / f"{name}_{i}.json"
            log = root / f"{name}_{i}.log"
            env = {**os.environ, "UNSLOTH_STUDIO_HOME": str(home), "PYTHONIOENCODING": "utf-8"}
            cmd = [args.python, str(cell_py), "--out", str(out), "--mode", mode,
                   "--latency-ms", str(latency), "--prefill-events", str(prefill)]
            with open(log, "w", encoding = "utf-8") as fh:
                try:
                    rc = subprocess.run(cmd, cwd = backend, env = env, stdout = fh, stderr = subprocess.STDOUT,
                                        timeout = args.timeout).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            if out.exists():
                row = json.loads(out.read_text(encoding = "utf-8"))
            else:
                row = {"error": f"cell exited {rc}",
                       "log_tail": log.read_text(encoding = "utf-8", errors = "replace")[-1500:]}
            row["rc"] = rc
            runs.append(row)
            shutil.rmtree(home, ignore_errors = True)
        obs["cells"][name] = runs
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
