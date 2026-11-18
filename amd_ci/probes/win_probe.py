#!/usr/bin/env python3
"""Probe (unslothai/unsloth#18313) for Windows ROCm: the checkout's varlen self-check verdict, and on base
the packed attention kernel bench (pure torch). Observes only."""

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
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    a = ap.parse_args()
    logs = a.out.parent
    work = Path(os.environ.get("RUNNER_TEMP") or os.environ.get("TEMP") or ".") / "i17738"
    work.mkdir(parents = True, exist_ok = True)
    env = dict(os.environ, PYTHONFAULTHANDLER = "1")
    obs = {"state": a.state}
    check = work / f"selfcheck_{a.state}.json"
    with open(logs / f"selfcheck_{a.state}.log", "w", encoding = "utf-8") as fh:
        obs["selfcheck_rc"] = subprocess.run([a.python, str(HERE / "selfcheck_stub_child.py"), a.checkout, str(check)],
                                             env = env, stdout = fh, stderr = subprocess.STDOUT, timeout = 900).returncode
    if check.is_file():
        obs["selfcheck"] = json.loads(check.read_text(encoding = "utf-8"))
    if a.state == "head":
        bench = work / "bench.json"
        with open(logs / "bench_head.log", "w", encoding = "utf-8") as fh:
            obs["bench_rc"] = subprocess.run([a.python, str(HERE / "attn_bench_child.py"), str(bench)], env = env,
                                             stdout = fh, stderr = subprocess.STDOUT, timeout = 3000).returncode
        if bench.is_file():
            obs["bench"] = json.loads(bench.read_text(encoding = "utf-8"))
    a.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
