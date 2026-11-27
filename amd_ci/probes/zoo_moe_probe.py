#!/usr/bin/env python3
"""Probe for an unsloth-zoo MoE PR at one state (a zoo checkout). Observes only.

Each case of --cases (tiny Qwen3-MoE / gpt-oss, 16-bit LoRA or 4-bit QLoRA, Unsloth gradient checkpointing)
trains in its own process via zoo_moe_train.py with the state's checkout first on PYTHONPATH; then pytest of
--tests that exist at this state (per-test outcomes from junit XML)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zoo1570_probe import run_pytest  # noqa: E402


def _env(checkout: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["UNSLOTH_IS_PRESENT"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def run_case(checkout: str, case: str, out_dir: Path, state: str, python: str, timeout: int) -> dict:
    tag = case.replace(":", "_")
    out, log = out_dir / f"train_{state}_{tag}.json", out_dir / f"train_{state}_{tag}.log"
    cmd = [python, str(HERE / "zoo_moe_train.py"), "--case", case, "--out", str(out)]
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, cwd = checkout, env = _env(checkout), stdout = fh, stderr = subprocess.STDOUT,
                                timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rec: dict = {"rc": rc}
    if out.is_file():
        try:
            rec.update(json.loads(out.read_text(encoding = "utf-8")))
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = repr(e)
    else:
        rec["missing_output"] = True
    rec["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--cases", default = "qwen3moe:16bit,qwen3moe:4bit,gptoss:16bit,gptoss:4bit")
    ap.add_argument("--tests", nargs = "*", default = [])
    ap.add_argument("--timeout", type = int, default = 1800)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent
    obs: dict = {"state": args.state, "checkout": args.checkout}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout,
                                     capture_output = True, text = True).stdout.strip()
    obs["train"] = {}
    for case in args.cases.split(","):
        obs["train"][case] = run_case(args.checkout, case, out_dir, args.state, args.python, 1200)
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    if args.tests:
        obs["pytest"] = run_pytest(args.checkout, args.tests, out_dir, args.state, args.python, args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
