#!/usr/bin/env python3
"""Probe: FP8 block-scaled e4m3fn weight dequant route on ROCm at one state (an unsloth checkout).
Observes only.

1. fp8_hip_dequant_run.py in separate processes: UNSLOTH_FP8_HIP_BYTES_DEQUANT unset (default)
   and "0" (the head's kill switch; the base ignores it). Route counts, bit equality vs the torch
   loop and the fp8e4nv Triton kernel, per-call kernel time.
2. pytest over the state's own copies of --tests, per-test outcomes from junit XML.

The state's checkout is put first on PYTHONPATH so its unsloth shadows the installed one; the
imported file is recorded so the criteria can gate on it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = {"default": {}, "switch_off": {"UNSLOTH_FP8_HIP_BYTES_DEQUANT": "0"}}


def _env(checkout: str, extra: dict, out_dir: Path, tag: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in ("UNSLOTH_FP8_HIP_BYTES_DEQUANT", "UNSLOTH_TRITON_FAST_LAUNCH"):
        env.pop(k, None)
    env["UNSLOTH_COMPILE_LOCATION"] = str(out_dir / f"compiled_{tag}")
    env.update(extra)
    return env


def run_arm(checkout: str, arm: str, out_dir: Path, state: str, python: str, timeout: int) -> dict:
    out = out_dir / f"run_{state}_{arm}.json"
    log = out_dir / f"run_{state}_{arm}.log"
    cmd = [python, str(HERE / "fp8_hip_dequant_run.py"), "--out", str(out)]
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, cwd = str(out_dir), env = _env(checkout, ARMS[arm], out_dir, f"{state}_{arm}"),
                                stdout = fh, stderr = subprocess.STDOUT, timeout = timeout).returncode
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


def run_pytest(checkout: str, tests: list[str], out_dir: Path, state: str, python: str, timeout: int) -> dict:
    present = [t for t in tests if (Path(checkout) / t).exists()]  # relative to the checkout root
    obs: dict = {"selected": present, "absent_at_this_state": [t for t in tests if t not in present]}
    if not present:
        obs["note"] = "no selected tests exist at this state"
        return obs
    junit = out_dir / f"junit_{state}.xml"
    cmd = [python, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", f"--junitxml={junit}", *present]
    obs["cmd"] = " ".join(cmd)
    try:
        p = subprocess.run(cmd, cwd = checkout, env = _env(checkout, {}, out_dir, f"{state}_pytest"),
                           capture_output = True, text = True, timeout = timeout)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc, out, err = -1, (exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")), "TimeoutExpired"
    obs.update(rc = rc, tail = out[-6000:], stderr_tail = err[-2000:])
    passed, failed, errors, skipped = [], [], [], {}
    if junit.is_file():
        try:
            for tc in ET.parse(junit).getroot().iter("testcase"):
                tid = f"{tc.get('classname')}::{tc.get('name')}"
                kinds = {c.tag: c for c in tc}
                if "failure" in kinds:
                    failed.append(tid)
                elif "error" in kinds:
                    errors.append(tid)
                elif "skipped" in kinds:
                    skipped[tid] = (kinds["skipped"].get("message") or "")[:200]
                else:
                    passed.append(tid)
        except Exception as e:  # noqa: BLE001
            obs["junit_error"] = repr(e)
    obs.update(passed = sorted(passed), failed = sorted(failed), errors = sorted(errors), skipped = skipped,
               n_passed = len(passed), n_failed = len(failed) + len(errors), n_skipped = len(skipped))
    return obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--tests", nargs = "+", default = ["tests/test_fp8_pre_sm89_byte_dequant.py"])
    ap.add_argument("--timeout", type = int, default = 2400)
    # pytest_probe.py compatibility: --tests are already relative to the checkout root.
    ap.add_argument("--subdir", default = ".")
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent
    obs: dict = {"state": args.state, "checkout": args.checkout}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout,
                                     capture_output = True, text = True).stdout.strip()
    obs["arms"] = {}
    for arm in ARMS:
        obs["arms"][arm] = run_arm(args.checkout, arm, out_dir, args.state, args.python, 1500)
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    obs["pytest"] = run_pytest(args.checkout, args.tests, out_dir, args.state, args.python, args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
