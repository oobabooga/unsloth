#!/usr/bin/env python3
"""Probe for unsloth PR 16041 (Gemma 3 generate off flex_attention) at one state. Observes only.

1. pr16041_run.py twice (reps a, b), separate processes: load unsloth/gemma-3-270m-it 16-bit with
   FastLanguageModel, record attn impl, time greedy 128-token generate, record attn impl during / after
   generate, then LoRA r8 + 3 SFT steps. Two reps per state give the A/A spread.
2. pytest over the PR's CPU test. The file is added by the PR, so at a state that lacks it the head's
   copy is placed into this checkout's tests/ (the base is then expected to fail collection).

The state's checkout is first on PYTHONPATH so its unsloth shadows the installed one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPS = ("a", "b")
TEST = "tests/test_gemma3_generate_off_flex.py"


def _env(checkout: str, extra: dict) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("UNSLOTH_FLEX_ATTENTION_GENERATE", None)
    env.update(extra)
    return env


def run_rep(checkout: str, rep: str, out_dir: Path, state: str, python: str, timeout: int) -> dict:
    out = out_dir / f"run_{state}_{rep}.json"
    log = out_dir / f"run_{state}_{rep}.log"
    work = out_dir / f"work_{state}_{rep}"
    work.mkdir(parents = True, exist_ok = True)
    extra = {"UNSLOTH_COMPILE_LOCATION": str(out_dir / f"compiled_{state}_{rep}")}
    cmd = [python, str(HERE / "pr16041_run.py"), "--out", str(out), "--work", str(work)]
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, cwd = str(work), env = _env(checkout, extra), stdout = fh,
                                stderr = subprocess.STDOUT, timeout = timeout).returncode
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


def run_pytest(checkout: str, out_dir: Path, state: str, python: str, timeout: int) -> dict:
    obs: dict = {"test": TEST}
    target = Path(checkout) / TEST
    if not target.exists():
        donor = Path(checkout).parent / "head" / TEST
        if donor.is_file():
            shutil.copy2(donor, target)
            obs["copied_from_head"] = str(donor)
        else:
            obs["note"] = f"{TEST} absent here and no head copy at {donor}"
            return obs
    junit = out_dir / f"junit_{state}.xml"
    cmd = [python, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "-o", "addopts=",
           f"--junitxml={junit}", TEST]
    obs["cmd"] = " ".join(cmd)
    extra = {"UNSLOTH_COMPILE_LOCATION": str(out_dir / f"compiled_{state}_pytest")}
    try:
        p = subprocess.run(cmd, cwd = checkout, env = _env(checkout, extra), capture_output = True,
                           text = True, timeout = timeout)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc = -1
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = "TimeoutExpired"
    (out_dir / f"pytest_{state}.log").write_text(out + "\n--- stderr ---\n" + err, encoding = "utf-8")
    obs["rc"] = rc
    obs["tail"] = out[-4000:]
    obs["stderr_tail"] = err[-1500:]
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
    ap.add_argument("--timeout", type = int, default = 1500)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent
    obs: dict = {"state": args.state, "checkout": args.checkout}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout,
                                     capture_output = True, text = True).stdout.strip()
    obs["runs"] = {}
    for rep in REPS:
        obs["runs"][rep] = run_rep(args.checkout, rep, out_dir, args.state, args.python, args.timeout)
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    obs["pytest"] = run_pytest(args.checkout, out_dir, args.state, args.python, 900)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
