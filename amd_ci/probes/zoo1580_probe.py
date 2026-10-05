#!/usr/bin/env python3
"""Probe for unsloth-zoo PR 1580 at one state (a zoo checkout). Observes only.

1. Decode cells via zoo1580_decode.py, one process per (model, weights, batch): routed-path
   engagement counts with UNSLOTH_MOE_ROUTED_KERNEL default vs "0", logits / greedy tokens.
2. pytest of tests/test_moe_routed.py, tests/test_gpt_oss_routed_nf4.py,
   tests/test_gpt_oss_grouped_qlora.py; per-test outcomes and skip reasons from junit XML
   (a file the PR adds is absent at the base and recorded as such, not run).

The state's checkout is put first on PYTHONPATH so its unsloth_zoo shadows the installed one;
the imported file is recorded so the criteria can gate on it.
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
# (model, weights, batch, expert LoRA stash)
CELLS = [("hub", "bf16", 1, False), ("hub", "nf4", 1, False), ("kf", "bf16", 1, False), ("kf", "nf4", 1, False),
         ("kf", "nf4", 4, False), ("kf", "bf16", 1, True), ("kf", "bf16", 4, True), ("kf", "nf4", 1, True)]


def _env(checkout: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["UNSLOTH_IS_PRESENT"] = "1"
    for k in ("UNSLOTH_MOE_ROUTED_KERNEL", "UNSLOTH_MOE_ROUTED_MAX_SLOTS", "UNSLOTH_MOE_BACKEND"):
        env.pop(k, None)
    return env


def run_pytest(checkout: str, tests: list[str], out_dir: Path, state: str, python: str, timeout: int) -> dict:
    present = [t for t in tests if (Path(checkout) / t).exists()]
    obs: dict = {"selected": present, "absent_at_this_state": [t for t in tests if t not in present]}
    if not present:
        obs["note"] = "no selected tests exist at this state"
        return obs
    junit = out_dir / f"junit_{state}.xml"
    cmd = [python, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", f"--junitxml={junit}", *present]
    obs["cmd"] = " ".join(cmd)
    try:
        p = subprocess.run(cmd, cwd = checkout, env = _env(checkout), capture_output = True,
                           text = True, timeout = timeout)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc, out, err = -1, (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""), "TimeoutExpired"
    obs["rc"] = rc
    (out_dir / f"pytest_{state}.log").write_text(out + "\n--- stderr ---\n" + err, encoding = "utf-8")
    obs["tail"] = out[-6000:]
    obs["stderr_tail"] = err[-2000:]
    passed, failed, errors, skipped, fail_msgs = [], [], [], {}, {}
    if junit.is_file():
        try:
            for tc in ET.parse(junit).getroot().iter("testcase"):
                tid = f"{tc.get('classname')}::{tc.get('name')}"
                kinds = {c.tag: c for c in tc}
                if "failure" in kinds or "error" in kinds:
                    (failed if "failure" in kinds else errors).append(tid)
                    k = kinds.get("failure", kinds.get("error"))
                    fail_msgs[tid] = ((k.get("message") or "") + "\n" + (k.text or "")[-1500:])[:2000]
                elif "skipped" in kinds:
                    skipped[tid] = (kinds["skipped"].get("message") or "")[:200]
                else:
                    passed.append(tid)
        except Exception as e:  # noqa: BLE001
            obs["junit_error"] = repr(e)
    obs.update(passed = sorted(passed), failed = sorted(failed), errors = sorted(errors), skipped = skipped,
               fail_msgs = fail_msgs, n_passed = len(passed), n_failed = len(failed) + len(errors),
               n_skipped = len(skipped))
    return obs


def run_cell(checkout: str, model: str, weights: str, batch: int, lora: bool, out_dir: Path, state: str,
             python: str) -> dict:
    tag = f"{state}_{model}_{weights}_b{batch}" + ("_lora" if lora else "")
    out, logits, log = out_dir / f"decode_{tag}.json", out_dir / f"logits_{tag}.pt", out_dir / f"decode_{tag}.log"
    cmd = [python, str(HERE / "zoo1580_decode.py"), "--model", model, "--weights", weights,
           "--batch", str(batch), "--out", str(out), "--logits", str(logits)] + (["--lora"] if lora else [])
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, cwd = checkout, env = _env(checkout), stdout = fh,
                                stderr = subprocess.STDOUT, timeout = 900).returncode
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
    rec["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-2500:]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--tests", nargs = "+", default = ["tests/test_moe_routed.py", "tests/test_gpt_oss_routed_nf4.py",
                                                       "tests/test_gpt_oss_grouped_qlora.py"])
    ap.add_argument("--timeout", type = int, default = 3000)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent
    obs: dict = {"state": args.state, "checkout": args.checkout}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout,
                                     capture_output = True, text = True).stdout.strip()
    obs["decode"] = {}
    for model, weights, batch, lora in CELLS:
        key = f"{model}/{weights}/b{batch}" + ("/lora" if lora else "")
        obs["decode"][key] = run_cell(args.checkout, model, weights, batch, lora, out_dir, args.state, args.python)
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    obs["pytest"] = run_pytest(args.checkout, args.tests, out_dir, args.state, args.python, args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
