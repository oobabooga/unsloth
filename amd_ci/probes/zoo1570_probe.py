#!/usr/bin/env python3
"""Probe for unsloth-zoo PR 1570 at one state (a zoo checkout). Observes only.

1. pytest of the PR's grouped-QLoRA tests plus tests/test_gpt_oss_routed_nf4.py,
   per-test outcomes from junit XML (a file the PR adds is absent at the base and
   recorded as such, not run).
2. Two training arms in separate processes, UNSLOTH_GPTOSS_GROUPED unset (default)
   and "0" (forced per-expert loop), via zoo1570_train.py; then the max relative
   difference of the step-0 LoRA grads between the arms.

The state's checkout is put first on PYTHONPATH so its unsloth_zoo shadows the
installed one; the imported file is recorded so the criteria can gate on it.
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


def _env(checkout: str, extra: dict) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["UNSLOTH_IS_PRESENT"] = "1"
    env.pop("UNSLOTH_GPTOSS_GROUPED", None)
    env.update(extra)
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
        p = subprocess.run(cmd, cwd = checkout, env = _env(checkout, {}), capture_output = True,
                           text = True, timeout = timeout)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc, out, err = -1, (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""), "TimeoutExpired"
    obs["rc"] = rc
    obs["tail"] = out[-6000:]
    obs["stderr_tail"] = err[-2000:]
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


def run_arm(checkout: str, arm: str, out_dir: Path, state: str, python: str, timeout: int) -> dict:
    extra = {} if arm == "default" else {"UNSLOTH_GPTOSS_GROUPED": "0"}
    out = out_dir / f"train_{state}_{arm}.json"
    grads = out_dir / f"grads_{state}_{arm}.pt"
    log = out_dir / f"train_{state}_{arm}.log"
    cmd = [python, str(HERE / "zoo1570_train.py"), "--out", str(out), "--grads", str(grads)]
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, cwd = checkout, env = _env(checkout, extra), stdout = fh,
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
    rec["grads_file"] = str(grads) if grads.is_file() else None
    return rec


def grad_diff(a: str | None, b: str | None) -> dict:
    if not a or not b:
        return {"error": "missing grads file"}
    try:
        import torch
        ga, gb = torch.load(a), torch.load(b)
        names = sorted(set(ga) | set(gb))
        rel, missing = {}, []
        for n in names:
            if n not in ga or n not in gb:
                missing.append(n)
                continue
            rel[n] = float((ga[n] - gb[n]).norm() / (gb[n].norm() + 1e-12))
        worst = max(rel, key = rel.get) if rel else None
        return {"n": len(rel), "missing": missing, "max_rel": rel[worst] if worst else None, "worst": worst,
                "bit_identical": all(torch.equal(ga[n], gb[n]) for n in rel)}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--tests", nargs = "+",
                    default = ["tests/test_gpt_oss_grouped_qlora.py", "tests/test_gpt_oss_routed_nf4.py"])
    ap.add_argument("--timeout", type = int, default = 1800)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent
    obs: dict = {"state": args.state, "checkout": args.checkout}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout,
                                     capture_output = True, text = True).stdout.strip()
    obs["train"] = {arm: run_arm(args.checkout, arm, out_dir, args.state, args.python, 900)
                    for arm in ("default", "loop")}
    obs["grad_diff_default_vs_loop"] = grad_diff(obs["train"]["default"].get("grads_file"),
                                                 obs["train"]["loop"].get("grads_file"))
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    obs["pytest"] = run_pytest(args.checkout, args.tests, out_dir, args.state, args.python, args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
