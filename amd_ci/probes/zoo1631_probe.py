#!/usr/bin/env python3
"""Probe for unsloth-zoo PR 1631 (flex attention block-mask reuse) at one state (a zoo checkout). Observes only.

1. (--train) tiny gpt-oss bf16 LoRA cells via zoo1631_train.py, one process per cell, torch.compile ON (flex attention
   needs it): 3 AdamW steps, unsloth gradient checkpointing, mask builds counted.
   Cells reuse/r1, reuse/r2 (A/A), off/r1 (UNSLOTH_FLEX_MASK_REUSE=0; a no-op env at base).
2. pytest of --new-tests + tests/test_gpt_oss_*.py + tests/*flex*;
   the PR's new test file is copied from the head checkout into the base (recorded as "borrowed").

The state's checkout is first on PYTHONPATH so its unsloth_zoo shadows the installed one; the imported file is
recorded so the criteria can gate on it.
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
NEW_TESTS: list = []
PATTERNS = ("tests/test_gpt_oss_*.py", "tests/*flex*.py")
# (key, extra env)
CELLS = [
    ("reuse/r1", {}),
    ("reuse/r2", {}),
    ("off/r1", {"UNSLOTH_FLEX_MASK_REUSE": "0"}),
]
FAIL_MARKERS = ("WON'T CONVERT", "won't convert", "Backend compiler failed", "BackendCompilerFailed",
                "torch._dynamo.exc", "Unsupported:")


def _env(checkout: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["UNSLOTH_IS_PRESENT"] = "1"
    for k in ("UNSLOTH_MOE_BACKEND", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32", "UNSLOTH_GPTOSS_GROUPED",
              "UNSLOTH_MOE_GROUPED_TRITON", "UNSLOTH_DISABLE_MOE_TRITON", "UNSLOTH_MOE_GROUPED_LORA",
              "UNSLOTH_MOE_GROUPED", "UNSLOTH_MOE_GROUPED_NF4_STACK", "UNSLOTH_MOE_GROUPED_CACHE", "UNSLOTH_MOE_GROUPED_RECOMPUTE",
              "UNSLOTH_MOE_STACKED_LORA", "UNSLOTH_FAST_GRAD_PARAMS", "UNSLOTH_FLEX_MASK_REUSE",
              "UNSLOTH_ENABLE_FLEX_ATTENTION", "TORCHDYNAMO_DISABLE"):
        env.pop(k, None)
    return env


def run_pytest(checkout: str, out_dir: Path, state: str, python: str, timeout: int) -> dict:
    tests = sorted({str(p.relative_to(checkout)) for pat in PATTERNS
                    for p in Path(checkout).glob(pat)} | set(NEW_TESTS))
    present = [t for t in tests if (Path(checkout) / t).exists()]
    obs: dict = {"selected": present, "absent_at_this_state": [t for t in tests if t not in present]}
    if not present:
        obs["note"] = "no selected tests exist at this state"
        return obs
    junit = out_dir / f"junit_{state}.xml"
    cmd = [python, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--continue-on-collection-errors",
           f"--junitxml={junit}", *present]
    obs["cmd"] = " ".join(cmd)
    try:
        p = subprocess.run(cmd, cwd = checkout, env = _env(checkout), capture_output = True,
                           text = True, timeout = timeout)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc = -1
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = "TimeoutExpired"
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
                    skipped[tid] = (kinds["skipped"].get("message") or "")[:300]
                else:
                    passed.append(tid)
        except Exception as e:  # noqa: BLE001
            obs["junit_error"] = repr(e)
    obs.update(passed = sorted(passed), failed = sorted(failed), errors = sorted(errors), skipped = skipped,
               fail_msgs = fail_msgs, n_passed = len(passed), n_failed = len(failed) + len(errors),
               n_skipped = len(skipped))
    return obs


def run_cell(checkout: str, key: str, extra: dict, out_dir: Path, state: str, python: str) -> dict:
    tag = f"{state}_{key.replace('/', '_')}"
    env = _env(checkout)
    env.update(extra)
    out, log = out_dir / f"train_{tag}.json", out_dir / f"train_{tag}.log"
    cmd = [python, "-u", str(HERE / "zoo1631_train.py"), "--out", str(out)]
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, cwd = checkout, env = env, stdout = fh,
                                stderr = subprocess.STDOUT, timeout = 1500).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rec: dict = {"rc": rc, "extra_env": extra}
    if out.is_file():
        try:
            rec.update(json.loads(out.read_text(encoding = "utf-8")))
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = repr(e)
    else:
        rec["missing_output"] = True
    text = log.read_text(encoding = "utf-8", errors = "replace")
    lines = text.splitlines()
    rec["log_failure_lines"] = [ln[:400] for ln in lines if any(m in ln for m in FAIL_MARKERS)][:20]
    rec["n_log_failure_lines"] = sum(1 for ln in lines if any(m in ln for m in FAIL_MARKERS))
    rec["log_tail"] = text[-2500:]
    return rec


INF_CELLS = [("inf_only", {}), ("grad_then_inf", {}), ("inf_then_grad", {}),
             ("inf_then_grad/off", {"UNSLOTH_FLEX_MASK_REUSE": "0"}), ("inf_only/off", {"UNSLOTH_FLEX_MASK_REUSE": "0"})]


def run_infmode(checkout: str, out_dir: Path, state: str, python: str) -> dict:
    res = {}
    for key, extra in INF_CELLS:
        tag = f"{state}_{key.replace('/', '_')}"
        out, log = out_dir / f"inf_{tag}.json", out_dir / f"inf_{tag}.log"
        env = _env(checkout)
        env.update(extra)
        cmd = [python, "-u", str(HERE / "zoo1631_infmode.py"), "--scenario", key.split("/")[0], "--out", str(out)]
        with open(log, "wb") as fh:
            try:
                rc = subprocess.run(cmd, cwd = checkout, env = env, stdout = fh, stderr = subprocess.STDOUT,
                                    timeout = 900).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        rec = {"rc": rc, "extra_env": extra}
        if out.is_file():
            rec.update(json.loads(out.read_text(encoding = "utf-8")))
        else:
            rec["missing_output"] = True
            rec["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-2000:]
        res[key] = rec
    # the PR's own test, isolated (fresh process, no earlier test in the session)
    junit = out_dir / f"junit_isolated_{state}.xml"
    t = "tests/test_flex_mask_reuse.py::test_inference_mode_mask_kept_apart"
    try:
        p = subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", t],
                           cwd = checkout, env = _env(checkout), capture_output = True, text = True, timeout = 900)
        res["isolated_pytest"] = {"rc": p.returncode, "tail": (p.stdout or "")[-1500:]}
    except subprocess.TimeoutExpired:
        res["isolated_pytest"] = {"rc": "timeout"}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--head-checkout", required = True)
    ap.add_argument("--timeout", type = int, default = 5400)
    ap.add_argument("--skip-pytest", action = "store_true")
    ap.add_argument("--leg", choices = ("t4", "t5"), required = True)
    ap.add_argument("--pr", required = True)
    ap.add_argument("--new-tests", nargs = "+", required = True)
    ap.add_argument("--train", action = "store_true")
    ap.add_argument("--infmode", action = "store_true")
    ap.add_argument("--only-new-tests", action = "store_true")
    args = ap.parse_args()
    NEW_TESTS[:] = args.new_tests
    args.out = args.out.resolve()
    out_dir = args.out.parent
    obs: dict = {"state": args.state, "checkout": args.checkout, "leg": args.leg, "pr": args.pr,
                 "new_tests": NEW_TESTS, "train_enabled": bool(args.train)}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout,
                                     capture_output = True, text = True).stdout.strip()
    borrowed = []
    for t in NEW_TESTS:
        dst, src = Path(args.checkout) / t, Path(args.head_checkout) / t
        if not dst.exists() and src.exists():
            shutil.copyfile(src, dst)
            borrowed.append(t)
    obs["borrowed_tests_from_head"] = borrowed
    obs["train"] = {}
    if args.train and args.state != "merge":
        for key, extra in CELLS:
            obs["train"][key] = run_cell(args.checkout, key, extra, out_dir, args.state, args.python)
            args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    if args.only_new_tests:
        global PATTERNS
        PATTERNS = ("tests/*flex*.py",)
    obs["infmode_enabled"] = bool(args.infmode)
    if args.infmode and args.state != "merge":
        obs["infmode"] = run_infmode(args.checkout, out_dir, args.state, args.python)
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    if not args.skip_pytest and args.state != "merge":
        obs["pytest"] = run_pytest(args.checkout, out_dir, args.state, args.python, args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
