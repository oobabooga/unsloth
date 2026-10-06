#!/usr/bin/env python3
"""Probe for unsloth-zoo PR 1604 at one state (a zoo checkout). Observes only.

1. Training cells via zoo1604_train.py, one process per cell: tiny Qwen3-MoE bf16 LoRA through FastLanguageModel,
   4 AdamW steps. Cells: eager r1 / r2 (A/A), eager with UNSLOTH_MOE_GROUPED_TRITON=1 (forces the PR's gate on
   wherever it can be on), compiled r1 (Unsloth defaults, dynamo on).
2. pytest of tests/test_moe_*.py + tests/test_gpt_oss_grouped_qlora.py; the PR's new test files are copied from
   the head checkout into the base (recorded as "borrowed"). Per-test outcomes from junit XML.

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
NEW_TESTS = ["tests/test_moe_grouped_triton_generic.py", "tests/test_moe_grouped_generic_config.py"]
PATTERNS = ("tests/test_moe_*.py", "tests/test_gpt_oss_grouped_qlora.py")
# (key, extra env, instrument). "eager" = TORCHDYNAMO_DISABLE=1.
CELLS = [
    ("eager/r1", {"TORCHDYNAMO_DISABLE": "1"}, True),
    ("eager/r2", {"TORCHDYNAMO_DISABLE": "1"}, True),
    ("eager-forced/r1", {"TORCHDYNAMO_DISABLE": "1", "UNSLOTH_MOE_GROUPED_TRITON": "1"}, True),
    ("compiled/r1", {}, False),
]
FAIL_MARKERS = ("WON'T CONVERT", "won't convert", "Backend compiler failed", "BackendCompilerFailed",
                "torch._dynamo.exc", "Unsupported:")


def _env(checkout: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["UNSLOTH_IS_PRESENT"] = "1"
    for k in ("UNSLOTH_MOE_BACKEND", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32", "UNSLOTH_GPTOSS_GROUPED",
              "UNSLOTH_MOE_GROUPED_TRITON", "UNSLOTH_DISABLE_MOE_TRITON", "UNSLOTH_MOE_GROUPED_TRITON_MAX_ROWS"):
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


def run_cell(checkout: str, key: str, extra: dict, instrument: bool, out_dir: Path, state: str, python: str) -> dict:
    tag = f"{state}_{key.replace('/', '_')}"
    env = _env(checkout)
    env.update(extra)
    out, log = out_dir / f"train_{tag}.json", out_dir / f"train_{tag}.log"
    cmd = [python, "-u", str(HERE / "zoo1604_train.py"), "--out", str(out)] + (["--instrument"] if instrument else [])
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--head-checkout", required = True)
    ap.add_argument("--timeout", type = int, default = 5400)
    ap.add_argument("--skip-pytest", action = "store_true")
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent
    obs: dict = {"state": args.state, "checkout": args.checkout}
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
    for key, extra, instrument in CELLS:
        obs["train"][key] = run_cell(args.checkout, key, extra, instrument, out_dir, args.state, args.python)
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    merge = args.state == "merge"   # merge: cells only, no pytest (time); base vs head is the comparison
    if not args.skip_pytest and not merge:
        obs["pytest"] = run_pytest(args.checkout, out_dir, args.state, args.python, args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
