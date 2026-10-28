#!/usr/bin/env python3
"""Probe for unsloth PR 13326 at one state (an unsloth checkout). Observes only.

1. Three arms of pr13326_run.py (tiny packed FastModel LoRA SFT, separate processes):
     sliding     sliding_window=32, wrapper on (default)
     sliding_ref sliding_window=32, UNSLOTH_HF_PACKED_VARLEN=0 (the masked sdpa reference)
     full        no sliding window, wrapper on (control: is the wrapper's fast route reachable at all)
   UNSLOTH_FLASH_ATTENTION_IMPL is left unset (the default the PR ships).
2. pytest over the PR's test selection, per-test outcomes from junit XML (files absent at a state
   are recorded as such).

The state's checkout is first on PYTHONPATH so its unsloth shadows the installed one.
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
WINDOW = 32
ARMS = {
    "sliding": (WINDOW, {}),
    "sliding_ref": (WINDOW, {"UNSLOTH_HF_PACKED_VARLEN": "0"}),
    "full": (0, {}),
}
DEFAULT_TESTS = ["tests/test_hf_packed_attention.py", "tests/utils/test_flash_impl.py", "tests/utils/test_packing.py",
                 "tests/test_padding_free_eval.py", "tests/test_padding_free_signature_gate.py"]


def _env(checkout: str, extra: dict) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in ("UNSLOTH_HF_PACKED_VARLEN", "UNSLOTH_FLASH_ATTENTION_IMPL", "UNSLOTH_TORCH_VARLEN"):
        env.pop(k, None)
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
    window, extra_env = ARMS[arm]
    extra = {"UNSLOTH_COMPILE_LOCATION": str(out_dir / f"compiled_{state}_{arm}"), **extra_env}
    out = out_dir / f"run_{state}_{arm}.json"
    log = out_dir / f"run_{state}_{arm}.log"
    # One model dir per window, shared across states: built once, identical bytes for base and head.
    model_dir = out_dir / f"tiny_mistral_w{window or 'none'}"
    cmd = [python, str(HERE / "pr13326_run.py"), "--out", str(out), "--model-dir", str(model_dir),
           "--window", str(window)]
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, cwd = str(out_dir), env = _env(checkout, extra), stdout = fh,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--tests", nargs = "+", default = DEFAULT_TESTS)
    ap.add_argument("--timeout", type = int, default = 2400)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent
    obs: dict = {"state": args.state, "checkout": args.checkout}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout,
                                     capture_output = True, text = True).stdout.strip()
    obs["train"] = {}
    for arm in ARMS:
        obs["train"][arm] = run_arm(args.checkout, arm, out_dir, args.state, args.python, 1500)
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    obs["pytest"] = run_pytest(args.checkout, args.tests, out_dir, args.state, args.python, args.timeout)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
