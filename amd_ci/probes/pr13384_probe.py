#!/usr/bin/env python3
"""Probe for unsloth PR 13384 at one state (an unsloth checkout). Observes only.

1. Three arms of pr13384_run.py (tiny packed FastModel LoRA SFT on a 2-layer Gemma-4 text model:
   sliding head_dim 256 window 32 + global head_dim 512; separate processes):
     default        wrapper on, efficient route allowed (the PR's default)
     efficient_off  UNSLOTH_HF_PACKED_EFFICIENT=0
     ref            UNSLOTH_HF_PACKED_VARLEN=0 (the masked sdpa reference)
2. pytest over the PR's test selection from the state's repo root, per-test outcomes from junit
   XML (files absent at a state are recorded as such).

The state's checkout is first on PYTHONPATH so its unsloth shadows the installed one. --zoo PATH
puts an unsloth_zoo checkout ahead of the installed one too (local dry runs). Unless
--allow-compile, UNSLOTH_COMPILE_DISABLE=1 for every arm and pytest, so the comparison does not
depend on whether the installed unsloth_zoo's compiler handles Gemma 4.
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
ARMS = {
    "default": {},
    "efficient_off": {"UNSLOTH_HF_PACKED_EFFICIENT": "0"},
    "ref": {"UNSLOTH_HF_PACKED_VARLEN": "0"},
}
DEFAULT_TESTS = ["tests/test_hf_packed_attention.py", "tests/test_processor_text_packing_gate.py",
                 "tests/test_flex_gemma4_sliding_scope.py", "tests/utils/test_packing.py",
                 "tests/utils/test_flash_impl.py"]
CLEARED = ("UNSLOTH_HF_PACKED_VARLEN", "UNSLOTH_HF_PACKED_EFFICIENT", "UNSLOTH_FLASH_ATTENTION_IMPL",
           "UNSLOTH_TORCH_VARLEN", "UNSLOTH_PROCESSOR_TEXT_PACKING", "UNSLOTH_COMPILE_DISABLE")


def _env(checkout: str, extra: dict, zoo: str | None, compile_disable: bool) -> dict:
    env = dict(os.environ)
    front = [checkout] + ([zoo] if zoo else [])
    env["PYTHONPATH"] = os.pathsep.join(front + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for k in CLEARED:
        env.pop(k, None)
    if compile_disable:
        env["UNSLOTH_COMPILE_DISABLE"] = "1"
    env.update(extra)
    return env


def run_pytest(checkout, tests, out_dir: Path, state, python, timeout, zoo, compile_disable) -> dict:
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
        p = subprocess.run(cmd, cwd = checkout, env = _env(checkout, extra, zoo, compile_disable),
                           capture_output = True, text = True, timeout = timeout)
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


def run_arm(checkout, arm, out_dir: Path, state, python, timeout, zoo, compile_disable) -> dict:
    extra = {"UNSLOTH_COMPILE_LOCATION": str(out_dir / f"compiled_{state}_{arm}"), **ARMS[arm]}
    out = out_dir / f"run_{state}_{arm}.json"
    log = out_dir / f"run_{state}_{arm}.log"
    # One model dir shared across states and arms: built once, identical bytes everywhere.
    model_dir = out_dir / "tiny_gemma4_text"
    cmd = [python, str(HERE / "pr13384_run.py"), "--out", str(out), "--model-dir", str(model_dir)]
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, cwd = str(out_dir), env = _env(checkout, extra, zoo, compile_disable),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--tests", nargs = "+", default = DEFAULT_TESTS)
    ap.add_argument("--timeout", type = int, default = 2400)
    ap.add_argument("--zoo", default = None, help = "unsloth_zoo checkout to put ahead of the installed one")
    ap.add_argument("--allow-compile", action = "store_true",
                    help = "leave UNSLOTH_COMPILE_DISABLE unset (default: set to 1 for every arm)")
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent
    cd = not args.allow_compile
    zoo = str(Path(args.zoo).resolve()) if args.zoo else None
    obs: dict = {"state": args.state, "checkout": args.checkout, "compile_disable": cd, "zoo_override": zoo}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout,
                                     capture_output = True, text = True).stdout.strip()
    obs["train"] = {}
    for arm in ARMS:
        obs["train"][arm] = run_arm(args.checkout, arm, out_dir, args.state, args.python, 1500, zoo, cd)
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    obs["pytest"] = run_pytest(args.checkout, args.tests, out_dir, args.state, args.python, args.timeout, zoo, cd)
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
