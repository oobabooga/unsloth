#!/usr/bin/env python3
"""Generic probe: run a pytest selection in a checkout and report what happened.

Observes only. It records which tests failed, by id, so the criteria module can
compare base and head by SET rather than by count. Counts alone are misleading
whenever a PR adds tests: a PR that adds 23 tests and breaks nothing shows
"237 passed" then "1 failed, 260 passed", which reads like a regression and is
not one. That exact mistake happened before this toolkit existed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--subdir", default = "studio/backend")
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 900)
    ap.add_argument("--tests", nargs = "+", required = True)
    args = ap.parse_args()

    workdir = Path(args.checkout) / args.subdir
    obs: dict = {"state": args.state, "workdir": str(workdir), "tests": args.tests}

    if not workdir.is_dir():
        obs["error"] = f"no such directory: {workdir}"
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    # Only run selections that exist at this state. A test file added by the PR
    # is absent at the base, and pytest treats that as a collection error, which
    # would look like a failure rather than an absence.
    present, absent = [], []
    for t in args.tests:
        (present if (workdir / t.split("::")[0]).exists() else absent).append(t)
    obs["absent_at_this_state"] = absent
    obs["selected"] = present
    if not present:
        obs["note"] = "no selected tests exist at this state"
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    # --timeout needs the pytest-timeout plugin. Passing it unconditionally makes
    # pytest exit 4 (usage error) with EMPTY STDOUT and the reason on stderr, which
    # this probe did not capture -- so a run that executed nothing reported rc=4,
    # 0 passed, 0 failed, and the criteria called that no regression. Observed on
    # the gfx1151 runner, whose Studio venv has no pytest-timeout.
    have_timeout = subprocess.run(
        [args.python, "-c", "import pytest_timeout"],
        capture_output = True).returncode == 0
    obs["pytest_timeout_plugin"] = have_timeout
    cmd = [args.python, "-m", "pytest", "-q", "-rf"]
    if have_timeout:
        cmd += ["--timeout", str(args.timeout)]
    cmd += present
    obs["cmd"] = " ".join(cmd)
    # Wall-clock ceiling regardless, so a hang is still bounded without the plugin.
    try:
        p = subprocess.run(cmd, cwd = workdir, capture_output = True, text = True,
                           timeout = args.timeout + 120)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc = -1
        out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        err = "TimeoutExpired"
    tail = (out or "")[-20000:]
    obs["rc"] = rc
    obs["tail"] = tail[-4000:]
    # stderr is where pytest writes usage errors. Not capturing it is what made
    # this failure invisible.
    obs["stderr_tail"] = (err or "")[-2000:]
    obs["failed"] = sorted(set(re.findall(r"^FAILED\s+(\S+)", tail, re.M)))
    obs["errors"] = sorted(set(re.findall(r"^ERROR\s+(\S+)", tail, re.M)))
    m = re.search(r"(\d+) failed", tail)
    obs["n_failed"] = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) passed", tail)
    obs["n_passed"] = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) skipped", tail)
    obs["n_skipped"] = int(m.group(1)) if m else 0
    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
