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


def _summary_ids(out: str, kind: str) -> list[str]:
    """Node ids from pytest's short summary, for `FAILED` or `ERROR` lines.

    `^FAILED\\s+(\\S+)` is wrong and was wrong in a way that looked right. A
    parametrized id may contain SPACES -- `test_decision_matches_python[gfx1030-
    gfx1100-gfx1030-11.0.0--not corroborated]` is a real one in this repo -- so
    `\\S+` stops at the first space, truncating the id and, when several ids share
    a prefix, collapsing them into one after `set()`. Measured on the Windows
    gfx1151 runner: 75 failures at the base and 116 at the head both reduced to
    the SAME 70 strings, which `head_is_worse` read as an unchanged failure set.

    The line is `FAILED <nodeid>` optionally followed by ` - <reason>`, so split on
    the first such separator and keep everything to its left.
    """
    ids: list[str] = []
    for line in (out or "").splitlines():
        if not line.startswith(kind + " "):
            continue
        body = line[len(kind) + 1:].strip()
        # The reason is separated by " - "; a nodeid cannot contain that sequence
        # because pytest renders it from the module path and the param id.
        nodeid = body.split(" - ", 1)[0].rstrip()
        if nodeid:
            ids.append(nodeid)
    return sorted(set(ids))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--subdir", default = "studio/backend")
    ap.add_argument("--python", default = sys.executable)
    # 900 s was not enough for a whole-directory selection: `pytest tests/` under
    # studio/backend hit TimeoutExpired at BOTH states about 9% in, so the
    # observation carried failed=[] and errors=[] twice over and the criteria had
    # nothing to compare. A timeout at both states is not "no regression", it is
    # no result, so the ceiling has to be generous enough that hitting it really
    # does mean a hang. Narrow the selection as well; do not rely on this alone.
    ap.add_argument("--timeout", type = int, default = 1800)
    ap.add_argument("--tests", nargs = "+", required = True)
    args = ap.parse_args()

    workdir = Path(args.checkout) / args.subdir
    obs: dict = {"state": args.state, "workdir": str(workdir), "tests": args.tests}

    if not workdir.is_dir():
        obs["error"] = f"no such directory: {workdir}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
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
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
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
    # -rE as well as -rf: a test that errors in a fixture never reaches its body,
    # so it is neither a pass nor a failure and the short summary omits it under
    # -rf alone. Observed on the Windows gfx1151 runner, where a session-scoped
    # conftest import of huggingface_hub errored 23 of 25 selected tests and the
    # observation recorded "0 failed" -- which a regression criteria reads as a
    # clean suite.
    # ONE -r, with both characters. pytest's -r is `store`, not `append`, so a
    # second -r REPLACES the first: `-rf -rE` asks for the error summary only, and
    # a run with failures but no errors then prints no short summary at all. That
    # left n_failed=56 beside an EMPTY failed-id list on the Windows gfx1151 runner
    # (run 34821272189), and comparing two empty sets reads as no regression.
    cmd = [args.python, "-m", "pytest", "-q", "-rfE"]
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
    # Ids come out of the WHOLE stdout, not the 20 kB tail: 116 failures of
    # parametrized tests overflow it, and the criteria then compares two id sets
    # that were silently truncated at different points.
    obs["failed"] = _summary_ids(out, "FAILED")
    obs["errors"] = _summary_ids(out, "ERROR")
    m = re.search(r"(\d+) failed", tail)
    obs["n_failed"] = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) passed", tail)
    obs["n_passed"] = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) skipped", tail)
    obs["n_skipped"] = int(m.group(1)) if m else 0
    # Errors are counted separately by pytest and were not counted here at all, so
    # a suite that collected 25 and errored 23 looked like a suite that collected 2.
    m = re.search(r"(\d+) error", tail)
    obs["n_errors"] = int(m.group(1)) if m else 0
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
