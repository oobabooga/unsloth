#!/usr/bin/env python3
"""Probe: run ONE set of test files, taken from the head, against every state.

The generic pytest probe runs each state's own tests, which cannot show a defect
for a change whose tests are added by the change itself: the base has no test to
fail, so the base leg is silent and the differential is vacuous.

This probe instead copies a fixed list of test files out of the HEAD checkout
into each state's checkout, at the same paths, and runs them there. The base then
runs the head's specification against the old source, which is what "the base
exhibits the defect" means for a behavioural change.

Observes only. It records failing test ids so the criteria module can compare by
SET, never by count, and it records whether each spec file was ADDED or
OVERWRITTEN at this state, so a criteria module can refuse a run in which the
backport silently did nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _restore(workdir: Path, rel: str) -> None:
    """Put ``rel`` back to what this state's commit says it is."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", rel],
                             cwd = workdir, capture_output = True).returncode == 0
    if tracked:
        subprocess.run(["git", "checkout", "--", rel], cwd = workdir, capture_output = True)
    else:
        (workdir / rel).unlink(missing_ok = True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--states", required = True, type = Path,
                    help = "states.json, used to locate the head checkout the spec comes from")
    ap.add_argument("--spec-state", default = "head")
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 900)
    ap.add_argument("--spec-tests", nargs = "+", required = True,
                    help = "test files, relative to the repository root, taken from the spec "
                           "state and copied into this one")
    ap.add_argument("--tests", nargs = "+", default = None,
                    help = "what to RUN, relative to the repository root. Defaults to the "
                           "backported files; widen it to a directory to get the regression "
                           "reading in the same leg")
    args = ap.parse_args()

    workdir = Path(args.checkout)
    obs: dict = {"state": args.state, "workdir": str(workdir),
                 "spec_tests": args.spec_tests, "tests": args.tests or args.spec_tests}

    states = json.loads(args.states.read_text())
    spec_root = Path(states["paths"][args.spec_state])
    obs["spec_state"] = args.spec_state
    obs["spec_root"] = str(spec_root)

    applied: list[dict] = []
    backported: list[str] = []
    for rel in args.spec_tests:
        src = spec_root / rel
        dst = workdir / rel
        # Restore this state's own copy first. A second run against a worktree
        # this probe has already written to would otherwise see the file it
        # copied last time, record UNCHANGED, and quietly measure the head's
        # source at both states. Absent from the index means the state does not
        # have the file at all, which is the ADDED case.
        if args.state != args.spec_state:
            _restore(workdir, rel)
        if not src.is_file():
            applied.append({"path": rel, "action": "MISSING_AT_SPEC_STATE"})
            continue
        existed = dst.is_file()
        same = existed and (src.samefile(dst) or dst.read_bytes() == src.read_bytes())
        dst.parent.mkdir(parents = True, exist_ok = True)
        if not (existed and src.samefile(dst)):
            shutil.copyfile(src, dst)
        applied.append({
            "path": rel,
            # UNCHANGED is the expected action at the spec state itself; anywhere
            # else it means the file this probe exists to backport was already
            # identical, so this state is not really being told anything new.
            "action": "UNCHANGED" if same else ("OVERWRITTEN" if existed else "ADDED"),
        })
        backported.append(rel)

    obs["spec_files"] = applied
    obs["backported"] = backported

    # What to run. A selection absent at this state is an absence, not a failure,
    # and pytest reports it as a collection error, so it is dropped rather than
    # counted.
    present = [t for t in (args.tests or backported) if (workdir / t.split("::")[0]).exists()]
    obs["absent_at_this_state"] = [t for t in (args.tests or backported) if t not in present]
    obs["selected"] = present
    if not backported or not present:
        obs["note"] = ("no spec test file could be copied into this state" if not backported
                       else "no selected tests exist at this state")
        args.out.write_text(json.dumps(obs, indent = 2))
        return 0

    # Caches are per-checkout, but a stale one from a previous leg of the same
    # worktree would let pytest reuse a collected module compiled against the
    # other state's source.
    # `--timeout` is an argument pytest REJECTS outright when pytest-timeout is
    # absent, so an interpreter without the plugin exits 4 having collected
    # nothing, which reads as a suite that ran and found nothing wrong.
    have_timeout = subprocess.run(
        [args.python, "-c", "import pytest_timeout"], capture_output = True).returncode == 0
    obs["pytest_timeout_plugin"] = have_timeout
    cmd = [args.python, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           *(["--timeout", str(args.timeout)] if have_timeout else []), "-rf", *present]
    p = subprocess.run(cmd, cwd = workdir, capture_output = True, text = True)
    tail = (p.stdout or "") + "\n" + (p.stderr or "")
    obs["rc"] = p.returncode
    obs["tail"] = tail[-6000:]
    obs["failed"] = sorted(set(re.findall(r"^FAILED\s+(\S+)", tail, re.M)))
    obs["errors"] = sorted(set(re.findall(r"^ERROR\s+(\S+)", tail, re.M)))
    for key, pat in (("n_failed", r"(\d+) failed"), ("n_passed", r"(\d+) passed"),
                     ("n_skipped", r"(\d+) skipped"), ("n_errors", r"(\d+) error")):
        m = re.search(pat, tail)
        obs[key] = int(m.group(1)) if m else 0
    obs["n_collected"] = obs["n_failed"] + obs["n_passed"] + obs["n_skipped"]
    args.out.write_text(json.dumps(obs, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
