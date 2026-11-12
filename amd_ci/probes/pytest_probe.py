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
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# "3 failed, 120 passed in 12.34s", framed or not, optionally "(0:01:02)" after the seconds.
_SUMMARY_RE = re.compile(
    r"^=*\s*(?P<counts>(?:\d+ [a-z]+(?:, )?)+) in [\d.]+s(?: \([\d:]+\))?\s*=*\s*$",
    re.M)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--subdir", default = "studio/backend")
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 900, help = "per test (pytest-timeout)")
    # The whole run, separately: a ~1000-test suite on a Windows runner takes over 17 min, which the
    # per-test limit (+120 s) used to cut off at ~95%, so every Windows run read 0 collected.
    ap.add_argument("--suite-timeout", type = int, default = 3600, help = "whole pytest run, seconds")
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
    # The junit file is the count of record: nothing the code under test prints can bury it.
    junit = args.out.resolve().with_name(args.out.stem + ".junit.xml")  # pytest runs from workdir
    junit.unlink(missing_ok = True)
    cmd = [args.python, "-m", "pytest", "-q", "-rfE", "--color=no",
           f"--junitxml={junit}", "-o", "junit_family=xunit1"]
    if have_timeout:
        cmd += ["--timeout", str(args.timeout)]
    cmd += present
    obs["cmd"] = " ".join(cmd)
    # Wall-clock ceiling regardless, so a hang is still bounded without the plugin.
    # Own process group on POSIX, so a timeout also kills pytest's children (servers, xdist workers)
    # instead of leaving them on a runner that other jobs share.
    p = subprocess.Popen(cmd, cwd = workdir, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True,
                         start_new_session = os.name == "posix")
    try:
        out, err = p.communicate(timeout = args.suite_timeout)
        rc, out, err = p.returncode, out or "", err or ""
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            # TERM first so pytest's teardown (fixtures stopping servers) runs; KILL what is left.
            for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 0)):
                try:
                    os.killpg(p.pid, sig)
                except (ProcessLookupError, PermissionError):
                    break
                try:
                    p.wait(timeout = grace)
                except subprocess.TimeoutExpired:
                    pass
        else:
            p.kill()
        try:
            out, _ = p.communicate(timeout = 30)
        except subprocess.TimeoutExpired:
            out = ""
        rc, out, err = -1, out or "", f"TimeoutExpired: the whole run exceeded --suite-timeout {args.suite_timeout}s"
    out = _ANSI_RE.sub("", out or "")
    obs["rc"] = rc
    # stderr is where pytest writes usage errors. Not capturing it is what made
    # this failure invisible.
    obs["stderr_tail"] = (err or "")[-2000:]
    args.out.with_name(args.out.stem + ".stdout.txt").write_text(
        out[-5_000_000:], encoding = "utf-8")
    # Code under test can print after the summary (20 kB of atexit logging on gfx1151 made the
    # tail read 0 collected), so the summary line is searched for anywhere; junit decides.
    summaries = list(_SUMMARY_RE.finditer(out))
    summary = summaries[-1] if summaries else None
    end = summary.end() if summary else len(out)
    obs["summary_line"] = summary.group(0).strip("= \n") if summary else None
    obs["tail"] = out[max(0, end - 4000):end]
    report = out[:end]
    obs["failed"] = sorted(set(re.findall(r"^FAILED\s+(\S+)", report, re.M)))
    obs["errors"] = sorted(set(re.findall(r"^ERROR\s+(\S+)", report, re.M)))
    counts = summary.group("counts") if summary else ""
    for key, word in (("n_failed", "failed"), ("n_passed", "passed"),
                      ("n_skipped", "skipped"), ("n_error", "errors?")):
        m = re.search(rf"(\d+) {word}\b", counts)
        obs[key] = int(m.group(1)) if m else 0
    parsed = _read_junit(junit, Path(workdir).resolve())
    obs["counts_from"] = "junit" if parsed else ("summary line" if summary else "nothing")
    if parsed:
        obs.update(parsed)
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


def _collected_module(cls: str, roots: list[Path]) -> tuple[str, list[str]] | None:
    """Split a junit classname into the collected test file and its classes. The classname comes from the
    node id; `file` is where the method is defined, which differs for a method inherited from another module."""
    parts = cls.split(".")
    for i in range(len(parts), 0, -1):
        rel = "/".join(parts[:i]) + ".py"
        if any((r / rel).is_file() for r in roots):
            return rel, parts[i:]
    return None


def _read_junit(path: Path, workdir: Path | None = None) -> dict | None:
    """Counts and failing node ids from pytest's junit file (xunit1, which carries `file`)."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    roots = [] if workdir is None else [workdir, *list(workdir.parents)[:3]]
    res = {"n_passed": 0, "n_failed": 0, "n_skipped": 0, "n_error": 0, "failed": set(), "errors": set()}
    for case in root.iter("testcase"):
        file = (case.get("file") or "").replace("\\", "/")
        cls, name = case.get("classname") or "", case.get("name") or ""
        if file:
            module = file[:-3].replace("/", ".") if file.endswith(".py") else file
            if cls.startswith(module + ".") or cls == module:
                nodeid = "::".join([file, *([p for p in cls[len(module) + 1:].split(".") if p]), name])
            else:  # inherited from another module: identify by the collected class, never just the definition
                found = _collected_module(cls, roots)
                nodeid = "::".join([found[0], *found[1], name] if found else [file, cls, name])
        else:  # a collection error names the module, not a test
            nodeid = name
        kids = {child.tag for child in case}
        if "error" in kids:
            res["n_error"] += 1
            res["errors"].add(nodeid)
        elif "failure" in kids:
            res["n_failed"] += 1
            res["failed"].add(nodeid)
        elif "skipped" in kids:
            res["n_skipped"] += 1
        else:
            res["n_passed"] += 1
    res["failed"], res["errors"] = sorted(res["failed"]), sorted(res["errors"])
    return res


if __name__ == "__main__":
    sys.exit(main())
