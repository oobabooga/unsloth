#!/usr/bin/env python3
"""Probe: run a set of standalone bash test scripts in a checkout.

Observes only. `tests/sh/*.sh` are not pytest, so pytest_probe.py cannot reach
them, and they are where install.sh's ROCm arch gate, torch index constraint,
uv cache colocation and shell-rc handling are actually asserted.

The observation shape deliberately matches pytest_probe.py -- `failed` is a list
of ids, not a count -- so criteria/pytest_no_regression.py can judge it unchanged.
Comparing counts would report a change that ADDS a passing test as a difference.

A script absent at this state is recorded as absent rather than failed: a test
file the change introduces does not exist at the base, and calling that a failure
would invert the reading.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--timeout", type = int, default = 600)
    ap.add_argument("--tests", nargs = "+", required = True)
    ap.add_argument("--scripts-from", default = "",
                    help = "copy each script from this checkout before running it, so a "
                           "test the change ADDS is run against the OLD implementation "
                           "too. Without it a new test is merely absent at the base, "
                           "which shows nothing about whether the base was broken.")
    args = ap.parse_args()

    # Resolved: the scripts are invoked with cwd = root, so a relative checkout
    # path would be re-anchored against itself and every script read as missing.
    root = Path(args.checkout).resolve()
    obs: dict = {"state": args.state, "workdir": str(root), "tests": args.tests}

    # The scripts locate the implementation as $SCRIPT_DIR/../../install.sh, so a
    # script copied into THIS checkout tests THIS checkout's install.sh. That is
    # the whole mechanism: same assertions, older implementation.
    ported = []
    if args.scripts_from:
        source = Path(args.scripts_from).resolve()
        for rel in args.tests:
            src, dst = source / rel, root / rel
            if src.is_file() and src.resolve() != dst.resolve():
                dst.parent.mkdir(parents = True, exist_ok = True)
                if not dst.is_file() or dst.read_bytes() != src.read_bytes():
                    dst.write_bytes(src.read_bytes())
                    ported.append(rel)
    obs["ported_from"] = args.scripts_from
    obs["ported"] = ported

    present, absent = [], []
    for rel in args.tests:
        (present if (root / rel).is_file() else absent).append(rel)
    obs["absent_at_this_state"] = absent
    obs["selected"] = present
    if not present:
        obs["note"] = "no selected scripts exist at this state"
        obs["rc"] = 5                       # pytest's "nothing collected", so the
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0                            # shared criteria gate catches it

    failed, passed, tails = [], [], {}
    for rel in present:
        try:
            proc = subprocess.run(
                ["bash", str(root / rel)], cwd = root, capture_output = True,
                text = True, timeout = args.timeout,
            )
            rc, out = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            rc, out = -1, "TimeoutExpired"
        (passed if rc == 0 else failed).append(rel)
        if rc != 0:
            tails[rel] = out[-3000:]

    obs["failed"] = failed
    obs["errors"] = []
    obs["n_passed"] = len(passed)
    obs["n_failed"] = len(failed)
    obs["n_skipped"] = 0
    # 0 and 1 are the only codes criteria/pytest_no_regression.py treats as "the
    # suite ran and judged something", so the summary code has to speak that.
    obs["rc"] = 1 if failed else 0
    obs["tails"] = tails
    obs["tail"] = "\n".join(f"--- {k}\n{v}" for k, v in tails.items())[-4000:]

    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
