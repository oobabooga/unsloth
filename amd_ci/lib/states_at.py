#!/usr/bin/env python3
"""Resolve an arbitrary base/head commit pair as worktrees.

`states.py` answers "this PR against its merge base", which is the right pair for
judging a PR. It is the wrong pair for judging a FIX THAT LANDED INSIDE a PR: the
merge base predates the whole feature, so the base leg has nothing to exhibit and
the differential is INCONCLUSIVE rather than informative.

This resolves the two commits that actually bracket such a change. Same output
contract as `states.py`, so `differential.py` consumes it unchanged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    p = subprocess.run(cmd, cwd = cwd, capture_output = True, text = True)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--fetch-ref", action = "append", default = [],
                    help = "extra refspec to fetch, e.g. refs/pull/10618/head:pr-head. "
                           "Repeatable. Needed because the commits may live only on a "
                           "fork's PR branch and not on any branch of --repo.")
    ap.add_argument("--base", required = True, help = "commit that should exhibit the defect")
    ap.add_argument("--head", required = True, help = "commit that should not")
    ap.add_argument("--root", required = True, type = Path)
    ap.add_argument("--out", type = Path, default = None)
    args = ap.parse_args()
    # Absolute, because the paths are handed to a probe that runs elsewhere and a
    # relative one resolves against the probe's cwd instead of this script's.
    args.root = args.root.resolve()

    src = args.root / "repo"
    if not src.exists():
        run(["git", "clone", "-q", "--filter=blob:none", args.repo, str(src)])
    for spec in args.fetch_ref:
        run(["git", "fetch", "-q", "origin", spec], cwd = src)

    states = {}
    for name, rev in (("base", args.base), ("head", args.head)):
        states[name] = run(["git", "rev-parse", rev], cwd = src)

    paths: dict[str, str] = {}
    for name, sha in states.items():
        wt = args.root / "states" / name
        if not wt.exists():
            run(["git", "worktree", "add", "-q", "--detach", str(wt), sha], cwd = src)
        paths[name] = str(wt)
        print(f"{name:6s} {sha[:9]}  {wt}")

    doc = {"base_rev": args.base, "head_rev": args.head, "commits": states, "paths": paths}
    if args.out:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(doc, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
