#!/usr/bin/env python3
"""Resolve states from EXPLICIT refs, as worktrees.

`states.py` answers "this PR, against its base". That is the right shape when the
question is "does this PR do what it says". It is the wrong shape when the
question is "does the defect still exist on what ships TODAY", because the head
it produces is the merge commit of a PR that landed weeks ago and everything
merged since is invisible to it.

This resolves whatever refs it is given:

    states_at.py --root R --out R/out/states.json \\
        base=16d596d53^ head=origin/main

The output is the same shape `differential.py` consumes, so the criteria and
probes do not know or care which resolver ran. Names other than base/head/merge
are rejected: `differential.py` and every criteria module are written against
those three, and a fourth state would be silently dropped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

KNOWN = ("base", "head", "merge")


def run(cmd: list[str], cwd: Path | None = None) -> str:
    p = subprocess.run(cmd, cwd = cwd, capture_output = True, text = True)
    if p.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--root", required = True, type = Path)
    ap.add_argument("--out", type = Path, default = None)
    ap.add_argument("--label", default = "", help = "recorded in the output for the report")
    ap.add_argument("refs", nargs = "+", help = "name=ref pairs, e.g. base=abc123^ head=origin/main")
    args = ap.parse_args()

    src = args.root / "repo"
    if not src.exists():
        # NOT --filter=blob:none here: states.py can afford it because it fetches
        # the PR refs it needs afterwards, while an arbitrary ref may be any age
        # and a partial clone would fault blobs in one at a time over the network
        # for every file a worktree checks out.
        run(["git", "clone", "-q", args.repo, str(src)])
    run(["git", "fetch", "-q", "origin", "+refs/heads/*:refs/remotes/origin/*"], cwd = src)

    wanted: dict[str, str] = {}
    for pair in args.refs:
        name, _, ref = pair.partition("=")
        name = name.strip()
        if name not in KNOWN:
            raise SystemExit(f"state name {name!r} is not one of {KNOWN}")
        if not ref.strip():
            raise SystemExit(f"no ref given for state {name!r}")
        wanted[name] = ref.strip()
    if "base" not in wanted or "head" not in wanted:
        raise SystemExit("both base= and head= are required")

    commits: dict[str, str] = {}
    paths: dict[str, str] = {}
    for name, ref in wanted.items():
        sha = run(["git", "rev-parse", f"{ref}^{{commit}}"], cwd = src)
        commits[name] = sha
        wt = args.root / "states" / name
        if not wt.exists():
            run(["git", "worktree", "add", "-q", "--detach", str(wt), sha], cwd = src)
        paths[name] = str(wt)
        print(f"{name:6s} {sha[:9]}  {ref}  {wt}")

    if commits.get("base") == commits.get("head"):
        # Not fatal here, but it makes every differential VOID downstream, and
        # saying so at resolve time is cheaper than reading it out of a verdict.
        print("WARNING: base and head resolve to the SAME commit; "
              "any differential over these states is VOID by construction")

    doc = {"pr": None, "merged": False, "label": args.label,
           "refs": wanted, "commits": commits, "paths": paths}
    if args.out:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(doc, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
