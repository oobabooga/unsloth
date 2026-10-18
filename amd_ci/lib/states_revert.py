#!/usr/bin/env python3
"""states.json for a differential where base = head with a list of commits reverted.

Same output shape as amd_ci/lib/states.py (paths.base / paths.head), so differential.py
reads it unchanged. Used for re-verifying already-merged PRs as a group: head is a pinned
main SHA, base is that SHA with the group's squash commits reverted (newest first).
Optionally also checks out a companion repo at a pinned SHA (e.g. unsloth-zoo head) for
PYTHONPATH overlays.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd = None):
    print("+", " ".join(map(str, cmd)), flush = True)
    return subprocess.run(cmd, cwd = cwd, check = True, capture_output = True, text = True).stdout.strip()


def clone(repo, dest):
    if not dest.exists():
        run(["git", "clone", "-q", "--filter=blob:none", repo, str(dest)])
    run(["git", "fetch", "-q", "origin"], cwd = dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required = True)
    ap.add_argument("--head", required = True, help = "pinned head SHA")
    ap.add_argument("--revert", nargs = "+", required = True, help = "commits to revert, newest first")
    ap.add_argument("--root", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--companion-repo", default = None)
    ap.add_argument("--companion-sha", default = None)
    ap.add_argument("--companion-dir", type = Path, default = None)
    args = ap.parse_args()

    src = clone(args.repo, args.root / "repo")
    paths = {}
    for name in ("head", "base"):
        wt = args.root / "states" / name
        if not wt.exists():
            run(["git", "worktree", "add", "-q", "--detach", str(wt), args.head], cwd = src)
        if name == "base":
            for c in args.revert:
                run(["git", "-c", "user.name=ci", "-c", "user.email=ci@ci", "revert", "--no-edit", c], cwd = wt)
        paths[name] = str(wt)
    commits = {n: run(["git", "rev-parse", "HEAD"], cwd = Path(p)) for n, p in paths.items()}
    for n in paths:
        print(f"{n:6s} {commits[n][:9]}  {paths[n]}")
    if args.companion_repo:
        d = args.companion_dir
        clone(args.companion_repo, d)
        run(["git", "checkout", "-q", args.companion_sha], cwd = d)
        print(f"companion {args.companion_sha[:9]}  {d}")
    doc = {"pr": 0, "merged": True, "commits": commits, "paths": paths,
           "reverted": args.revert, "head_sha": args.head}
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(doc, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
