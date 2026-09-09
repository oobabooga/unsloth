#!/usr/bin/env python3
"""Resolve the states a differential needs, as worktrees.

For a PR that is "base, head, and the merge GitHub would produce". For a merged
PR it is "the merge commit and its first parent", because a post-merge check has
no head ref to fetch and comparing against `main` would compare against a moving
target that already contains the change.

Emits a JSON map of state name to checkout path, which the differential runner
consumes.
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


def clone(repo: str, dest: Path) -> Path:
    if not dest.exists():
        # blob:none keeps the clone small; the worktrees below fill in what they need.
        run(["git", "clone", "-q", "--filter=blob:none", repo, str(dest)])
    return dest


def open_pr_states(src: Path, pr: int) -> dict[str, str]:
    """base / head / merge for an open PR."""
    run(["git", "fetch", "-q", "origin", f"refs/pull/{pr}/head:pr{pr}-head"], cwd = src)
    # The merge ref only exists while GitHub considers the PR mergeable.
    have_merge = subprocess.run(
        ["git", "fetch", "-q", "origin", f"refs/pull/{pr}/merge:pr{pr}-merge"],
        cwd = src, capture_output = True,
    ).returncode == 0
    base = run(["git", "merge-base", "origin/main", f"pr{pr}-head"], cwd = src)
    states = {"base": base, "head": run(["git", "rev-parse", f"pr{pr}-head"], cwd = src)}
    if have_merge:
        states["merge"] = run(["git", "rev-parse", f"pr{pr}-merge"], cwd = src)
    return states


def merged_pr_states(src: Path, pr: int) -> dict[str, str]:
    """merge commit and its first parent, for a PR already in main."""
    sha = run(["git", "log", "origin/main", "--format=%H", f"--grep=(#{pr})", "-1"], cwd = src)
    if not sha:
        raise SystemExit(f"cannot find a merge commit for #{pr} in origin/main")
    return {"base": run(["git", "rev-parse", f"{sha}^"], cwd = src), "head": sha}


def explicit_states(src: Path, specs: list[str]) -> dict[str, str]:
    """name=<repo-url>#<ref> pairs, for a change that is not (yet) a PR.

    The branch under test may live on a fork or a staging repo while the PR is
    still being prepared, and a differential still needs a base to compare
    against. Each ref is fetched from its own repo into the one clone, so the
    worktrees below are created exactly as for a PR. `base` and `head` are
    required; anything else is an extra state judged like the head.
    """
    states: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec or "#" not in spec:
            raise SystemExit(f"--state wants name=<repo-url>#<ref>, got {spec!r}")
        name, rest = spec.split("=", 1)
        repo, ref = rest.rsplit("#", 1)
        local = f"amd-ci-state-{name}"
        run(["git", "fetch", "-q", repo, f"{ref}:refs/heads/{local}"], cwd = src)
        states[name] = run(["git", "rev-parse", local], cwd = src)
    for required in ("base", "head"):
        if required not in states:
            raise SystemExit(f"--state must include {required}=<repo-url>#<ref>")
    return states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--pr", type = int, default = 0,
                    help = "PR number on --repo; 0 with --state resolves explicit refs instead")
    ap.add_argument("--merged", action = "store_true",
                    help = "the PR is already in main; compare the merge commit to its parent")
    ap.add_argument("--state", action = "append", default = [], metavar = "NAME=REPO#REF",
                    help = "explicit state, repeatable; base= and head= are required when used")
    ap.add_argument("--root", required = True, type = Path)
    ap.add_argument("--out", type = Path, default = None)
    args = ap.parse_args()

    src = clone(args.repo, args.root / "repo")
    if args.state:
        states = explicit_states(src, args.state)
    elif args.pr:
        states = merged_pr_states(src, args.pr) if args.merged else open_pr_states(src, args.pr)
    else:
        raise SystemExit("need --pr N or --state name=repo#ref")

    paths: dict[str, str] = {}
    for name, sha in states.items():
        wt = args.root / "states" / name
        if not wt.exists():
            run(["git", "worktree", "add", "-q", "--detach", str(wt), sha], cwd = src)
        paths[name] = str(wt)
        print(f"{name:6s} {sha[:9]}  {wt}")

    doc = {"pr": args.pr, "merged": args.merged, "explicit": args.state,
           "commits": states, "paths": paths}
    if args.out:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(doc, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
