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
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    p = subprocess.run(cmd, cwd = cwd, capture_output = True, text = True)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def is_network_url(repo: str) -> bool:
    return repo.startswith(("https://", "http://", "ssh://", "git://", "git@"))


def clone(repo: str, dest: Path, attempts: int = 3) -> Path:
    if dest.exists():
        return dest
    # blob:none keeps it small; runner networks drop long transfers (curl 18 / 56), so wipe and retry.
    # Never filter a local source: it does not honour the filter, the clone still registers it as a
    # promisor, and git 2.43 then lazy-fetches recursively from it until the host OOMs (2026-10-21).
    # A plain local clone hardlinks objects, so it is already cheap.
    cmd = ["git", "clone", "-q"] + (["--filter=blob:none"] if is_network_url(repo) else []) + [repo, str(dest)]
    for attempt in range(1, attempts + 1):
        p = subprocess.run(cmd, capture_output = True, text = True)
        if p.returncode == 0:
            return dest
        shutil.rmtree(dest, ignore_errors = True)
        if attempt == attempts:
            raise SystemExit(f"command failed after {attempts} attempts: git clone {repo}\n{p.stderr.strip()}")
        print(f"git clone attempt {attempt} failed, retrying: {p.stderr.strip()[-300:]}", flush = True)
        time.sleep(15 * attempt)
    return dest


def default_branch(src: Path) -> str:
    """origin's default branch: `main` here, `master` on a llama.cpp fork."""
    ref = run(["git", "symbolic-ref", "-q", "refs/remotes/origin/HEAD"], cwd = src, check = False)
    return ref.removeprefix("refs/remotes/") if ref else "origin/main"


def open_pr_states(src: Path, pr: int) -> dict[str, str]:
    """base / head / merge for an open PR."""
    run(["git", "fetch", "-q", "origin", f"refs/pull/{pr}/head:pr{pr}-head"], cwd = src)
    # The merge ref only exists while GitHub considers the PR mergeable.
    have_merge = subprocess.run(
        ["git", "fetch", "-q", "origin", f"refs/pull/{pr}/merge:pr{pr}-merge"],
        cwd = src, capture_output = True,
    ).returncode == 0
    base = run(["git", "merge-base", default_branch(src), f"pr{pr}-head"], cwd = src)
    states = {"base": base, "head": run(["git", "rev-parse", f"pr{pr}-head"], cwd = src)}
    if have_merge:
        states["merge"] = run(["git", "rev-parse", f"pr{pr}-merge"], cwd = src)
    return states


def merged_pr_states(src: Path, pr: int) -> dict[str, str]:
    """merge commit and its first parent, for a PR already in main."""
    main = default_branch(src)
    sha = run(["git", "log", main, "--format=%H", f"--grep=(#{pr})", "-1"], cwd = src)
    if not sha:
        raise SystemExit(f"cannot find a merge commit for #{pr} in {main}")
    return {"base": run(["git", "rev-parse", f"{sha}^"], cwd = src), "head": sha}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--pr", type = int, required = True)
    ap.add_argument("--merged", action = "store_true",
                    help = "the PR is already in main; compare the merge commit to its parent")
    ap.add_argument("--root", required = True, type = Path)
    ap.add_argument("--out", type = Path, default = None)
    ap.add_argument("--patch-base", default = None,
                    help = "with --patch: base = this commit, head = this commit plus the patch (a change that is "
                           "not a PR yet, e.g. a fix branch's diff against main)")
    ap.add_argument("--patch", type = Path, default = None, help = "git diff to apply on --patch-base for head")
    args = ap.parse_args()
    if bool(args.patch) != bool(args.patch_base):
        raise SystemExit("--patch and --patch-base go together")
    if args.patch and args.merged:
        raise SystemExit("--patch compares a commit with itself plus a diff; --merged does not apply")
    # git worktree add runs with cwd = the clone, so a relative root would land inside it.
    args.root = args.root.resolve()

    src = clone(args.repo, args.root / "repo")
    if args.patch:
        base_sha = run(["git", "rev-parse", f"{args.patch_base}^{{commit}}"], cwd = src)
        states = {"base": base_sha, "head": base_sha}
    else:
        states = merged_pr_states(src, args.pr) if args.merged else open_pr_states(src, args.pr)

    paths: dict[str, str] = {}
    for name, sha in states.items():
        wt = args.root / "states" / name
        if not wt.exists():
            run(["git", "worktree", "add", "-q", "--detach", str(wt), sha], cwd = src)
        if name == "head" and args.patch:
            run(["git", "apply", "--index", str(args.patch.resolve())], cwd = wt)
            run(["git", "-c", "user.name=amd-ci", "-c", "user.email=amd-ci@localhost", "commit", "-q", "-m",
                 f"head: {states['base'][:9]} + {args.patch.name}"], cwd = wt)
            sha = states["head"] = run(["git", "rev-parse", "HEAD"], cwd = wt)
            print(run(["git", "show", "--stat", "--format=%s", "HEAD"], cwd = wt))
        paths[name] = str(wt)
        print(f"{name:6s} {sha[:9]}  {wt}")

    doc = {"pr": args.pr, "merged": args.merged, "commits": states, "paths": paths}
    if args.patch:
        import hashlib

        doc["patch"] = {"file": args.patch.name, "base": states["base"],
                        "sha256": hashlib.sha256(args.patch.read_bytes()).hexdigest()}
    if args.out:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(doc, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
