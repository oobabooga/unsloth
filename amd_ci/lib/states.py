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
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


COMMAND_TIMEOUT_S = 300


def invoke(cmd: list[str], cwd: Path | None = None, *, timeout: float = COMMAND_TIMEOUT_S) -> subprocess.CompletedProcess:
    # Files avoid communicate() waiting forever on a pipe inherited by a surviving Git child.
    with tempfile.TemporaryFile(mode = "w+", encoding = "utf-8") as out, tempfile.TemporaryFile(mode = "w+", encoding = "utf-8") as err:
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        process = subprocess.Popen(cmd, cwd = cwd, stdout = out, stderr = err, **options)
        try:
            code = process.wait(timeout = timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                try:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL, timeout = 10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                if process.poll() is None:
                    process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout = 5)
            except subprocess.TimeoutExpired:
                pass
            return subprocess.CompletedProcess(cmd, 124, "", f"timed out after {timeout}s")
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(cmd, code, out.read(), err.read())


def execute(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    progress = cmd[:2] in (["git", "clone"], ["git", "fetch"], ["git", "worktree"])
    started = time.monotonic()
    if progress:
        print(f"states: {' '.join(cmd[:3])} starting (timeout {COMMAND_TIMEOUT_S}s)",
              file = sys.stderr, flush = True)
    try:
        result = invoke(cmd, cwd = cwd, timeout = COMMAND_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        result = subprocess.CompletedProcess(cmd, 124, "", f"timed out after {COMMAND_TIMEOUT_S}s")
    if progress:
        print(f"states: {' '.join(cmd[:2])} finished in {time.monotonic() - started:.1f}s "
              f"(exit {result.returncode})", file = sys.stderr, flush = True)
    return result


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    p = execute(cmd, cwd = cwd)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def is_network_url(repo: str) -> bool:
    return repo.startswith(("https://", "http://", "ssh://", "git://", "git@"))


def clone(repo: str, dest: Path, attempts: int = 3) -> Path:
    if dest.exists():
        return dest
    # Fetch PR refs explicitly below; avoid every branch and the initial checkout's blob download.
    # Never filter a local source: it does not honour the filter, the clone still registers it as a
    # promisor, and git 2.43 then lazy-fetches recursively from it until the host OOMs (2026-10-21).
    # A plain local clone hardlinks objects, so it is already cheap.
    cmd = ["git", "clone", "-q"] + (
        ["--single-branch", "--no-checkout", "--no-tags", "--filter=blob:none"]
        if is_network_url(repo) else []
    ) + [repo, str(dest)]
    for attempt in range(1, attempts + 1):
        p = execute(cmd)
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
    merge = execute(["git", "fetch", "-q", "origin", f"refs/pull/{pr}/merge:pr{pr}-merge"], cwd = src)
    if merge.returncode == 124:
        raise SystemExit(f"merge ref fetch timed out: {merge.stderr}")
    have_merge = merge.returncode == 0
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


def pinned_states(repo: str, root: Path, base: str, head: str) -> tuple[Path, Path, dict[str, str]]:
    namespace = root / "pinned" / f"{base[:12]}-{head[:12]}"
    namespace.mkdir(parents = True, exist_ok = True)
    attempt = Path(tempfile.mkdtemp(prefix = "source-", dir = namespace))
    src = attempt / "repo"
    run(["git", "init", "-q", str(src)])
    run(["git", "remote", "add", "origin", repo], cwd = src)
    filters = []
    if is_network_url(repo):
        run(["git", "config", "remote.origin.promisor", "true"], cwd = src)
        run(["git", "config", "remote.origin.partialclonefilter", "blob:none"], cwd = src)
        filters = ["--filter=blob:none"]
    run(["git", "fetch", "--depth=1", "--no-tags", *filters, "origin",
         f"{base}:refs/pinned/base", f"{head}:refs/pinned/head"], cwd = src)
    for name, sha in (("base", base), ("head", head)):
        actual = run(["git", "rev-parse", f"refs/pinned/{name}^{{commit}}"], cwd = src)
        if actual != sha:
            raise SystemExit(f"pinned {name} resolved to {actual}, expected {sha}")
    return src, attempt, {"base": base, "head": head}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--pr", type = int, required = True)
    ap.add_argument("--merged", action = "store_true",
                    help = "the PR is already in main; compare the merge commit to its parent")
    ap.add_argument("--root", required = True, type = Path)
    ap.add_argument("--out", type = Path, default = None)
    ap.add_argument("--base-sha", help = "immutable merge base already verified by the dispatcher")
    ap.add_argument("--head-sha", help = "with --base-sha: fetch only these two commits; no runner merge-base calculation")
    ap.add_argument("--patch-base", default = None,
                    help = "with --patch: base = this commit, head = this commit plus the patch (a change that is "
                           "not a PR yet, e.g. a fix branch's diff against main)")
    ap.add_argument("--patch", type = Path, default = None, help = "git diff to apply on --patch-base for head")
    args = ap.parse_args()
    if bool(args.base_sha) != bool(args.head_sha):
        ap.error("--base-sha and --head-sha go together")
    if args.base_sha:
        if not all(re.fullmatch(r"[0-9a-fA-F]{40}", value) for value in (args.base_sha, args.head_sha)):
            ap.error("--base-sha and --head-sha require full 40-hex SHAs")
        if args.merged or args.patch or args.patch_base:
            ap.error("pinned SHAs cannot be combined with --merged or --patch")
        args.base_sha, args.head_sha = args.base_sha.lower(), args.head_sha.lower()
    if bool(args.patch) != bool(args.patch_base):
        raise SystemExit("--patch and --patch-base go together")
    if args.patch and args.merged:
        raise SystemExit("--patch compares a commit with itself plus a diff; --merged does not apply")
    # git worktree add runs with cwd = the clone, so a relative root would land inside it.
    args.root = args.root.resolve()

    work_root = args.root
    if args.base_sha:
        src, work_root, states = pinned_states(args.repo, args.root, args.base_sha, args.head_sha)
    else:
        src = clone(args.repo, args.root / "repo")
    if args.patch:
        base_sha = run(["git", "rev-parse", f"{args.patch_base}^{{commit}}"], cwd = src)
        states = {"base": base_sha, "head": base_sha}
    elif not args.base_sha:
        states = merged_pr_states(src, args.pr) if args.merged else open_pr_states(src, args.pr)

    paths: dict[str, str] = {}
    for name, sha in states.items():
        wt = work_root / "states" / name
        if not wt.exists():
            run(["git", "worktree", "add", "-q", "--detach", str(wt), sha], cwd = src)
        if name == "head" and args.patch:
            run(["git", "apply", "--index", str(args.patch.resolve())], cwd = wt)
            run(["git", "-c", "user.name=amd-ci", "-c", "user.email=amd-ci@localhost", "commit", "-q", "-m",
                 f"head: {states['base'][:9]} + {args.patch.name}"], cwd = wt)
            sha = states["head"] = run(["git", "rev-parse", "HEAD"], cwd = wt)
            print(run(["git", "show", "--stat", "--format=%s", "HEAD"], cwd = wt))
        paths[name] = str(wt)
        print(f"{name:6s} {sha[:9]}  {wt}", flush = True)

    doc = {"pr": args.pr, "merged": args.merged, "commits": states, "paths": paths}
    if args.base_sha:
        doc["provenance"] = {"mode": "pinned_commits", "base_dispatcher_verified": True,
                             "merge_base_computed_on_runner": False, "repo": args.repo,
                             "base_sha": args.base_sha, "head_sha": args.head_sha}
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
