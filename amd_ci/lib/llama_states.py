#!/usr/bin/env python3
"""Resolve base / head for a llama.cpp patch, as worktrees.

base = the public ggml-org/llama.cpp at a pinned commit.
head = base + a patch committed inside this CI branch (git am, so head is a real commit).

Emits the same states.json shape as lib/states.py, which differential.py consumes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> str:
    p = subprocess.run(cmd, cwd = cwd, capture_output = True, text = True)
    if p.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{p.stdout.strip()}\n{p.stderr.strip()}")
    return p.stdout.strip()


def clone(repo: str, dest: Path, attempts: int = 3) -> Path:
    if dest.exists():
        return dest
    for attempt in range(1, attempts + 1):
        p = subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", repo, str(dest)],
                           capture_output = True, text = True)
        if p.returncode == 0:
            return dest
        shutil.rmtree(dest, ignore_errors = True)
        if attempt == attempts:
            raise SystemExit(f"git clone {repo} failed after {attempts} attempts\n{p.stderr.strip()}")
        print(f"git clone attempt {attempt} failed, retrying: {p.stderr.strip()[-300:]}", flush = True)
        time.sleep(15 * attempt)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default = "https://github.com/ggml-org/llama.cpp")
    ap.add_argument("--base-sha", required = True)
    ap.add_argument("--patch", required = True, type = Path)
    ap.add_argument("--root", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    src = clone(args.repo, args.root / "repo")
    base = run(["git", "rev-parse", "--verify", f"{args.base_sha}^{{commit}}"], cwd = src)

    paths: dict[str, str] = {}
    wt_base = args.root / "states" / "base"
    wt_head = args.root / "states" / "head"
    if not wt_base.exists():
        run(["git", "worktree", "add", "-q", "--detach", str(wt_base), base], cwd = src)
    if not wt_head.exists():
        run(["git", "worktree", "add", "-q", "--detach", str(wt_head), base], cwd = src)
        run(["git", "-c", "user.name=amd-ci", "-c", "user.email=amd-ci@localhost",
             "am", "-q", "--3way", str(args.patch.resolve())], cwd = wt_head)
    head = run(["git", "rev-parse", "HEAD"], cwd = wt_head)
    if head == base:
        raise SystemExit("the patch produced no commit; head == base")
    changed = run(["git", "diff", "--stat", base, head], cwd = wt_head)
    print(changed)
    paths["base"] = str(wt_base)
    paths["head"] = str(wt_head)
    print(f"base   {base[:9]}  {wt_base}")
    print(f"head   {head[:9]}  {wt_head}")

    doc = {"pr": 0, "merged": False, "commits": {"base": base, "head": head}, "paths": paths}
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(doc, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
