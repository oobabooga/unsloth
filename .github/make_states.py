#!/usr/bin/env python3
"""Fixed states for the on-demand SSM check: base = 9f5b13325 (setup-time kernels), main =
origin/main, head = PR 13971 head 7092b2bdf. Writes <work>/out/states.json for differential.py."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SHAS = {
    "base": "9f5b133254e79a044cc80ff948f05cc08d79ba44",
    "main": "origin/main",
    "head": "7092b2bdf2e26f3701165a8e2190d0e9cd7c1692",
}


def git(*a):
    print("+ git", *a, flush = True)
    subprocess.run(["git", *a], check = True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--only", default = "base,main,head")
    args = ap.parse_args()
    w = Path(args.work)
    repo = w / "repo"
    (w / "out").mkdir(parents = True, exist_ok = True)
    (w / "states").mkdir(parents = True, exist_ok = True)
    git("-c", "core.longpaths=true", "clone", "-q", "--filter=blob:none",
        "https://github.com/unslothai/unsloth", str(repo))
    git("-C", str(repo), "config", "core.longpaths", "true")
    git("-C", str(repo), "fetch", "-q", "origin", "pull/13971/head:pr13971")
    paths, commits = {}, {}
    for name in args.only.split(","):
        p = w / "states" / name
        git("-C", str(repo), "worktree", "add", "-q", "--detach", str(p), SHAS[name])
        paths[name] = str(p)
        commits[name] = subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output = True,
                                       text = True).stdout.strip()
    (w / "out" / "states.json").write_text(json.dumps(
        {"pr": 13971, "merged": False, "commits": commits, "paths": paths}, indent = 2), encoding = "utf-8")
    print(commits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
