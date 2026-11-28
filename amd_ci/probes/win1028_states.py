#!/usr/bin/env python3
"""states.json for two plain commits (base = a fixed SHA, head = origin/<branch> tip), as worktrees.

states.py resolves PR / merged-PR / --patch states; its --patch mode is Linux-only in scaffold.py (git apply over a
CRLF checkout), so on Windows the head is a real commit instead. Same JSON shape as states.py writes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from states import clone, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--base", required = True)
    ap.add_argument("--head-branch", default = "main")
    ap.add_argument("--min-head", default = None, help = "head must contain this commit")
    ap.add_argument("--root", required = True, type = Path)
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()
    args.root = args.root.resolve()
    src = clone(args.repo, args.root / "repo")
    run(["git", "fetch", "-q", "origin", args.head_branch], cwd = src)
    states = {"base": run(["git", "rev-parse", f"{args.base}^{{commit}}"], cwd = src),
              "head": run(["git", "rev-parse", f"origin/{args.head_branch}"], cwd = src)}
    if args.min_head:
        ok = subprocess.run(["git", "merge-base", "--is-ancestor", args.min_head, states["head"]], cwd = src).returncode == 0
        if not ok:
            raise SystemExit(f"head {states['head'][:10]} does not contain {args.min_head}")
    paths = {}
    for name, sha in states.items():
        wt = args.root / "states" / name
        if not wt.exists():
            run(["git", "worktree", "add", "-q", "--detach", str(wt), sha], cwd = src)
        paths[name] = str(wt)
        print(f"{name:6s} {sha[:10]}  {wt}")
    doc = {"pr": 0, "merged": False, "commits": states, "paths": paths}
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(doc, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
