#!/usr/bin/env python3
"""Branch-only states, for a change that has no PR yet: base = a pinned upstream commit,
head = base + a `git format-patch` file that travels with this CI branch.

Same output shape as states.py ({"commits", "paths"}), so differential.py consumes it as is.
The base is pinned (not `main`) so the comparison is against the commit the patch was made on,
not a moving target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from states import clone, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth")
    ap.add_argument("--base", required = True, help = "full upstream commit SHA the patch applies to")
    ap.add_argument("--patch", required = True, type = Path)
    ap.add_argument("--root", required = True, type = Path)
    ap.add_argument("--out", type = Path, default = None)
    args = ap.parse_args()

    src = clone(args.repo, args.root / "repo")
    run(["git", "fetch", "-q", "origin", args.base], cwd = src, check = False)
    base = run(["git", "rev-parse", f"{args.base}^{{commit}}"], cwd = src)
    paths = {}
    for name in ("base", "head"):
        wt = args.root / "states" / name
        if not wt.exists():
            run(["git", "worktree", "add", "-q", "--detach", str(wt), base], cwd = src)
        paths[name] = str(wt)
    head_wt = Path(paths["head"])
    if run(["git", "rev-parse", "HEAD"], cwd = head_wt) == base:
        run(["git", "-c", "user.name=amd-ci", "-c", "user.email=amd-ci@localhost", "am", "-q", "--3way",
             str(args.patch.resolve())], cwd = head_wt)
    head = run(["git", "rev-parse", "HEAD"], cwd = head_wt)
    if head == base:
        raise SystemExit("the patch did not produce a head commit")
    changed = run(["git", "diff", "--stat", base, head], cwd = head_wt)
    states = {"base": base, "head": head}
    for name, sha in states.items():
        print(f"{name:6s} {sha[:9]}  {paths[name]}")
    print(changed)
    doc = {"pr": 0, "merged": False, "commits": states, "paths": paths, "patch": str(args.patch)}
    if args.out:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(doc, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
