#!/usr/bin/env python3
"""States that are RELEASE TAGS rather than git commits.

`states.py` answers "what did this repository look like before and after a PR".
Some defects are not in this repository at all: unsloth#7371 is a throughput
collapse between two `unslothai/llama.cpp` prebuilts that carry the IDENTICAL
Unsloth patch mix (`fb3d4ca` either side), so a worktree differential would
compare two identical trees and could only ever answer NO_REGRESSION.

So the states are named builds. The probe is handed the tag through its own
`--tag-map`; `--checkout` still points at the repository checkout, because a
probe may want the tree for anything else and the differential runner always
passes one.

The contract with `differential.py` is only the shape of the JSON, so this is a
few lines rather than a subclass of anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", action = "append", required = True, metavar = "NAME=TAG",
                    help = "repeatable; order is preserved and `base` must come first")
    ap.add_argument("--checkout", required = True,
                    help = "path handed to every probe as --checkout")
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    commits: dict[str, str] = {}
    for raw in args.state:
        name, sep, tag = raw.partition("=")
        if not sep or not name.strip() or not tag.strip():
            raise SystemExit(f"--state wants NAME=TAG, got {raw!r}")
        commits[name.strip()] = tag.strip()

    if "base" not in commits or "head" not in commits:
        raise SystemExit("a differential needs both a `base` and a `head` state")

    doc = {
        "states_are": "release tags, not commits",
        "commits": commits,
        "paths": {name: args.checkout for name in commits},
    }
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(doc, indent = 2), encoding = "utf-8")
    for name, tag in commits.items():
        print(f"{name:6s} {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
