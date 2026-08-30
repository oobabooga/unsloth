#!/usr/bin/env python3
"""Write a states.json whose states are directories rather than git worktrees.

`lib/states.py` resolves a PR into base/head/merge checkouts. That is the usual
case and it does not fit here: the thing being compared is two GGUF revisions, or
two quant recipes of one revision, and there is no PR at all.

`lib/differential.py` never imports `lib/states.py`. The entire contract it reads
is `{"paths": {name: str}}`, optionally `{"commits": {name: str}}` for the header
line, so writing that directly is supported rather than a workaround.

The state names must be exactly `base` and `head`: differential.py looks those
up by name, and in regression mode any other key is ignored - which would show up
as a confident verdict computed from the wrong pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def kv(pairs: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--{what} wants name=value, got {p!r}")
        name, value = p.split("=", 1)
        out[name] = value
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", action = "append", default = [], metavar = "NAME=DIR")
    ap.add_argument("--label", action = "append", default = [], metavar = "NAME=TEXT")
    ap.add_argument("--require-file", default = "",
                    help = "filename that must exist in every state directory")
    ap.add_argument("--out", required = True, type = Path)
    args = ap.parse_args()

    paths = kv(args.path, "path")
    labels = kv(args.label, "label")

    if set(paths) != {"base", "head"}:
        raise SystemExit(f"states must be exactly base and head, got {sorted(paths)}")

    # A state directory that is missing its model produces a probe-level
    # setup_error two steps later, which reads as a defect. Catch it here, where
    # it is unambiguously a setup problem.
    problems = []
    for name, d in paths.items():
        p = Path(d)
        if not p.is_dir():
            problems.append(f"{name}: {d} is not a directory")
        elif args.require_file and not (p / args.require_file).is_file():
            problems.append(f"{name}: no {args.require_file} in {d}")
    resolved = {n: str(Path(d).resolve()) for n, d in paths.items()}
    if len(set(resolved.values())) != len(resolved):
        problems.append(f"base and head resolve to the same directory: {resolved}")
    if problems:
        raise SystemExit("; ".join(problems))

    doc = {"paths": resolved, "commits": {n: labels.get(n, n) for n in paths}}
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(doc, indent = 2))
    print(json.dumps(doc, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
