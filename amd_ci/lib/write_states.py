#!/usr/bin/env python3
"""Write a states.json for two directories that already exist (a prebuilt
differential has no worktrees). Refuses a missing directory, a missing
`--require-file`, or two states that resolve to one place.

  python amd_ci/lib/write_states.py --require-file llama-server \\
      --path "base=$ROCM_BASE_BIN" --path "head=$ROCM_HEAD_BIN" \\
      --label "base=b10715 rocm-gfx1151" --label "head=b10798 rocm-gfx1151" \\
      --out "$AMD_CI_WORK/out/states.json"
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
    args.out.write_text(json.dumps(doc, indent = 2), encoding = "utf-8")
    print(json.dumps(doc, indent = 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
