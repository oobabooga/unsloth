#!/usr/bin/env python3
"""Hardware gate. Fails the run rather than skipping it.

A skip reads as "nothing to see here" in the UI, which is the wrong summary for
"the machine we were asked to test on was not the machine we got". Every gate
failure here is exit 1.

Usage:
  python gate.py --require rocm gpu --expect-arch gfx1151 --out $AMD_CI_WORK/out/host.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from capability import detect  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require", nargs = "*", default = [],
                    help = "capabilities that must be present, else exit 1")
    ap.add_argument("--expect-arch", default = None,
                    help = "substring every GPU arch must contain, e.g. gfx1151")
    ap.add_argument("--min-gpus", type = int, default = 0)
    ap.add_argument("--out", type = Path, default = None)
    ap.add_argument("--no-torch", action = "store_true")
    args = ap.parse_args()

    p = detect(require_torch = not args.no_torch)
    print(p.to_json())
    if args.out:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(p.to_json())

    problems: list[str] = []
    for cap in args.require:
        if not p.capabilities.get(cap, False):
            problems.append(f"required capability missing: {cap}")
    if args.min_gpus and p.gpu_count < args.min_gpus:
        problems.append(f"need >= {args.min_gpus} GPUs, found {p.gpu_count}")
    if args.expect_arch:
        bad = [a for a in p.gpu_archs if args.expect_arch not in a]
        if bad or not p.gpu_archs:
            problems.append(f"expected arch containing {args.expect_arch!r}, got {p.gpu_archs}")

    if problems:
        print("\nGATE FAILED:")
        for x in problems:
            print(f"  - {x}")
        print("\nThis is a failure, not a skip: the run was asked for hardware it did not get, "
              "and a green skip would misreport that as nothing to see.")
        return 1

    print("\ngate ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
