#!/usr/bin/env python3
"""Probe for unsloth-zoo PR 2765 (block swap disk tier) on gfx1151, one state. Observes only; JSON via --out.
Builds a zoo venv on the stock gfx1151 torch, records which unsloth_zoo each state imports and the HIP device,
then runs pytest per file of --tests in that state's checkout (files absent at a state are recorded as such)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zoo_venv  # noqa: E402

PIN = ("torch==2.11.0+rocm7.13.0", "amd")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--head-checkout", required = True)
    ap.add_argument("--venv", type = Path, required = True)
    ap.add_argument("--tests", nargs = "+", required = True)
    ap.add_argument("--timeout", type = int, default = 3600)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent / f"{args.state}_artifacts"
    out_dir.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "checkout": args.checkout, "tests": args.tests}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout, capture_output = True,
                                     text = True).stdout.strip()
    obs["has_disk_module"] = (Path(args.checkout) / "unsloth_zoo" / "block_swap_disk.py").is_file()

    def dump():
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    obs["venv"] = zoo_venv.build_venv(args.venv.resolve(), args.head_checkout, PIN[0], PIN[1], None, out_dir)
    dump()
    if obs["venv"].get("error"):
        return 0
    python = obs["venv"]["python"]
    imp = zoo_venv.zoo_import(python, args.checkout)
    obs["unsloth_zoo_file"], obs["unsloth_zoo_import_error"] = imp["file"], imp["error"]
    dump()
    obs["pytest"] = zoo_venv.run_pytest_per_file(python, args.checkout, args.tests, out_dir, args.state, args.timeout)
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
