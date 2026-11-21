#!/usr/bin/env python3
"""Probe for unsloth-zoo PR 2644 (GC keeps activations on device on unified memory) on gfx1151, one state.
Observes only; JSON via --out. Venv via zoo_venv.build_venv (ROCm torch + head deps), shared by both states.
Cells (one process each, state checkout first on PYTHONPATH): pr2644_cell.py (detector answer, reentrant
offload lease count, non-reentrant pack decision, grad sha256, peak memory), run twice; then the PR's own test file
if present at this state."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zoo_venv  # noqa: E402

TEST = "tests/test_gc_unified_memory_keep_on_device.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--head-checkout", required = True)
    ap.add_argument("--venv", type = Path, required = True)
    ap.add_argument("--torch", default = "torch==2.11.0+rocm7.13.0")
    ap.add_argument("--timeout", type = int, default = 1800)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent / f"{args.state}_artifacts"
    out_dir.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "checkout": args.checkout}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout, capture_output = True,
                                     text = True).stdout.strip()

    def dump():
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    obs["venv"] = zoo_venv.build_venv(args.venv.resolve(), args.head_checkout, args.torch, "amd", None, out_dir)
    dump()
    if obs["venv"].get("error"):
        return 0
    python = obs["venv"]["python"]
    obs["cells"] = []
    for i in range(2):
        obs["cells"].append(zoo_venv.run_cell(python, HERE / "pr2644_cell.py", [args.checkout], args.checkout, out_dir,
                                              f"{args.state}_{i}", timeout = args.timeout))
        dump()
    if (Path(args.checkout) / TEST).is_file():
        obs["pytest"] = zoo_venv.run_pytest_per_file(python, args.checkout, [TEST], out_dir, args.state, args.timeout)
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
