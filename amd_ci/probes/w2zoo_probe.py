#!/usr/bin/env python3
"""W2 zoo probe (Qwen3.8-Flash-Next fixes: QSA eager dispatch under compile, block-swap release timing, fla float32
IEEE dots, FlashQLA dispatch default, QSA filled-slot guard, GDN short-conv seq_idx, bit-packed flex membership
table) at one unsloth-zoo state on gfx1151 ROCm. Observes only; JSON via --out.

0. One venv shared by both states (zoo_venv.build_venv): ROCm torch from AMD's gfx1151 index, the head checkout's
   deps, transformers --transformers (5.22.1: the qwen4_exp model the tests import).
1. pytest, one process per file, of --tests as they exist at this state (head-only tests are absent at the base).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zoo_venv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--head-checkout", required = True)
    ap.add_argument("--venv", type = Path, required = True)
    ap.add_argument("--torch", default = "torch==2.11.0+rocm7.13.0")
    ap.add_argument("--transformers", default = "5.22.1")
    ap.add_argument("--tests", nargs = "+", required = True)
    ap.add_argument("--timeout", type = int, default = 3600)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent / f"{args.state}_artifacts"
    out_dir.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "checkout": args.checkout, "tests": args.tests}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout, capture_output = True,
                                     text = True).stdout.strip()

    def dump():
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    obs["venv"] = zoo_venv.build_venv(args.venv.resolve(), args.head_checkout, args.torch, "amd", args.transformers, out_dir)
    dump()
    if obs["venv"].get("error"):
        return 0
    python = obs["venv"]["python"]
    obs["zoo_import"] = zoo_venv.zoo_import(python, args.checkout)
    dump()
    obs["pytest"] = zoo_venv.run_pytest_per_file(python, args.checkout, args.tests, out_dir, args.state, args.timeout)
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
