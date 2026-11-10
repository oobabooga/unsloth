#!/usr/bin/env python3
"""Probe for unsloth-zoo PR 2241 (Inductor's rand4x random path off on torch 2.13 / 2.14) on gfx1151, one state.
Observes only; JSON via --out.

--lane gate   stock gfx1151 torch (2.11, AMD index). The 4x path does not exist there, so the patch must NOT
              install. Records the import state (zoo2241_cell.py --mode state) and pytest of --tests per file.
--lane t214   torch 2.14.1 ROCm (download.pytorch.org/whl/rocm7.14), where the 4x path exists. Records the
              import state, the dropout statistics after importing the state's unsloth_zoo (--mode dropout), and,
              where the state has the patch module, rand4x_proof.py run against it (exit 0 pass / 1 fail /
              2 no GPU / 3 void), plus pytest of --tests.
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

LANES = {
    "gate": ("torch==2.11.0+rocm7.13.0", "amd"),
    "t214": ("torch==2.14.1+rocm7.14", "https://download.pytorch.org/whl/rocm7.14"),
}
PATCH = "unsloth_zoo/temporary_patches/inductor_rand4x.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--head-checkout", required = True)
    ap.add_argument("--venv", type = Path, required = True)
    ap.add_argument("--lane", choices = sorted(LANES), required = True)
    ap.add_argument("--tests", nargs = "+", required = True)
    ap.add_argument("--timeout", type = int, default = 3600)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent / f"{args.state}_artifacts"
    out_dir.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "checkout": args.checkout, "lane": args.lane, "tests": args.tests}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout, capture_output = True,
                                     text = True).stdout.strip()
    obs["has_patch_module"] = (Path(args.checkout) / PATCH).is_file()

    def dump():
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    pin, index = LANES[args.lane]
    obs["venv"] = zoo_venv.build_venv(args.venv.resolve(), args.head_checkout, pin, index, None, out_dir)
    dump()
    if obs["venv"].get("error"):
        return 0
    python = obs["venv"]["python"]
    imp = zoo_venv.zoo_import(python, args.checkout)
    obs["unsloth_zoo_file"], obs["unsloth_zoo_import_error"] = imp["file"], imp["error"]
    obs["import_state"] = zoo_venv.run_cell(python, HERE / "zoo2241_cell.py", ["--mode", "state"], args.checkout,
                                            out_dir, "state", timeout = 900)
    dump()
    if args.lane == "t214":
        obs["dropout"] = zoo_venv.run_cell(python, HERE / "zoo2241_cell.py", ["--mode", "dropout"], args.checkout,
                                           out_dir, "dropout", timeout = 1800,
                                           extra_env = {"TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1"})
        dump()
        if obs["has_patch_module"]:
            log = out_dir / "rand4x_proof.log"
            rc = zoo_venv.sh([python, str(HERE / "rand4x_proof.py"), "--patch-file", str(Path(args.checkout) / PATCH)],
                             log, timeout = 3600, env = zoo_venv.state_env(args.checkout), cwd = args.checkout)
            obs["proof"] = {"rc": rc, "tail": log.read_text(encoding = "utf-8", errors = "replace")[-6000:]}
            dump()
    obs["pytest"] = zoo_venv.run_pytest_per_file(python, args.checkout, args.tests, out_dir, args.state, args.timeout)
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
