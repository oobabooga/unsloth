#!/usr/bin/env python3
"""Probe for unsloth-zoo PR 2240 (GLM-5.3-Flash / glm5_next sparse flex) on gfx1151 ROCm, one state. Observes only;
JSON via --out.

0. Venv (zoo_venv.build_venv, once, shared): ROCm torch from AMD's gfx1151 index, the head checkout's deps, then
   transformers --transformers (5.21.0: glm5_next only exists in recent transformers; zoo's declared ceiling is
   lower, so the patch and its tests would otherwise no-op / skip).
1. pytest of --tests, one process per file (the new glm5_next file is absent at the base: excluded by id).
2. If the state has tests/test_glm5_next_sparse.py (the head): zoo2240_cell.py, modes on / off / upstream / fp32,
   one process each, then the compare step.
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

TEST_FILE = "tests/test_glm5_next_sparse.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--head-checkout", required = True)
    ap.add_argument("--venv", type = Path, required = True)
    ap.add_argument("--torch-pin", default = "torch==2.11.0+rocm7.13.0")
    ap.add_argument("--torch-index", default = "amd")
    ap.add_argument("--transformers", default = "5.21.0")
    ap.add_argument("--tests", nargs = "+", required = True)
    ap.add_argument("--timeout", type = int, default = 3600)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent / f"{args.state}_artifacts"
    out_dir.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "checkout": args.checkout, "tests": args.tests}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout, capture_output = True,
                                     text = True).stdout.strip()
    obs["has_glm5_next_patch"] = (Path(args.checkout) / "unsloth_zoo/temporary_patches/glm5_next_sparse_flex.py").is_file()

    def dump():
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    obs["venv"] = zoo_venv.build_venv(args.venv.resolve(), args.head_checkout, args.torch_pin, args.torch_index,
                                      args.transformers, out_dir)
    dump()
    if obs["venv"].get("error"):
        return 0
    python = obs["venv"]["python"]
    imp = zoo_venv.zoo_import(python, args.checkout)
    obs["unsloth_zoo_file"], obs["unsloth_zoo_import_error"] = imp["file"], imp["error"]
    dump()
    if (Path(args.checkout) / TEST_FILE).is_file():
        cells: dict = {}
        for mode in ("on", "off", "upstream", "fp32"):
            cells[mode] = zoo_venv.run_cell(python, HERE / "zoo2240_cell.py",
                                            ["--mode", mode, "--pt", str(out_dir / f"glm5next_{mode}.pt")],
                                            args.checkout, out_dir, f"glm5next_{mode}")
            dump()
        cells["compare"] = zoo_venv.run_cell(python, HERE / "zoo2240_cell.py", ["--compare", str(out_dir)],
                                             args.checkout, out_dir, "glm5next_compare", timeout = 600)
        obs["cells"] = cells
        dump()
    obs["pytest"] = zoo_venv.run_pytest_per_file(python, args.checkout, args.tests, out_dir, args.state, args.timeout)
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
