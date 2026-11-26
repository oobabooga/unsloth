#!/usr/bin/env python3
"""moe6 probe (zoo MoE LoRA fold into the base grouped GEMM + GC keep-base replay) at one unsloth-zoo state on
gfx1151 ROCm. Observes only; JSON via --out, in the zoo_moe_probe schema ({"train": {...}, "pytest": {...}}) so
criteria/zoo_moe_train_rocm.py reads it.

0. One venv shared by both states (zoo_venv.build_venv): ROCm torch from AMD's gfx1151 index (generic ROCm wheels
   segfault in _grouped_mm on gfx1151), the head checkout's deps, transformers --transformers, then TRL + PEFT and
   unsloth at --unsloth-ref (git, --no-deps), bitsandbytes for the 4-bit cases.
1. moe6_train.py per case in its own process (state checkout first on PYTHONPATH). ":eager" cases set
   UNSLOTH_COMPILE_DISABLE=1, where the keep-base replay engages; counters record fold / keep engagement.
2. pytest, one process per file (monkeypatched torch._grouped_mm must not leak across files).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zoo_venv  # noqa: E402


def _pip(py: str):
    uv = shutil.which("uv")
    return ([uv, "pip", "install", "--python", py, "--index-strategy", "unsafe-best-match"] if uv
            else [py, "-m", "pip", "install", "-q"])


def run_case(py: str, checkout: str, case: str, out_dir: Path, state: str, timeout: int) -> dict:
    tag = case.replace(":", "_")
    out, log = out_dir / f"train_{state}_{tag}.json", out_dir / f"train_{state}_{tag}.log"
    rc = zoo_venv.sh([py, "-u", str(HERE / "moe6_train.py"), "--case", case, "--out", str(out)], log,
                     timeout = timeout, env = zoo_venv.state_env(checkout), cwd = checkout)
    rec: dict = {"rc": rc}
    if out.is_file():
        try:
            rec.update(json.loads(out.read_text(encoding = "utf-8")))
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = repr(e)
    else:
        rec["missing_output"] = True
    rec["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:] if log.is_file() else ""
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--head-checkout", required = True)
    ap.add_argument("--venv", type = Path, required = True)
    ap.add_argument("--torch-pin", default = "torch==2.11.0+rocm7.13.0")
    ap.add_argument("--torch-index", default = "amd")
    ap.add_argument("--transformers", default = "5.17.0")
    ap.add_argument("--trl-peft", default = "trl==1.18.0,peft==0.21.3")
    ap.add_argument("--unsloth-ref", required = True)
    ap.add_argument("--cases", default = "qwen3moe:16bit,qwen3moe:4bit,qwen3moe:16bit:eager,qwen3moe:4bit:eager,"
                                         "gptoss:16bit,gptoss:16bit:eager")
    ap.add_argument("--tests", nargs = "*", default = [])
    ap.add_argument("--timeout", type = int, default = 1500)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent / f"{args.state}_artifacts"
    out_dir.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "checkout": args.checkout}
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = args.checkout, capture_output = True,
                                     text = True).stdout.strip()

    def dump():
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    venv = args.venv.resolve()
    obs["venv"] = zoo_venv.build_venv(venv, args.head_checkout, args.torch_pin, args.torch_index, args.transformers,
                                      out_dir)
    dump()
    if obs["venv"].get("error"):
        return 0
    py = obs["venv"]["python"]
    log = out_dir / "moe6_extra_installs.log"
    marker = venv / ".moe6_extra_ready"
    extra: dict = {}
    if not marker.is_file():
        extra["trl_peft"] = zoo_venv.sh(_pip(py) + [*args.trl_peft.split(","), f"transformers=={args.transformers}"],
                                        log, timeout = 1800)
        extra["unsloth"] = zoo_venv.sh(_pip(py) + ["--no-deps", "--reinstall",
                                                   f"unsloth @ git+https://github.com/unslothai/unsloth@{args.unsloth_ref}"],
                                       log, timeout = 1800)
        extra["bitsandbytes"] = zoo_venv.sh(_pip(py) + ["bitsandbytes"], log, timeout = 1800)
        if extra.get("unsloth") == 0 and extra.get("trl_peft") == 0:
            marker.write_text("ok", encoding = "utf-8")
    else:
        extra["reused"] = True
    obs["extra_installs"] = extra
    obs["extra_log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-2500:] if log.is_file() else ""
    imp = zoo_venv.zoo_import(py, args.checkout)
    obs["unsloth_zoo_file"], obs["unsloth_zoo_import_error"] = imp["file"], imp["error"]
    dump()
    obs["train"] = {}
    for case in args.cases.split(","):
        obs["train"][case] = run_case(py, args.checkout, case, out_dir, args.state, args.timeout)
        dump()
    if args.tests:
        obs["pytest"] = zoo_venv.run_pytest_per_file(py, args.checkout, args.tests, out_dir, args.state,
                                                     args.timeout)
        obs["pytest"]["selected"] = [t for t in args.tests if (Path(args.checkout) / t).exists()]
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
