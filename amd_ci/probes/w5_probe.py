#!/usr/bin/env python3
"""W5 probe (fused CE loss, GA normalisation, offload_layers block swap, gpt-oss MXFP4 export) at one unsloth-zoo
state on gfx1151 ROCm. Observes only; JSON via --out.

0. One venv shared by both states (zoo_venv.build_venv): ROCm torch from AMD's gfx1151 index, the head checkout's
   deps, transformers 5.23.0, then TRL 1.18.0 + PEFT 0.21.3 and unsloth at --unsloth-ref (git, --no-deps).
   Corner A is an overlay dir on PYTHONPATH: transformers 4.57.6, TRL 0.22.2, PEFT 0.18.0 (+ the hub /
   tokenizers / datasets they need), installed --no-deps once.
1. w5_run_all.py per corner (B = venv as built, A = overlay first): w5_cells.py per tiny model in its own process,
   the state's checkout first on PYTHONPATH so its unsloth_zoo shadows the installed one.
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

OVERLAY_A = ["transformers==4.57.6", "trl==0.22.2", "peft==0.18.0", "huggingface_hub==0.36.2", "tokenizers==0.22.2",
             "datasets==3.6.0"]


def _pip(py: str):
    import shutil
    uv = shutil.which("uv")
    return ([uv, "pip", "install", "--python", py, "--index-strategy", "unsafe-best-match"] if uv
            else [py, "-m", "pip", "install", "-q"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--head-checkout", required = True)
    ap.add_argument("--venv", type = Path, required = True)
    ap.add_argument("--torch-pin", default = "torch==2.11.0+rocm7.13.0")
    ap.add_argument("--torch-index", default = "amd")
    ap.add_argument("--unsloth-ref", required = True)
    ap.add_argument("--corners", default = "B,A")
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
    obs["venv"] = zoo_venv.build_venv(venv, args.head_checkout, args.torch_pin, args.torch_index, "5.23.0", out_dir)
    dump()
    if obs["venv"].get("error"):
        return 0
    py = obs["venv"]["python"]
    log = out_dir / "w5_extra_installs.log"
    marker = venv / ".w5_extra_ready"
    overlay = venv.parent / "w5_overlay_A"
    extra: dict = {}
    if not marker.is_file():
        extra["trl_peft"] = zoo_venv.sh(_pip(py) + ["trl==1.18.0", "peft==0.21.3", "datasets>=4.7"], log, timeout = 1800)
        extra["unsloth"] = zoo_venv.sh(_pip(py) + ["--no-deps", "--reinstall",
                                                   f"unsloth @ git+https://github.com/unslothai/unsloth@{args.unsloth_ref}"],
                                       log, timeout = 1800)
        # NF4 for the offload_qlora cells; a failure here is recorded and those cells report it.
        extra["bitsandbytes"] = zoo_venv.sh(_pip(py) + ["bitsandbytes"], log, timeout = 1800)
        extra["overlay_A"] = zoo_venv.sh(_pip(py) + ["--no-deps", "--target", str(overlay), *OVERLAY_A], log,
                                         timeout = 1800)
        if extra.get("unsloth") == 0 and extra.get("trl_peft") == 0:
            marker.write_text("ok", encoding = "utf-8")
    else:
        extra["reused"] = True
    obs["extra_installs"] = extra
    obs["extra_log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-2500:] if log.is_file() else ""
    imp = zoo_venv.zoo_import(py, args.checkout)
    obs["unsloth_zoo_file"], obs["unsloth_zoo_import_error"] = imp["file"], imp["error"]
    dump()
    obs["corners"] = {}
    for corner in args.corners.split(","):
        cdir = out_dir / f"corner_{corner}"
        # Corner A: the overlay sits after the checkout and ahead of site-packages.
        pp = args.checkout + ((":" + str(overlay)) if corner == "A" else "")
        log_c = out_dir / f"corner_{corner}.log"
        rc = zoo_venv.sh([py, "-u", str(HERE / "w5_run_all.py"), "--python", py, "--out-dir", str(cdir),
                          "--pythonpath", pp, "--timeout", str(args.timeout)], log_c, timeout = 6 * args.timeout,
                         env = zoo_venv.state_env(args.checkout), cwd = args.checkout)
        summ = cdir / "summary.json"
        obs["corners"][corner] = {"rc": rc, "summary": json.loads(summ.read_text(encoding = "utf-8")) if summ.is_file()
                                  else None, "log_tail": log_c.read_text(encoding = "utf-8", errors = "replace")[-2500:]}
        dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
