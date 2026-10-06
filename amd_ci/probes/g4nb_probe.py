#!/usr/bin/env python3
"""Probe: run the AMD Gemma-4 26B-A4B Text + Vision notebooks (instrumented scripts) at one arm.

Observes only. The arms are package states of ONE venv (the notebook's own install cell ran
in the Environment step), probed in order base then head:
  base  the notebook's installs: PyPI unsloth 2026.9.14 / unsloth_zoo 2026.9.9
  head  additionally `--no-deps --force-reinstall` unsloth @ $HEAD_UNSLOTH_SHA and
        unsloth_zoo @ $HEAD_ZOO_SHA from git
Each notebook runs in its own process and working dir with a fresh Unsloth compile cache.
The instrumented script writes its own JSON (cells reached, per-generate timing and routed
census, step times, versions); this probe adds rc, log tail and the failing cell.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parent / "nb"
TAGS = ("g4_Text", "g4_Vision")


def cell_source(tag: str, k) -> str:
    try:
        lines = (NB / f"{tag}.py").read_text(encoding = "utf-8").splitlines()
        i = lines.index(f"_amd_cell({k})")
        return lines[i + 1][:160]
    except Exception:  # noqa: BLE001
        return "?"


def pip_versions(python: str) -> dict:
    p = subprocess.run(["uv", "pip", "list", "--python", python, "--format", "json"], capture_output = True, text = True)
    try:
        return {d["name"].lower(): d["version"] for d in json.loads(p.stdout)}
    except Exception:  # noqa: BLE001
        return {"_error": (p.stderr or "")[-500:]}


def head_install(python: str, out_dir: Path) -> dict:
    us, zs = os.environ["HEAD_UNSLOTH_SHA"], os.environ["HEAD_ZOO_SHA"]
    cmd = ["uv", "pip", "install", "--python", python, "--no-deps", "--force-reinstall", "--no-cache",
           f"unsloth @ git+https://github.com/unslothai/unsloth@{us}",
           f"unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@{zs}"]
    p = subprocess.run(cmd, capture_output = True, text = True)
    (out_dir / "head_install.log").write_text((p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""), encoding = "utf-8")
    return {"cmd": " ".join(cmd), "rc": p.returncode, "tail": (p.stderr or "")[-1500:]}


def fix_install(python: str, out_dir: Path) -> dict:
    zs = os.environ["FIX_ZOO_SHA"]
    cmd = ["uv", "pip", "install", "--python", python, "--no-deps", "--force-reinstall", "--no-cache",
           f"unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@{zs}"]
    p = subprocess.run(cmd, capture_output = True, text = True)
    (out_dir / "after_fix_install.log").write_text((p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""), encoding = "utf-8")
    return {"cmd": " ".join(cmd), "rc": p.returncode, "tail": (p.stderr or "")[-1500:]}


def run_nb(tag: str, state: str, python: str, root: Path, out_dir: Path, timeout: int) -> dict:
    wd = root / "runs" / f"{state}_{tag}"
    wd.mkdir(parents = True, exist_ok = True)
    obs_path = out_dir / f"nb_{state}_{tag}.json"
    log = out_dir / f"nb_{state}_{tag}.log"
    env = dict(os.environ)
    env.update(AMD_NB_OUT = str(obs_path), UNSLOTH_COMPILE_LOCATION = str(wd / "unsloth_compiled_cache"),
               PYTHONUNBUFFERED = "1")
    # GitHub Actions sets CI=true, and torch defaults functorch error_on_custom_op_aliasing to
    # bool(os.getenv("CI")): the first-call aliasing analyzer then RAISES on inductor's own
    # inductor::_alloc_from_pool (runs 37408657305 / 37410230820, both arms, first generate) where a
    # user's machine only warns. Run the notebooks as a user would.
    env.pop("CI", None)
    env["TORCHINDUCTOR_ERROR_ON_CUSTOM_OP_ALIASING"] = "0"
    for k in ("UNSLOTH_MOE_ROUTED_KERNEL", "UNSLOTH_MOE_ROUTED_FUSED", "UNSLOTH_MOE_BACKEND", "UNSLOTH_COMPILE_DISABLE"):
        env.pop(k, None)
    t0 = time.time()
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run([python, str(NB / f"{tag}.py")], cwd = wd, env = env, stdout = fh,
                                stderr = subprocess.STDOUT, timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rec: dict = {"rc": rc, "wall_s": round(time.time() - t0, 1)}
    if obs_path.is_file():
        try:
            rec.update(json.loads(obs_path.read_text(encoding = "utf-8")))
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = repr(e)
    else:
        rec["missing_output"] = True
    text = log.read_text(encoding = "utf-8", errors = "replace")
    rec["log_tail"] = text[-4000:]
    i = text.rfind("Traceback (most recent call last)")
    rec["traceback"] = text[i:][-4000:] if i >= 0 else None
    rec["passed"] = rc == 0 and rec.get("completed") is True
    if not rec["passed"]:
        rec["failing_cell"] = rec.get("last_cell")
        rec["failing_cell_src"] = cell_source(tag, rec.get("last_cell"))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 6000)
    args = ap.parse_args()
    args.out = args.out.resolve()
    out_dir = args.out.parent
    root = Path(os.environ["AMD_CI_WORK"])
    obs: dict = {"state": args.state}
    if args.state == "head":
        obs["head_install"] = head_install(args.python, out_dir)
    elif args.state == "after_fix":
        # Probed after head: the venv already holds unsloth @ HEAD_UNSLOTH_SHA; swap only the zoo.
        obs["head_install"] = fix_install(args.python, out_dir)
    elif args.state != "base":
        obs["error"] = f"unknown state {args.state}"
    obs["pip"] = {k: v for k, v in pip_versions(args.python).items()
                  if k in ("torch", "torchvision", "triton", "triton-rocm", "pytorch-triton-rocm", "bitsandbytes", "transformers",
                           "trl", "peft", "unsloth", "unsloth-zoo", "unsloth_zoo", "accelerate", "datasets", "tokenizers")}
    obs["notebooks"] = {}
    for tag in TAGS:
        obs["notebooks"][tag] = run_nb(tag, args.state, args.python, root, out_dir, args.timeout)
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
