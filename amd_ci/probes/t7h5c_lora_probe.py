#!/usr/bin/env python3
"""Probe (T7h.5c): unsloth #16071 fused LoRA order on gfx1151. Observes only (criteria/t7h5c_lora.py judges).

Per state, with the state's unsloth checkout first on PYTHONPATH:
  - a real bf16 LoRA SFT + greedy generate for each --configs entry (probes/t7h5c_lora_scenario.py, fresh process);
  - at the head, the same again with UNSLOTH_LORA_FUSED_PASSES=0 (kill switch), and
    tests/test_lora_fused_passes.py (head-only file) via pytest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _run(cmd, env, cwd, wall):
    try:
        p = subprocess.run(cmd, cwd = cwd, env = env, capture_output = True, encoding = "utf-8", errors = "replace",
                           timeout = wall)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        return -1, out or "", "TimeoutExpired"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--wall", type = int, default = 2400)
    ap.add_argument("--configs", default = "unsloth/Llama-3.2-1B-Instruct:0,unsloth/Llama-3.2-1B-Instruct-bnb-4bit:1")
    args = ap.parse_args()
    co = Path(args.checkout)
    obs: dict = {"state": args.state, "checkout": str(co), "runs": {}}
    obs["sha"] = subprocess.run(["git", "-C", str(co), "rev-parse", "HEAD"], capture_output = True,
                                encoding = "utf-8", errors = "replace").stdout.strip()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(co)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("UNSLOTH_IS_PRESENT", "1")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    flags = ["1", "0"] if args.state == "head" else ["1"]
    for cfg in args.configs.split(","):
        model, q4 = cfg.rsplit(":", 1)
        for flag in flags:
            key = f"{model.split('/')[-1]}|flag{flag}"
            res = args.out.with_name(f"{args.out.stem}_{len(obs['runs'])}.json")
            e = dict(env, UNSLOTH_LORA_FUSED_PASSES = flag,
                     UNSLOTH_COMPILE_LOCATION = str(args.out.parent / f"cache_{args.state}_{len(obs['runs'])}"))
            rc, out, errt = _run([args.python, "-u", str(HERE / "t7h5c_lora_scenario.py"), str(res), model, q4], e,
                                 str(args.out.parent), args.wall)
            Path(str(res) + ".log").write_text(out + "\n--- stderr ---\n" + errt, encoding = "utf-8")
            r = {"rc": rc, "stderr_tail": errt[-2000:]}
            if res.is_file():
                try:
                    r["scenario"] = json.loads(res.read_text(encoding = "utf-8"))
                except Exception as ex:  # noqa: BLE001
                    r["parse_error"] = f"{type(ex).__name__}: {ex}"
            obs["runs"][key] = r
            args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    if args.state == "head" and (co / "tests" / "test_lora_fused_passes.py").is_file():
        rc, out, errt = _run([args.python, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "-q", "-rfE",
                              "tests/test_lora_fused_passes.py"], env, str(co), 1800)
        obs["head_only_pytest"] = {"rc": rc, "tail": (out + errt)[-3000:]}
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
