#!/usr/bin/env python3
"""Probe (T7h.5e): unsloth #16071 fused LoRA order on gfx1151. Observes only (criteria/t7h5e_lora.py judges).

Per state, with the state's unsloth checkout first on PYTHONPATH, each arm a fresh process of
probes/t7h5e_lora_scenario.py (real LoRA SFT + greedy generate), no instrumentation in the timed / loss arms:
  - every state: bf16 run per --configs entry with UNSLOTH_LORA_FUSED_PASSES=1 (key '<model>|flag1');
  - base: an fp32 reference run of each 16-bit model ('|fp32');
  - head: the kill switch UNSLOTH_LORA_FUSED_PASSES=0 ('|flag0'), then short count arms ('|count1' / '|count0',
    T7H5E_COUNT=1 wraps _lora_fused_ok in a counter; excluded from timing / loss), and
    tests/test_lora_fused_passes.py + tests/test_lora_first_addmm_op.py (head-only files) via pytest;
    'prof1' / 'prof0' arms profile one extra step for device-to-device copy counts.
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
    ap.add_argument("--prev-sha", default = os.environ.get("T7H5E_PREV_SHA", "dc64c6f6e"),
                    help = "head only: an earlier PR commit timed in the same run (16-bit ABBA with the head's flag1)")
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
    arms = [("flag1", {"UNSLOTH_LORA_FUSED_PASSES": "1"})]
    if args.state == "base":
        arms.append(("fp32", {"UNSLOTH_LORA_FUSED_PASSES": "1", "T7H5E_DTYPE": "float32"}))
    prev = None
    if args.state == "head" and args.prev_sha:
        prev = co.parent / f"prev_{args.prev_sha}"
        if not prev.is_dir():
            p = subprocess.run(["git", "-C", str(co), "worktree", "add", "-q", "--detach", str(prev), args.prev_sha],
                               capture_output = True, encoding = "utf-8", errors = "replace")
            obs["prev_worktree"] = {"rc": p.returncode, "err": p.stderr[-500:]}
        obs["prev_sha"] = subprocess.run(["git", "-C", str(prev), "rev-parse", "HEAD"], capture_output = True,
                                         encoding = "utf-8", errors = "replace").stdout.strip()
        obs["prev_checkout"] = str(prev)
    if args.state == "head":
        if prev is not None:
            # ABBA against the earlier commit: flag1 (above), prev1, prev1b, flag1b.
            arms += [("prev1", {"UNSLOTH_LORA_FUSED_PASSES": "1", "_CO": str(prev)}),
                     ("prev1b", {"UNSLOTH_LORA_FUSED_PASSES": "1", "_CO": str(prev)}),
                     ("flag1b", {"UNSLOTH_LORA_FUSED_PASSES": "1"})]
        arms += [("flag0", {"UNSLOTH_LORA_FUSED_PASSES": "0"}),
                 ("count1", {"UNSLOTH_LORA_FUSED_PASSES": "1", "T7H5E_COUNT": "1", "T7H5E_STEPS": "6"}),
                 ("count0", {"UNSLOTH_LORA_FUSED_PASSES": "0", "T7H5E_COUNT": "1", "T7H5E_STEPS": "6"}),
                 ("prof1", {"UNSLOTH_LORA_FUSED_PASSES": "1", "T7H5E_PROFILE": "1", "T7H5E_STEPS": "4"}),
                 ("prof0", {"UNSLOTH_LORA_FUSED_PASSES": "0", "T7H5E_PROFILE": "1", "T7H5E_STEPS": "4"})]
    for cfg in args.configs.split(","):
        model, q4 = cfg.rsplit(":", 1)
        for arm, extra in arms:
            if arm == "fp32" and q4 == "1":
                continue
            if arm in ("prev1b", "flag1b") and q4 == "1":
                continue
            extra = dict(extra)
            arm_co = extra.pop("_CO", None)
            key = f"{model.split('/')[-1]}|{arm}"
            res = args.out.with_name(f"{args.out.stem}_{len(obs['runs'])}.json")
            e = dict(env, **extra,
                     **({"PYTHONPATH": os.pathsep.join([arm_co] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))}
                        if arm_co else {}),
                     UNSLOTH_COMPILE_LOCATION = str(args.out.parent / f"cache_{args.state}_{len(obs['runs'])}"))
            rc, out, errt = _run([args.python, "-u", str(HERE / "t7h5e_lora_scenario.py"), str(res), model, q4], e,
                                 str(args.out.parent), args.wall)
            Path(str(res) + ".log").write_text(out + "\n--- stderr ---\n" + errt, encoding = "utf-8")
            r = {"rc": rc, "stderr_tail": errt[-2000:], "checkout": arm_co or str(co)}
            if res.is_file():
                try:
                    r["scenario"] = json.loads(res.read_text(encoding = "utf-8"))
                except Exception as ex:  # noqa: BLE001
                    r["parse_error"] = f"{type(ex).__name__}: {ex}"
            obs["runs"][key] = r
            args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    tests = [t for t in ("tests/test_lora_fused_passes.py", "tests/test_lora_first_addmm_op.py") if (co / t).is_file()]
    if args.state == "head" and tests:
        rc, out, errt = _run([args.python, "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "-q", "-rfE",
                              *tests], env, str(co), 1800)
        full = out + errt
        obs["head_only_pytest"] = {"rc": rc, "files": tests, "tail": full[-3000:],
                                   "failed_ids": sorted({ln.split(" ")[1] for ln in full.splitlines()
                                                         if ln.startswith(("FAILED ", "ERROR ")) and len(ln.split(" ")) > 1})}
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
