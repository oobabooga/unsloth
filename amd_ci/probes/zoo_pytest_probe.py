#!/usr/bin/env python3
"""Probe: run an unsloth-zoo pytest selection IN the state's zoo checkout, with that checkout first on PYTHONPATH.

Observes only (criteria/zoo_gpu_tests.py judges). Beyond pytest_probe.py it records, from inside the pytest process
(probes/zoo_pytest_plugin.py): torch.cuda.is_available(), torch.version.hip, the device arch, per-test outcomes by id,
which tests launched the fused_linear Triton kernel (CALLS["triton"] delta), and the unsloth_zoo file actually
imported, so the criteria can prove the GPU path ran against THIS state's zoo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 2400, help = "per test (pytest-timeout)")
    ap.add_argument("--wall", type = int, default = 5400, help = "whole selection")
    ap.add_argument("--only-states", default = "base,head", help = "other states (merge) record a skip (time)")
    ap.add_argument("--tests", nargs = "+", required = True)
    args = ap.parse_args()

    co = Path(args.checkout)
    obs: dict = {"state": args.state, "checkout": str(co), "tests": args.tests}
    obs["zoo_sha"] = subprocess.run(["git", "-C", str(co), "rev-parse", "HEAD"], capture_output = True,
                                    text = True).stdout.strip()
    if args.state not in args.only_states.split(","):
        obs["skipped"] = f"state {args.state} not in --only-states {args.only_states}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    present = [t for t in args.tests if (co / t.split("::")[0]).exists()]
    obs["absent_at_this_state"] = [t for t in args.tests if t not in present]
    obs["selected"] = present
    if not present:
        obs["note"] = "no selected tests exist at this state"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    plug_out = args.out.with_name(args.out.stem + "_plugin.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(co), str(HERE)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["ZOO_PLUGIN_OUT"] = str(plug_out)
    env.setdefault("UNSLOTH_IS_PRESENT", "1")
    have_timeout = subprocess.run([args.python, "-c", "import pytest_timeout"], capture_output = True,
                                  env = env).returncode == 0
    obs["pytest_timeout_plugin"] = have_timeout
    cmd = [args.python, "-m", "pytest", "-q", "-rfE", "-p", "zoo_pytest_plugin", "-p", "no:cacheprovider"]
    if have_timeout:
        cmd += ["--timeout", str(args.timeout)]
    cmd += present
    obs["cmd"] = " ".join(cmd)
    try:
        p = subprocess.run(cmd, cwd = co, env = env, capture_output = True, text = True, timeout = args.wall)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc, err = -1, "TimeoutExpired"
        out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
    obs["rc"] = rc
    obs["tail"] = (out or "")[-6000:]
    obs["stderr_tail"] = (err or "")[-3000:]
    (args.out.with_name(args.out.stem + "_pytest.log")).write_text((out or "") + "\n--- stderr ---\n" + (err or ""),
                                                                    encoding = "utf-8")
    plug = {}
    if plug_out.is_file():
        try:
            plug = json.loads(plug_out.read_text(encoding = "utf-8"))
        except Exception as e:  # noqa: BLE001
            obs["plugin_parse_error"] = f"{type(e).__name__}: {e}"
    else:
        obs["plugin_missing"] = True
    outcomes = plug.get("outcomes", {})
    obs["torch"] = plug.get("torch", {})
    obs["unsloth_zoo_file"] = plug.get("unsloth_zoo_file")
    obs["calls"] = plug.get("calls")
    obs["triton_broken"] = plug.get("triton_broken")
    obs["triton_tests"] = plug.get("triton_tests", [])
    obs["failed"] = sorted(k for k, v in outcomes.items() if v == "failed")
    obs["errors"] = sorted(k for k, v in outcomes.items() if v == "error")
    obs["n_passed"] = sum(v == "passed" for v in outcomes.values())
    obs["n_failed"] = len(obs["failed"])
    obs["n_errors"] = len(obs["errors"])
    obs["n_skipped"] = sum(v == "skipped" for v in outcomes.values())
    # GPU-parametrised tests that ran and passed (ids carry [cuda...] / -cuda- from the DEVICES parametrisation).
    obs["cuda_passed"] = sorted(k for k, v in outcomes.items() if v == "passed" and "cuda" in k.split("[", 1)[-1])
    obs["cuda_skipped"] = sorted(k for k, v in outcomes.items() if v == "skipped" and "cuda" in k.split("[", 1)[-1])
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
