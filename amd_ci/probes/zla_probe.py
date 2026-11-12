#!/usr/bin/env python3
"""Probe for the zoo ROCm gfx11 float32 gated-delta fix, one state (an unsloth-zoo checkout). Observes only.

Cells (zla_cell.py, one process each; a HIP illegal access kills the process, a hang hits the timeout):
  op_fp32 / op_bf16      fla chunk_gated_delta_rule fwd + bwd at the tiny Qwen3.5-MoE shape vs an fp64 recurrence
  model_fp32             tiny Qwen3.5-MoE forward + backward (tests/test_multi_lora_experts.py's model, fp32)
  model_fp32_torchpath   the same with fla never injected (transformers' pure-torch reference)
  model_bf16             the same in bfloat16 (fla kernels at both states)
  sft_fp32 / sft_bf16    FastModel LoRA SFT, 5 steps
  gen_fp32 / gen_bf16    greedy generate, 8 tokens (fla fused_recurrent decode)
Then tests/test_multi_lora_experts.py from the unsloth checkout, one process per test.
The state's checkout is first on PYTHONPATH so its unsloth_zoo shadows the installed one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELL = HERE / "zla_cell.py"
SHAPE = "3:6:32:128:128"
CELLS = {
    "op_fp32": (f"op:float32:{SHAPE}", {}),
    "op_bf16": (f"op:bfloat16:{SHAPE}", {}),
    "model_fp32": ("model:float32", {}),
    "model_fp32_torchpath": ("model:float32", {"UNSLOTH_DISABLE_VENDORED_FLA": "1"}),
    "model_bf16": ("model:bfloat16", {}),
    "sft_fp32": ("sft:float32", {}),
    "sft_bf16": ("sft:bfloat16", {}),
    "gen_fp32": ("gen:float32", {}),
    "gen_bf16": ("gen:bfloat16", {}),
}
TEST_FILE = "tests/test_multi_lora_experts.py"
CLEAR = ("UNSLOTH_DISABLE_VENDORED_FLA", "UNSLOTH_FORCE_VENDORED_FLA", "UNSLOTH_COMPILE_DISABLE", "AMDGCN_USE_BUFFER_OPS",
         "AMD_SERIALIZE_KERNEL", "ZLA_PIN", "ZLA_UPSTREAM_FLA", "PYTEST_ADDOPTS", "UNSLOTH_FORCE_FLOAT32")


def _env(checkout, extra, work):
    env = dict(os.environ)
    env["PYTHONPATH"] = checkout + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in CLEAR:
        env.pop(k, None)
    env["UNSLOTH_IS_PRESENT"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["UNSLOTH_COMPILE_LOCATION"] = str(work / "compiled")
    env.update(extra)
    return env


def _crashed(log_text, rc):
    return ("illegal memory access" in log_text) or ("hipErrorIllegalAddress" in log_text) or rc not in (0,)


def run_cell(name, spec, extra, checkout, out_dir, state, python, timeout):
    work = out_dir / f"{state}_{name}"
    work.mkdir(parents = True, exist_ok = True)
    cmd = [python, str(CELL), "--cell", spec, "--out", str(work / "cell.json"),
           "--launch-log", str(work / "launches.jsonl"), "--tensor-out", str(work / "tensors.pt")]
    t0 = time.time()
    with open(work / "log.txt", "wb") as fh:
        try:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = _env(checkout, extra, work),
                                timeout = timeout, cwd = str(work)).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rec = {"rc": rc, "seconds": round(time.time() - t0, 1), "spec": spec, "env": extra}
    try:
        rec.update(json.loads((work / "cell.json").read_text(encoding = "utf-8")))
    except Exception as e:  # noqa: BLE001
        rec["json_err"] = repr(e)
    log = (work / "log.txt").read_text(encoding = "utf-8", errors = "replace")
    rec["illegal_access"] = ("illegal memory access" in log) or ("hipErrorIllegalAddress" in log)
    rec["crashed"] = _crashed(log, rc) or rec.get("stage") != "done"
    rec["log_tail"] = log[-1500:]
    try:
        lines = (work / "launches.jsonl").read_text(encoding = "utf-8").splitlines()
        rec["last_launch"] = next((json.loads(x) for x in reversed(lines) if '"k"' in x), None)
        names = {json.loads(x).get("k") for x in lines if '"k"' in x}
        rec["fla_kernels_launched"] = sorted(n for n in names if n and ("gated_delta" in n or "wy_repr" in n
                                                                         or n.startswith("chunk_")))
    except Exception:  # noqa: BLE001
        pass
    rec["tensors"] = str(work / "tensors.pt") if (work / "tensors.pt").is_file() else None
    return rec


def compare_tensors(a_path, b_path):
    import torch
    a, b = torch.load(a_path), torch.load(b_path)
    out = {"logits_max_abs": float((a["logits"] - b["logits"]).abs().max()),
           "logits_ref_absmax": float(b["logits"].abs().max())}
    common = sorted(set(a["grads"]) & set(b["grads"]))
    out["n_grads"] = len(common)
    out["grads_max_abs"] = max((float((a["grads"][n] - b["grads"][n]).abs().max()) for n in common), default = None)
    out["grads_ref_absmax"] = max((float(b["grads"][n].abs().max()) for n in common), default = None)
    return out


def _junit(path):
    res = {"passed": [], "failed": [], "errors": [], "skipped": []}
    try:
        root = ET.parse(path).getroot()
    except Exception as e:  # noqa: BLE001
        res["parse_error"] = repr(e)
        return res
    for tc in root.iter("testcase"):
        tid = f"{tc.get('classname')}::{tc.get('name')}"
        kinds = [c.tag for c in tc]
        key = "failed" if "failure" in kinds else "errors" if "error" in kinds else "skipped" if "skipped" in kinds else "passed"
        res[key].append(tid)
    return res


def run_tests(unsloth_src, checkout, out_dir, state, python):
    work = out_dir / f"{state}_pytest"
    work.mkdir(parents = True, exist_ok = True)
    env = _env(checkout, {}, work)
    col = subprocess.run([python, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", TEST_FILE],
                         cwd = unsloth_src, env = env, capture_output = True, text = True, timeout = 900)
    ids = [ln.strip() for ln in col.stdout.splitlines() if ln.startswith(TEST_FILE + "::")]
    res = {"collected": ids, "collect_rc": col.returncode, "collect_tail": (col.stdout + col.stderr)[-1500:], "tests": {}}
    for tid in ids:
        tag = tid.split("::", 1)[1].replace("[", "_").replace("]", "").replace("/", "_")
        junit = work / f"junit_{tag}.xml"
        log = work / f"pytest_{tag}.log"
        cmd = [python, "-m", "pytest", "-q", "-rfEs", "--tb=short", "-p", "no:cacheprovider", f"--junitxml={junit}", tid]
        with open(log, "wb") as fh:
            try:
                rc = subprocess.run(cmd, cwd = unsloth_src, env = env, stdout = fh, stderr = subprocess.STDOUT,
                                    timeout = 1200).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        j = _junit(junit)
        text = log.read_text(encoding = "utf-8", errors = "replace")
        outcome = ("passed" if j["passed"] and not (j["failed"] or j["errors"]) and rc == 0 else
                   "skipped" if j["skipped"] and not (j["failed"] or j["errors"] or j["passed"]) else "failed")
        res["tests"][tid] = {"rc": rc, "outcome": outcome, "illegal_access": "illegal memory access" in text,
                             "tail": text[-800:] if outcome != "passed" else ""}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--unsloth-src", required = True)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 900)
    a = ap.parse_args()
    out_dir = a.out.parent
    res: dict = {"state": a.state, "checkout": a.checkout, "cells": {}}

    def save():
        a.out.write_text(json.dumps(res, indent = 2, default = str), encoding = "utf-8")

    for name, (spec, extra) in CELLS.items():
        res["cells"][name] = run_cell(name, spec, extra, a.checkout, out_dir, a.state, a.python, a.timeout)
        c = res["cells"][name]
        print(f"[{a.state}] {name}: rc={c['rc']} crashed={c['crashed']} {c['seconds']}s steps={c.get('steps')}", flush = True)
        save()
    m, t = res["cells"].get("model_fp32", {}), res["cells"].get("model_fp32_torchpath", {})
    if m.get("tensors") and t.get("tensors"):
        try:
            res["model_fp32_vs_torchpath"] = compare_tensors(m["tensors"], t["tensors"])
        except Exception as e:  # noqa: BLE001
            res["model_fp32_vs_torchpath"] = {"error": repr(e)}
    save()
    res["pytest"] = run_tests(a.unsloth_src, a.checkout, out_dir, a.state, a.python)
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
