#!/usr/bin/env python3
"""Probe for the Gemma 3 float32 pair: unsloth #15879 (state checkout) x unsloth-zoo #2227 (cloned here). Observes only.

Arms (the state's unsloth checkout + one zoo worktree, both first on PYTHONPATH):
  state base: base         = unsloth base + zoo base
              zoo_only     = unsloth base + zoo #2227 head
  state head: head         = unsloth head + zoo head
              unsloth_only = unsloth head + zoo base
Cells (g3f32_cell.py, one process each), per model (tiny random Gemma3, gemma-3-270m-it):
  a    UNSLOTH_FORCE_FLOAT32=1 at import, bf16 load, SFT
  b    env unset, fp16 load then bf16 load in one process, SFT on the bf16 model
  c16 / c4   fp16 load alone (forced path), 16-bit / 4-bit, SFT
  d16 / d4   bf16 load alone, 16-bit / 4-bit, SFT
Full arms run every cell; single-PR arms run a and b. The base arm repeats c16 / d16 (A/A).
Then the PRs' new tests: unsloth tests/test_force_float32_import_patches.py and zoo
tests/test_gemma3_forced_float32_undo.py, taken from the PR heads and run against the state's code.
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
CELL = HERE / "g3f32_cell.py"
ZOO_URL = "https://github.com/unslothai/unsloth-zoo"
U_TEST = "tests/test_force_float32_import_patches.py"
Z_TEST = "tests/test_gemma3_forced_float32_undo.py"
LOADER = {"tiny": "fast_language_model", "270m": "fast_model"}
SPECS = {
    "a": {"seq": "bfloat16", "quant": 16, "import_flag": "1"},
    "b": {"seq": "float16,bfloat16", "quant": 16},
    "c16": {"seq": "float16", "quant": 16},
    "c4": {"seq": "float16", "quant": 4},
    "d16": {"seq": "bfloat16", "quant": 16},
    "d4": {"seq": "bfloat16", "quant": 4},
}
FULL = ("a", "b", "c16", "c4", "d16", "d4")
SINGLE = ("a", "b")
REPEAT = ("c16", "d16")
CLEAR = ("UNSLOTH_FORCE_FLOAT32", "TORCHDYNAMO_DISABLE", "UNSLOTH_COMPILE_DISABLE", "PYTEST_ADDOPTS",
         "UNSLOTH_ALLOW_CPU")


def run(cmd, cwd = None):
    p = subprocess.run(cmd, cwd = cwd, capture_output = True, text = True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}: {p.stderr[-500:]}")
    return p.stdout.strip()


def zoo_worktrees(work: Path, zoo_base: str, zoo_pr: int):
    src = work / "zoo_src"
    if not (src / ".git").exists():
        for attempt in range(3):
            p = subprocess.run(["git", "clone", "-q", "--filter=blob:none", ZOO_URL, str(src)], capture_output = True, text = True)
            if p.returncode == 0:
                break
            subprocess.run(["rm", "-rf", str(src)])
            time.sleep(15 * (attempt + 1))
        run(["git", "fetch", "-q", "origin", f"refs/pull/{zoo_pr}/head:zpr-head"], cwd = src)
    head = run(["git", "rev-parse", "zpr-head"], cwd = src)
    out = {}
    for name, sha in (("zoo_base", zoo_base), ("zoo_head", head)):
        wt = work / name
        if not wt.exists():
            run(["git", "worktree", "add", "-q", "--detach", str(wt), sha], cwd = src)
        out[name] = (str(wt), run(["git", "rev-parse", "HEAD"], cwd = wt))
    return src, out


def env_for(unsloth_ck, zoo_ck, extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([unsloth_ck, zoo_ck] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for k in CLEAR:
        env.pop(k, None)
    env["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra)
    return env


def run_cell(arm, model, cell, spec, unsloth_ck, zoo_ck, out_dir, work, python, timeout):
    tag = f"{arm}_{model}_{cell}"
    cw = work / tag
    cw.mkdir(parents = True, exist_ok = True)
    out = out_dir / f"cell_{tag}.json"
    log = out_dir / f"cell_{tag}.log"
    cmd = [python, "-u", str(CELL), "--out", str(out), "--model", model, "--seq", spec["seq"], "--quant", str(spec["quant"]),
           "--import-flag", spec.get("import_flag", "unset"), "--loader", LOADER[model], "--work", str(cw)]
    env = env_for(unsloth_ck, zoo_ck, {"UNSLOTH_COMPILE_LOCATION": str(cw / "compiled")})
    t0 = time.time()
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, cwd = str(cw), env = env, stdout = fh, stderr = subprocess.STDOUT,
                                timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rec = {"rc": rc, "seconds": round(time.time() - t0, 1)}
    try:
        rec.update(json.loads(out.read_text(encoding = "utf-8")))
    except Exception as e:  # noqa: BLE001
        rec["json_err"] = repr(e)
    text = log.read_text(encoding = "utf-8", errors = "replace")
    rec["log_tail"] = text[-1500:]
    rec["log_has_same_dtype"] = "same dtype" in text
    return rec


def run_pytest(test, root, unsloth_ck, zoo_ck, out_dir, tag, python):
    junit = out_dir / f"junit_{tag}.xml"
    log = out_dir / f"pytest_{tag}.log"
    cmd = [python, "-m", "pytest", "-q", "-rfEs", "--tb=short", "-p", "no:cacheprovider", f"--junitxml={junit}", test]
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, cwd = root, env = env_for(unsloth_ck, zoo_ck, {}), stdout = fh,
                                stderr = subprocess.STDOUT, timeout = 1800).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res = {"rc": rc, "passed": [], "failed": [], "skipped": []}
    try:
        for tc in ET.parse(junit).getroot().iter("testcase"):
            tid = f"{tc.get('classname')}::{tc.get('name')}"
            kinds = [c.tag for c in tc]
            k = "failed" if ("failure" in kinds or "error" in kinds) else "skipped" if "skipped" in kinds else "passed"
            res[k].append(tid)
    except Exception as e:  # noqa: BLE001
        res["junit_error"] = repr(e)
    res["tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-2500:]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--work", required = True, type = Path)
    ap.add_argument("--zoo-base", required = True)
    ap.add_argument("--zoo-pr", type = int, default = 2227)
    ap.add_argument("--unsloth-head", required = True)
    ap.add_argument("--models", default = "tiny,270m")
    ap.add_argument("--timeout", type = int, default = 1500)
    a = ap.parse_args()
    a.out = a.out.resolve()
    out_dir = a.out.parent
    a.work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": a.state, "checkout": a.checkout, "arms": {}, "pytest": {}}
    obs["git_head"] = run(["git", "rev-parse", "HEAD"], cwd = a.checkout)

    def save():
        a.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    if a.state == "merge":
        obs["note"] = "merge state not probed (base and head carry the evidence)"
        save()
        return 0
    zsrc, wts = zoo_worktrees(a.work, a.zoo_base, a.zoo_pr)
    obs["zoo"] = {k: {"path": p, "sha": s} for k, (p, s) in wts.items()}
    if a.state == "base":
        arms = [("base", "zoo_base", FULL + tuple(f"{c}_r2" for c in REPEAT)), ("zoo_only", "zoo_head", SINGLE)]
    else:
        arms = [("head", "zoo_head", FULL), ("unsloth_only", "zoo_base", SINGLE)]
    models = a.models.split(",")
    # Download once, before any cell, so no cell races the hub.
    try:
        from huggingface_hub import snapshot_download
        from g3f32_cell import MODELS  # noqa: E402
        for m in models:
            snapshot_download(MODELS[m], allow_patterns = ["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt", "*.jinja"])
    except Exception as e:  # noqa: BLE001
        obs["prefetch_error"] = repr(e)
    for arm, zk, cells in arms:
        zoo_ck = wts[zk][0]
        obs["arms"][arm] = {"zoo": zk, "zoo_sha": wts[zk][1], "unsloth_sha": obs["git_head"], "cells": {}}
        for model in models:
            for cell in cells:
                spec = SPECS[cell.removesuffix("_r2")]
                r = run_cell(arm, model, cell, spec, a.checkout, zoo_ck, out_dir, a.work, a.python, a.timeout)
                obs["arms"][arm]["cells"][f"{model}/{cell}"] = r
                print(f"[{arm}] {model}/{cell}: rc={r['rc']} stage={r.get('stage')} {r['seconds']}s "
                      f"losses={r.get('losses')} err={str(r.get('error'))[:120]}", flush = True)
                save()
    # The PRs' new tests, from the PR heads, against this state's code.
    zoo_ck = wts["zoo_base" if a.state == "base" else "zoo_head"][0]
    try:
        body = run(["git", "show", f"{a.unsloth_head}:{U_TEST}"], cwd = a.checkout)
        Path(a.checkout, U_TEST).write_text(body + "\n", encoding = "utf-8")
        obs["pytest"]["unsloth"] = run_pytest(U_TEST, a.checkout, a.checkout, zoo_ck, out_dir, f"{a.state}_unsloth", a.python)
    except Exception as e:  # noqa: BLE001
        obs["pytest"]["unsloth"] = {"error": repr(e)}
    save()
    try:
        body = run(["git", "show", f"zpr-head:{Z_TEST}"], cwd = zsrc)
        Path(zoo_ck, Z_TEST).write_text(body + "\n", encoding = "utf-8")
        obs["pytest"]["zoo"] = run_pytest(Z_TEST, zoo_ck, a.checkout, zoo_ck, out_dir, f"{a.state}_zoo", a.python)
    except Exception as e:  # noqa: BLE001
        obs["pytest"]["zoo"] = {"error": repr(e)}
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
