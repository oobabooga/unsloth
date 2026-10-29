#!/usr/bin/env python3
"""Probe: unsloth-zoo compiled-module key follow-up to #2227 (compile pass redone when UNSLOTH_FORCE_FLOAT32 changes).

Both states use the same unsloth (patch mode on unsloth main). The head state's only change is an inert file,
amd_ci_zoo_fix.patch, holding the zoo fix. Arms:
  state base: base = unsloth main + zoo main (6ecdc1d3, #2227 merged)
  state head: fix  = unsloth main + zoo main + amd_ci_zoo_fix.patch
Cells (g3f32_cell.py, one process each unless noted), per model (tiny random Gemma3, gemma-3-270m-it):
  b    fp16 load then bf16 load in one process, SFT the bf16 model
  bb   bf16 then bf16 in one process (control)
  r    bf16 load then fp16 load in one process, SFT the fp16 model
  c16 / c4   fp16 alone; d16 / d4   bf16 alone
  xb   fp16 process, then a bf16 process reading the SAME compiled cache dir (the bf16 process is recorded)
  xr   bf16 process, then an fp16 process on the same cache dir (the fp16 process is recorded)
The base arm repeats c16 / d16 (A/A). Then the fix's CPU test (tests/test_compile_force_float32_key.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELL = HERE / "g3f32_cell.py"
ZOO_URL = "https://github.com/unslothai/unsloth-zoo"
PATCH = "amd_ci_zoo_fix.patch"
Z_TEST = "tests/test_compile_force_float32_key.py"
LOADER = {"tiny": "fast_language_model", "270m": "fast_model"}
SPECS = {
    "b": {"seq": "float16,bfloat16", "quant": 16},
    "bb": {"seq": "bfloat16,bfloat16", "quant": 16},
    "r": {"seq": "bfloat16,float16", "quant": 16},
    "c16": {"seq": "float16", "quant": 16},
    "c4": {"seq": "float16", "quant": 4},
    "d16": {"seq": "bfloat16", "quant": 16},
    "d4": {"seq": "bfloat16", "quant": 4},
    "xb": {"seq": "bfloat16", "quant": 16, "pre": "float16"},
    "xr": {"seq": "float16", "quant": 16, "pre": "bfloat16"},
}
FULL = ("b", "bb", "r", "c16", "c4", "d16", "d4", "xb", "xr")
REPEAT = ("c16", "d16")
CLEAR = ("UNSLOTH_FORCE_FLOAT32", "TORCHDYNAMO_DISABLE", "UNSLOTH_COMPILE_DISABLE", "PYTEST_ADDOPTS",
         "UNSLOTH_ALLOW_CPU", "UNSLOTH_COMPILE_OVERWRITE")


def run(cmd, cwd = None):
    p = subprocess.run(cmd, cwd = cwd, capture_output = True, text = True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}: {p.stderr[-800:]}")
    return p.stdout.strip()


def zoo_worktree(work: Path, zoo_base: str, name: str, patch: Path | None):
    src = work / "zoo_src"
    if not (src / ".git").exists():
        for attempt in range(3):
            p = subprocess.run(["git", "clone", "-q", "--filter=blob:none", ZOO_URL, str(src)], capture_output = True, text = True)
            if p.returncode == 0:
                break
            subprocess.run(["rm", "-rf", str(src)])
            time.sleep(15 * (attempt + 1))
    wt = work / name
    if not wt.exists():
        run(["git", "worktree", "add", "-q", "--detach", str(wt), zoo_base], cwd = src)
        if patch is not None:
            run(["git", "apply", "--index", str(patch)], cwd = wt)
            run(["git", "-c", "user.name=amd-ci", "-c", "user.email=amd-ci@localhost", "commit", "-q", "-m",
                 "zoo fix"], cwd = wt)
    return str(wt), run(["git", "rev-parse", "HEAD"], cwd = wt), run(["git", "diff", "--stat", zoo_base, "HEAD"], cwd = wt)


def env_for(unsloth_ck, zoo_ck, extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([unsloth_ck, zoo_ck] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for k in CLEAR:
        env.pop(k, None)
    env["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra)
    return env


def module_info(compiled: Path):
    f = compiled / "unsloth_compiled_module_gemma3.py"
    if not f.is_file():
        return {"present": False}
    text = f.read_text(encoding = "utf-8")
    # The modeling-module import line order is an unstable argsort; hash it as a set.
    norm = []
    pre = "from transformers.models.gemma3.modeling_gemma3 import ("
    for line in text.splitlines():
        if line.startswith(pre):
            line = pre + ",".join(sorted(x.strip() for x in line[len(pre):-1].split(","))) + ")"
        norm.append(line)
    return {"present": True, "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
            "norm_sha": hashlib.sha256("\n".join(norm).encode()).hexdigest()[:16],
            "forced": "_gemma3_rms_norm_float32" in text}


def one(cmd, env, cwd, log, timeout):
    with open(log, "ab") as fh:
        try:
            return subprocess.run(cmd, cwd = str(cwd), env = env, stdout = fh, stderr = subprocess.STDOUT,
                                  timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            return "timeout"


def run_cell(arm, model, cell, spec, unsloth_ck, zoo_ck, out_dir, work, python, timeout):
    tag = f"{arm}_{model}_{cell}"
    cw = work / tag
    cw.mkdir(parents = True, exist_ok = True)
    out = out_dir / f"cell_{tag}.json"
    log = out_dir / f"cell_{tag}.log"
    env = env_for(unsloth_ck, zoo_ck, {"UNSLOTH_COMPILE_LOCATION": str(cw / "compiled")})
    base_cmd = [python, "-u", str(CELL), "--model", model, "--quant", str(spec["quant"]), "--import-flag", "unset",
                "--loader", LOADER[model]]
    rec = {}
    t0 = time.time()
    if spec.get("pre"):
        pre_out = out_dir / f"cell_{tag}_pre.json"
        pw = cw / "pre"
        pw.mkdir(exist_ok = True)
        rec["pre_rc"] = one(base_cmd + ["--out", str(pre_out), "--seq", spec["pre"], "--work", str(pw)], env, cw, log, timeout)
        rec["pre_module"] = module_info(cw / "compiled")
        try:
            pre = json.loads(pre_out.read_text(encoding = "utf-8"))
            rec["pre_stage"], rec["pre_losses"] = pre.get("stage"), pre.get("losses")
        except Exception as e:  # noqa: BLE001
            rec["pre_json_err"] = repr(e)
    rc = one(base_cmd + ["--out", str(out), "--seq", spec["seq"], "--work", str(cw)], env, cw, log, timeout)
    rec.update({"rc": rc, "seconds": round(time.time() - t0, 1)})
    try:
        rec.update(json.loads(out.read_text(encoding = "utf-8")))
    except Exception as e:  # noqa: BLE001
        rec["json_err"] = repr(e)
    text = log.read_text(encoding = "utf-8", errors = "replace")
    rec["log_tail"] = text[-1500:]
    rec["module"] = module_info(cw / "compiled")
    dst = out_dir / f"compiled_{tag}"
    dst.mkdir(exist_ok = True)
    for f in (cw / "compiled").glob("*gemma3*.py"):
        shutil.copy2(f, dst / f.name)
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
        obs["note"] = "merge state not probed"
        save()
        return 0
    patch = Path(a.checkout, PATCH)
    obs["patch_present"] = patch.is_file()
    if a.state == "base":
        arm, zname, cells, zpatch = "base", "zoo_base", FULL + tuple(f"{c}_r2" for c in REPEAT), None
    else:
        arm, zname, cells, zpatch = "fix", "zoo_fix", FULL, patch.resolve()
        if not patch.is_file():
            obs["error"] = f"{PATCH} missing from the head checkout"
            save()
            return 0
    zoo_ck, zsha, zstat = zoo_worktree(a.work, a.zoo_base, zname, zpatch)
    obs["zoo"] = {zname: {"path": zoo_ck, "sha": zsha, "diffstat": zstat}}
    models = a.models.split(",")
    try:
        from huggingface_hub import snapshot_download
        sys.path.insert(0, str(HERE))
        from g3f32_cell import MODELS  # noqa: E402
        for m in models:
            snapshot_download(MODELS[m], allow_patterns = ["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt", "*.jinja"])
    except Exception as e:  # noqa: BLE001
        obs["prefetch_error"] = repr(e)
    obs["arms"][arm] = {"zoo": zname, "zoo_sha": zsha, "unsloth_sha": obs["git_head"], "cells": {}}
    for model in models:
        for cell in cells:
            spec = SPECS[cell.removesuffix("_r2")]
            r = run_cell(arm, model, cell, spec, a.checkout, zoo_ck, out_dir, a.work, a.python, a.timeout)
            obs["arms"][arm]["cells"][f"{model}/{cell}"] = r
            print(f"[{arm}] {model}/{cell}: rc={r['rc']} stage={r.get('stage')} {r['seconds']}s "
                  f"losses={r.get('losses')} module={r.get('module')} err={str(r.get('error'))[:120]}", flush = True)
            save()
    # The fix's CPU test, run against this arm's zoo (copied into the base worktree).
    try:
        if a.state == "base":
            # The base checkout has no patch; take the test from the head state's (states.py made both).
            head_patch = (Path(a.checkout).parent / "head" / PATCH).resolve()
            fix_wt, _, _ = zoo_worktree(a.work, a.zoo_base, "zoo_fix_for_test", head_patch)
            shutil.copy2(Path(fix_wt, Z_TEST), Path(zoo_ck, Z_TEST))
        obs["pytest"]["zoo"] = run_pytest(Z_TEST, zoo_ck, a.checkout, zoo_ck, out_dir, f"{a.state}_zoo", a.python)
    except Exception as e:  # noqa: BLE001
        obs["pytest"]["zoo"] = {"error": repr(e)}
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
