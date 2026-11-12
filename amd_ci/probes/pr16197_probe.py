#!/usr/bin/env python3
"""Probe for unsloth #16197 (state checkout) x unsloth-zoo #2296 (cloned here): reset the inference stance
counter at every generate. Observes only.

Arms (the state's unsloth checkout + one zoo worktree, both first on PYTHONPATH):
  state base: base = unsloth merge base + zoo merge base (main of both repos at the PR merge bases)
  state head: head = unsloth PR head + zoo PR head
              ufix = unsloth PR head + zoo merge base (the older-zoo fallback in vision.py), 270m sequence only
  state merge: not probed (base and head carry the evidence)

Cells per arm, one fresh process each, own UNSLOTH_COMPILE_LOCATION (pr16197_cell.py):
  270m: sequence --phases, then --same repeats of --same-shape, then 3 LoRA SFT steps
  1b:   sequence --phases only
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
CELL = HERE / "pr16197_cell.py"
ZOO_URL = "https://github.com/unslothai/unsloth-zoo"
CLEAR = ("UNSLOTH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "UNSLOTH_FORCE_FLOAT32", "PYTEST_ADDOPTS",
         "UNSLOTH_COMPILE_OVERWRITE", "UNSLOTH_FLEX_ATTENTION_GENERATE", "UNSLOTH_ENABLE_LOGGING")


def run(cmd, cwd = None):
    p = subprocess.run(cmd, cwd = cwd, capture_output = True, text = True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}: {p.stderr[-500:]}")
    return p.stdout.strip()


def zoo_worktrees(work: Path, zoo_pr: int):
    src = work / "zoo_src"
    if not (src / ".git").exists():
        for attempt in range(3):
            p = subprocess.run(["git", "clone", "-q", "--filter=blob:none", ZOO_URL, str(src)], capture_output = True,
                               text = True)
            if p.returncode == 0:
                break
            subprocess.run(["rm", "-rf", str(src)])
            time.sleep(15 * (attempt + 1))
        run(["git", "fetch", "-q", "origin", f"refs/pull/{zoo_pr}/head:zpr-head"], cwd = src)
    head = run(["git", "rev-parse", "zpr-head"], cwd = src)
    main = run(["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"], cwd = src)
    base = run(["git", "merge-base", main, "zpr-head"], cwd = src)
    out = {}
    for name, sha in (("zoo_base", base), ("zoo_head", head)):
        wt = work / name
        if not wt.exists():
            run(["git", "worktree", "add", "-q", "--detach", str(wt), sha], cwd = src)
        out[name] = (str(wt), run(["git", "rev-parse", "HEAD"], cwd = wt))
    return out


def env_for(unsloth_ck, zoo_ck, extra):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([unsloth_ck, zoo_ck] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for k in CLEAR:
        env.pop(k, None)
    env["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra)
    return env


def run_cell(arm, model, cell_args, unsloth_ck, zoo_ck, out_dir, work, python, timeout):
    tag = f"{arm}_{model}"
    cw = work / tag
    cw.mkdir(parents = True, exist_ok = True)
    out = out_dir / f"cell_{tag}.json"
    log = out_dir / f"cell_{tag}.log"
    cmd = [python, "-u", str(CELL), "--out", str(out), "--model", model, "--work", str(cw), *cell_args]
    env = env_for(unsloth_ck, zoo_ck, {"UNSLOTH_COMPILE_LOCATION": str(cw / "compiled")})
    t0 = time.time()
    with open(log, "wb") as fh:
        try:
            rc = subprocess.run(cmd, cwd = str(cw), env = env, stdout = fh, stderr = subprocess.STDOUT,
                                timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rec = {"rc": rc, "seconds": round(time.time() - t0, 1), "cmd": " ".join(cmd[2:])}
    try:
        rec.update(json.loads(out.read_text(encoding = "utf-8")))
    except Exception as e:  # noqa: BLE001
        rec["json_err"] = repr(e)
    rec["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-2500:]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--work", type = Path, default = None)
    ap.add_argument("--zoo-pr", type = int, default = 2296)
    ap.add_argument("--phases", default = "2048x1,2048x32,2048x1,2048x1,2048x32")
    ap.add_argument("--same", type = int, default = 5)
    ap.add_argument("--same-shape", default = "2048x32")
    ap.add_argument("--sft", type = int, default = 3)
    ap.add_argument("--models", default = "270m,1b")
    ap.add_argument("--timeout", type = int, default = 2700)
    a = ap.parse_args()
    a.out = a.out.resolve()
    out_dir = a.out.parent
    if a.work is None:
        a.work = Path(os.environ.get("AMD_CI_WORK") or out_dir.parent) / "p16197_work"
    a.work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": a.state, "checkout": a.checkout, "arms": {},
                 "params": {"phases": a.phases, "same": a.same, "same_shape": a.same_shape, "sft": a.sft,
                            "models": a.models}}
    obs["git_head"] = run(["git", "rev-parse", "HEAD"], cwd = a.checkout)

    def save():
        a.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    if a.state == "merge":
        obs["note"] = "merge state not probed (base and head carry the evidence)"
        save()
        return 0
    try:
        wts = zoo_worktrees(a.work, a.zoo_pr)
    except Exception as e:  # noqa: BLE001
        obs["zoo_error"] = repr(e)
        save()
        return 0
    obs["zoo"] = {k: {"path": p, "sha": s} for k, (p, s) in wts.items()}
    models = a.models.split(",")
    try:
        from huggingface_hub import snapshot_download
        sys.path.insert(0, str(HERE))
        from pr16197_cell import MODELS  # noqa: E402
        for m in models:
            snapshot_download(MODELS[m], allow_patterns = ["*.json", "*.safetensors", "*.model", "tokenizer*",
                                                           "*.txt", "*.jinja"])
    except Exception as e:  # noqa: BLE001
        obs["prefetch_error"] = repr(e)
    seq = ["--phases", a.phases]
    full = seq + ["--same", str(a.same), "--same-shape", a.same_shape, "--sft", str(a.sft)]
    if a.state == "base":
        plan = [("base", "zoo_base", "270m", full), ("base", "zoo_base", "1b", seq)]
    else:
        plan = [("head", "zoo_head", "270m", full), ("head", "zoo_head", "1b", seq), ("ufix", "zoo_base", "270m", seq)]
    for arm, zk, model, cargs in plan:
        if model not in models:
            continue
        arm_rec = obs["arms"].setdefault(arm, {"zoo": zk, "zoo_sha": wts[zk][1], "unsloth_sha": obs["git_head"],
                                               "zoo_path": wts[zk][0], "cells": {}})
        r = run_cell(arm, model, cargs, a.checkout, wts[zk][0], out_dir, a.work, a.python, a.timeout)
        arm_rec["cells"][model] = r
        print(f"[{arm}] {model}: rc={r['rc']} stage={r.get('stage')} {r['seconds']}s err={str(r.get('error'))[:200]}",
              flush = True)
        save()
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
