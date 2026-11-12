#!/usr/bin/env python3
"""Probe for unsloth #17492: strict Dynamo regions on a torch without `suppress_errors` (AMD's ROCm torch 2.11).
Per state: one fresh-process tiny Llama LoRA SFT cell (pr17492_cell.py) plus the state's own
tests/test_strict_dynamo_flag.py and tests/test_block_compile.py. Observes only; writes JSON to --out."""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = sys.executable)
ap.add_argument("--timeout", type = int, default = 1500)
a = ap.parse_args()
ck = str(Path(a.checkout).resolve())
env = dict(os.environ, PYTHONPATH = os.pathsep.join([ck] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
           UNSLOTH_DISABLE_AUTO_UPDATES = "1", PYTHONUNBUFFERED = "1")
for k in ("UNSLOTH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "UNSLOTH_BLOCK_COMPILE", "PYTEST_ADDOPTS"):
    env.pop(k, None)
obs = {"state": a.state, "checkout": ck}
work = Path(tempfile.mkdtemp(prefix = f"p17492_{a.state}_"))
cell_out = work / "cell.json"
cenv = dict(env, UNSLOTH_COMPILE_LOCATION = str(work / "ucache"))
try:
    p = subprocess.run([a.python, "-u", str(HERE / "pr17492_cell.py"), "--out", str(cell_out)], env = cenv, cwd = work,
                       capture_output = True, text = True, timeout = a.timeout)
    obs["cell"] = json.loads(cell_out.read_text()) if cell_out.exists() else {"stage": "no_output"}
    obs["cell"]["rc"] = p.returncode
    obs["cell"]["stderr_tail"] = p.stderr[-2000:]
except subprocess.TimeoutExpired:
    obs["cell"] = {"stage": "timeout"}
tests = [t for t in ("tests/test_strict_dynamo_flag.py", "tests/test_block_compile.py") if (Path(ck) / t).exists()]
obs["tests_present"] = tests
if tests:
    p = subprocess.run([a.python, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "-o", "addopts=", *tests],
                       cwd = ck, env = env, capture_output = True, text = True, timeout = a.timeout)
    tail = p.stdout[-6000:]
    cnt = lambda w: int(m.group(1)) if (m := re.search(rf"(\d+) {w}", tail.splitlines()[-1] if tail.strip() else "")) else 0
    obs["pytest"] = {"rc": p.returncode, "n_passed": cnt("passed"), "n_failed": cnt("failed"), "n_skipped": cnt("skipped"),
                     "n_errors": cnt("error"), "failed": re.findall(r"^(?:FAILED|ERROR) (\S+)", tail, re.M), "tail": tail[-2500:]}
a.out.write_text(json.dumps(obs, indent = 1))
