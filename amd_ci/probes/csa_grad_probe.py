#!/usr/bin/env python3
"""Probe: Studio saved-attention checkpoint around regionally compiled DiT blocks, one-step LoRA grads (gfx1151).

Observes only. Runs probes/csa/grad_job.py in a fresh process with this checkout's
studio/backend/core/training/diffusion_gc_budget.py as arm `comp_tree` (plus the bundled main / fixed copies as
comp_sac / comp_fix), dumps every arm's grads, and at the second state compares its dumps with the first state's
(two fresh processes, same job, same card): bitwise equality / max abs diff per arm.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOB = HERE / "csa" / "grad_job.py"

ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True)
ap.add_argument("--families", default = "zimage,qwen21,flux3")
ap.add_argument("--tsets", default = "mlp,preset")
ap.add_argument("--seeds", type = int, default = 2)
ap.add_argument("--res", type = int, default = 512)
ap.add_argument("--timeout", type = int, default = 2400)
args = ap.parse_args()

root = Path(os.environ.get("RUNNER_TEMP") or "/tmp") / "csa_probe"
work = root / args.state
dump = root / "dumps" / args.state
work.mkdir(parents = True, exist_ok = True)
gcb = Path(args.checkout).resolve() / "studio" / "backend" / "core" / "training" / "diffusion_gc_budget.py"
obs: dict = {"state": args.state, "gcb_tree": str(gcb), "gcb_tree_exists": gcb.is_file(), "python": sys.executable}

if gcb.is_file():
    import hashlib
    t = gcb.read_bytes()
    obs["gcb_tree_sha256"] = hashlib.sha256(t).hexdigest()
    obs["gcb_tree_equals_main_copy"] = t == (HERE / "csa" / "gcb_main.py").read_bytes()
    obs["gcb_tree_equals_fix_copy"] = t == (HERE / "csa" / "gcb_head.py").read_bytes()

env = dict(os.environ)
env.pop("UNSLOTH_DIFFUSION_GC_SAVE_ATTENTION", None)
env["TORCHINDUCTOR_CACHE_DIR"] = str(root / f"inductor_{args.state}")  # each process compiles from scratch
env["PYTHONUNBUFFERED"] = "1"
cmd = [sys.executable, "-u", str(JOB), "--families", args.families, "--tsets", args.tsets, "--seeds", str(args.seeds),
       "--res", str(args.res), "--time-steps", "0", "--out-dir", str(work / "artifacts"), "--dump-grads", str(dump)]
if gcb.is_file():
    cmd += ["--gcb-tree", str(gcb)]
obs["cmd"] = cmd
log = Path(args.out).resolve().parent / f"grad_job_{args.state}.log"
t0 = time.time()
with open(log, "wb") as fh:
    try:
        rc = subprocess.run(cmd, cwd = work, env = env, stdout = fh, stderr = subprocess.STDOUT,
                            timeout = args.timeout).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
obs["job_rc"] = rc
obs["job_s"] = round(time.time() - t0, 1)
obs["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace").splitlines()[-40:]
res_path = work / "artifacts" / "grad_results.json"
if res_path.is_file():
    obs["grad_results"] = json.loads(res_path.read_text(encoding = "utf-8"))

# Cross-process repeatability: compare with every earlier state's dumps.
xproc = {}
try:
    import torch

    for other in sorted(p for p in (root / "dumps").glob("*") if p.is_dir() and p.name != args.state):
        for f in sorted(glob.glob(str(dump / "*.pt"))):
            g = other / Path(f).name
            if not g.is_file():
                continue
            a, b = torch.load(f), torch.load(g)
            for key in sorted(set(a) & set(b)):
                if key.endswith("|comp_tree"):
                    continue  # differs by state on purpose (the checkout's module)
                ta, tb = a[key], b[key]
                if set(ta) != set(tb):
                    xproc[f"{other.name}|{key}"] = {"error": "param sets differ"}
                    continue
                mad, bitwise, num, den = 0.0, True, 0.0, 0.0
                for k in ta:
                    d = (ta[k].double() - tb[k].double())
                    mad = max(mad, float(d.abs().max()))
                    bitwise = bitwise and torch.equal(ta[k], tb[k])
                    num += float(d.norm() ** 2)
                    den += float(tb[k].double().norm() ** 2)
                xproc[f"{other.name}|{key}"] = {"bitwise": bitwise, "max_abs": mad,
                                                "rel": (num ** 0.5) / max(den ** 0.5, 1e-30)}
except Exception as e:  # noqa: BLE001
    obs["xproc_error"] = f"{type(e).__name__}: {e}"
obs["xproc"] = xproc
Path(args.out).write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
