#!/usr/bin/env python3
"""Probe: run standard jobs (jobs/*.py copies in probes/cum_jobs/) at one state, where a state is a PAIR
of (unsloth, unsloth_zoo) SHAs given on the command line, both installed --no-deps into a private
target dir that goes first on PYTHONPATH. --checkout (the PR worktree) is recorded, not used: the
cumulative comparison swaps both repos.

OBSERVES only (criteria/cum_jobs_no_regression.py judges). Result JSON goes to --out, never stdout.
Each job runs --repeat times per state; per-run metrics are kept so the criteria can take medians.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = HERE / "cum_jobs"


def sh(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output = True, text = True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--u-base", required = True)
    ap.add_argument("--u-head", required = True)
    ap.add_argument("--z-base", required = True)
    ap.add_argument("--z-head", required = True)
    ap.add_argument("--jobs", required = True, help = "';'-separated '<stem>:<args>'")
    ap.add_argument("--repeat", type = int, default = 2)
    ap.add_argument("--job-timeout", type = int, default = 2400)
    a = ap.parse_args()
    arm = "head" if a.state in ("head", "merge") else "base"
    u_sha, z_sha = (a.u_head, a.z_head) if arm == "head" else (a.u_base, a.z_base)
    root = Path(os.environ.get("RUNNER_TEMP") or "/tmp") / f"cum_{a.state}"
    site = root / "site"
    site.mkdir(parents = True, exist_ok = True)
    res = {"state": a.state, "arm": arm, "unsloth": u_sha, "unsloth_zoo": z_sha, "checkout": a.checkout,
           "python": sys.executable, "jobs": {}}
    for repo, sha in (("unslothai/unsloth-zoo", z_sha), ("unslothai/unsloth", u_sha)):
        spec = f"git+https://github.com/{repo}.git@{sha}"
        r = sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", site, spec])
        if r.returncode and "No module named pip" in (r.stdout + r.stderr):
            # uv venvs (the Studio one) carry no pip
            r = sh(["uv", "pip", "install", "--python", sys.executable, "--no-deps", "--target", site, spec])
        if r.returncode:
            res["error"] = f"install {repo}@{sha[:9]}: {(r.stdout + r.stderr)[-800:]}"
            a.out.write_text(json.dumps(res, indent = 2), encoding = "utf-8")
            return 0
    env = dict(os.environ, PYTHONPATH = str(site) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = sh([sys.executable, "-c", "import torch, json, unsloth, unsloth_zoo, importlib.util as u; "
            "print('CUMENV=' + json.dumps({'torch': torch.__version__, 'hip': torch.version.hip, "
            "'arch': getattr(torch.cuda.get_device_properties(0), 'gcnArchName', None), "
            "'zoo': unsloth_zoo.__file__, 'unsloth': u.find_spec('unsloth').origin}))"], env = env)
    try:
        # Import-time warnings (ROCm CK notices) share the stream: take the marked line only.
        line = next(l for l in reversed((r.stdout or "").splitlines()) if l.startswith("CUMENV="))
        res["env"] = json.loads(line[len("CUMENV="):])
    except (ValueError, StopIteration):
        res["env"] = {"error": (r.stdout + r.stderr)[-800:]}
    for spec in [s for s in a.jobs.split(";") if s.strip()]:
        stem, _, args = spec.partition(":")
        stem = stem.strip()
        runs = []
        for i in range(a.repeat):
            dest = root / f"{stem}_{abs(hash(args)) % 10**6}_{i}.json"
            dest.unlink(missing_ok = True)
            cwd = root / f"cwd_{stem}_{i}"
            cwd.mkdir(exist_ok = True)
            e = dict(env, UNSLOTH_COMPILE_LOCATION = str(root / f"compile_{stem}_{i}"),
                     TORCHINDUCTOR_CACHE_DIR = str(root / f"inductor_{stem}_{i}"))
            t0 = time.time()
            try:
                p = sh([sys.executable, "-u", JOBS / f"{stem}.py", *shlex.split(args), "--out", dest],
                       cwd = cwd, env = e, timeout = a.job_timeout)
                rc, tail = p.returncode, (p.stdout + p.stderr)[-1500:]
            except subprocess.TimeoutExpired:
                rc, tail = -9, "timeout"
            run = {"rc": rc, "secs": round(time.time() - t0, 1)}
            if dest.exists():
                d = json.loads(dest.read_text(encoding = "utf-8"))
                s = d.get("summary") or {}
                run.update(passed = d.get("passed"), finished = d.get("finished"), backend = d.get("backend"),
                           losses = [st.get("loss") for st in d.get("steps", [])],
                           steady_step_ms = s.get("steady_step_ms"), peak_mem_gb = s.get("peak_mem_gb"),
                           tokens_per_s = s.get("steady_tokens_per_s") or s.get("tokens_per_s"),
                           failed_checks = [k for k, v in (d.get("checks") or {}).items() if not v.get("ok")],
                           error = d.get("error"))
            else:
                run["error"] = tail
            runs.append(run)
        res["jobs"][spec] = runs
        a.out.write_text(json.dumps(res, indent = 2), encoding = "utf-8")
    a.out.write_text(json.dumps(res, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
