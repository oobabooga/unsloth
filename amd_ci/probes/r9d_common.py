#!/usr/bin/env python3
"""Shared helpers for the round-9 gfx1151 probe (r9d_probe.py) and its item modules. Observes only.

ctx (built by r9d_probe.main) keys:
  backend       head tree studio/backend (str)        base_backend  base tree studio/backend (str) (= origin/main)
  work          Path, out.parent / "r9work": UPLOADED with the artifact, keep it small (json, logs, small pngs)
  scratch       Path, job root / "r8_scratch": NOT uploaded (models, builds, inductor caches, big tensors)
  hf            Path, HF_HOME for every child (job root / "r8_hf_home", not uploaded)
  log           Path, the children log (uploaded)
  py            the Studio venv python (sys.executable)
  win           bool, running on Windows
  save          callable: rewrite obs_<state>.json now (call after every sub-result)
No token is read or passed: child envs drop every *TOKEN* / WANDB variable; downloads pass token=False and only public
unsloth/* repos are fetched.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R9 = HERE / "r9d"
R8 = R9
SECRET_HINTS = ("TOKEN", "WANDB_API_KEY", "HF_AUTH", "API_KEY")


def child_env(ctx: dict, extra: dict | None = None, tag: str = "x") -> dict:
    env = {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in SECRET_HINTS)}
    env.pop("PYTHONPATH", None)
    env.pop("HF_ENDPOINT", None)
    env.update({"HF_HOME": str(ctx["hf"]), "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DOWNLOAD_TIMEOUT": "300",
                "HF_HUB_ETAG_TIMEOUT": "60", "HF_XET_HIGH_PERFORMANCE": "1", "UNSLOTH_DISABLE_AUTO_UPDATES": "1",
                "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",
                "TORCHINDUCTOR_CACHE_DIR": str(Path(ctx["scratch"]) / f"ind_{tag}")})
    for k, v in (extra or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return env


def run_py(ctx: dict, script, args: list, out: Path | None, timeout: int = 3600, extra_env: dict | None = None,
           tag: str = "x", cwd: str | None = None, keep_inductor: bool = False) -> dict:
    """Run ``python script args`` in a fresh process, output appended to ctx['log']; returns rc / wall plus the JSON
    the child wrote to ``out`` (if any). rc is "timeout" on a timeout."""
    t = time.time()
    with open(ctx["log"], "ab") as fh:
        fh.write(f"\n=== [{tag}] {script} {' '.join(map(str, args))} env+={sorted((extra_env or {}).items())}\n".encode())
        fh.flush()
        try:
            rc = subprocess.run([ctx["py"], "-u", str(script), *map(str, args)], stdout = fh, stderr = subprocess.STDOUT,
                                env = child_env(ctx, extra_env, tag), timeout = timeout, cwd = cwd).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res = {"rc": rc, "wall_s": round(time.time() - t, 1), "tag": tag}
    if out is not None:
        out = Path(out)
        if out.is_file():
            try:
                res.update(json.loads(out.read_text(encoding = "utf-8")))
            except Exception as exc:  # noqa: BLE001
                res["parse_error"] = repr(exc)
        else:
            res["missing"] = True
    if not keep_inductor:
        shutil.rmtree(Path(ctx["scratch"]) / f"ind_{tag}", ignore_errors = True)
    return res


def run_code(ctx: dict, code: str, args: list, timeout: int = 900, extra_env: dict | None = None, tag: str = "x",
             marker: str = "R9JSON ") -> dict:
    """Run an inline ``python -c code args``; the child prints one ``R9JSON {...}`` line; returns it (or rc + tail)."""
    t = time.time()
    try:
        r = subprocess.run([ctx["py"], "-c", code, *map(str, args)], env = child_env(ctx, extra_env, tag),
                           capture_output = True, text = True, timeout = timeout, encoding = "utf-8", errors = "replace")
        rc, so, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as exc:
        rc, so, se = "timeout", (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (
            exc.stdout or ""), ""
    with open(ctx["log"], "ab") as fh:
        fh.write(f"\n=== [{tag}] inline env+={sorted((extra_env or {}).items())} rc={rc}\n{so[-4000:]}\n{se[-4000:]}\n"
                 .encode("utf-8", "replace"))
    lines = [x for x in so.splitlines() if x.startswith(marker)]
    if lines:
        try:
            rec = json.loads(lines[-1][len(marker):])
        except Exception as exc:  # noqa: BLE001
            rec = {"parse_error": repr(exc)}
    else:
        rec = {"missing": True, "tail": (so + se)[-1500:]}
    rec.update(rc = rc, wall_s = round(time.time() - t, 1), tag = tag)
    return rec


def prefetch(ctx: dict, repo: str, patterns: list | None = None, timeout: int = 7200) -> dict:
    """Token-free snapshot of a PUBLIC unsloth/* repo into ctx['hf'] (the shared hub cache of every child)."""
    if not repo.startswith("unsloth/"):
        return {"repo": repo, "rc": "refused", "why": "only public unsloth/* repos are fetched in this job"}
    code = ("import json, sys\nfrom huggingface_hub import snapshot_download\n"
            "p = snapshot_download(sys.argv[1], token = False, max_workers = 8, allow_patterns = json.loads(sys.argv[2]))\n"
            "print('R9JSON ' + json.dumps({'path': p}))\n")
    r = run_code(ctx, code, [repo, json.dumps(patterns)], timeout = timeout, tag = "prefetch")
    r["repo"] = repo
    return r


def e2e(ctx: dict, tree: str, repo: str, tag: str, args: list, timeout: int = 5400, extra_env: dict | None = None) -> dict:
    """One fresh-process Studio image render arm (probes/r8/e2e_arm.py): load at Studio defaults, N renders."""
    od = Path(ctx["work"]) / "e2e"
    od.mkdir(parents = True, exist_ok = True)
    rec = run_py(ctx, R8 / "e2e_arm.py", ["--root", tree, "--model", repo, "--tag", tag, "--out", od, *args], None,
                 timeout = timeout, extra_env = extra_env, tag = tag)
    rp = od / f"{tag}.json"
    if rp.is_file():
        try:
            rec.update(json.loads(rp.read_text(encoding = "utf-8")))
        except Exception as exc:  # noqa: BLE001
            rec["parse_error"] = repr(exc)
    else:
        rec["verdict"] = rec.get("verdict") or "missing"
    return rec


def gmean(xs):
    import math
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def spread(xs):
    """(median, (max - min) / median) of a list of positive numbers, or (None, None)."""
    import statistics as st
    xs = [x for x in xs if isinstance(x, (int, float)) and x > 0]
    if not xs:
        return None, None
    m = st.median(xs)
    return m, (max(xs) - min(xs)) / m if m else None


_ = sys
