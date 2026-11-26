#!/usr/bin/env python3
"""unsloth PR 19833 probe, one state: the ROCm Triton probe kernel (`_triton_probe` / `triton_runs_on_device`) on
gfx1151 under two torch venvs: an old ROCm torch whose Triton is < 3.4 (where a function-local `tl` is a NameError),
and the current one (no-regression arm). Each stage (s12b3_triton_cell.py) is its own process under
-X faulthandler with its own TRITON_CACHE_DIR, so a crash in one stage leaves the others. Observes only; JSON via --out."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s12b3_venv as V  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--old-torch", default = "torch==2.7.1+rocm7.10.0")
ap.add_argument("--old-torch-711", default = "torch==2.7.1+rocm7.11.0")
ap.add_argument("--new-torch", default = "torch==2.11.0+rocm7.13.0")
a = ap.parse_args()
a.out = a.out.resolve()
ck = Path(a.checkout).resolve()
out_dir = a.out.parent / f"{a.state}_artifacts"
out_dir.mkdir(parents = True, exist_ok = True)
root = Path(os.environ.get("AMD_CI_WORK") or out_dir.parent)
obs: dict = {"state": a.state, "checkout": str(ck), "arms": {}}


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


obs["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = ck, capture_output = True, text = True).stdout.strip()
save()
# torch 2.7.1 ROCm wheels stop at cp313: ask uv for 3.12 when the runner's python is newer.
old_py = "3.12" if sys.version_info >= (3, 14) else None
STAGES = ("torch_launch", "triton_driver", "compile_only", "probe", "runs_on_device")
for arm, pin, preq in (("old", a.old_torch, old_py), ("old711", a.old_torch_711, old_py), ("new", a.new_torch, None)):
    venv = root / "s12b3_shared" / ("venv_" + pin.replace("=", "_").replace("+", "_"))
    venv.parent.mkdir(parents = True, exist_ok = True)
    v = V.build(venv, [pin], out_dir / f"venv_{arm}.log", python_req = preq)
    rec: dict = {"pin": pin, "venv": {k: v.get(k) for k in ("error", "log_tail", "reused", "pip_list", "steps", "python_request", "env")},
                 "stages": {}}
    obs["arms"][arm] = rec
    save()
    if v.get("error"):
        continue
    for stage in STAGES:
        cache = root / "s12b3_shared" / "triton_cache" / f"{a.state}_{arm}_{stage}"
        cache.mkdir(parents = True, exist_ok = True)
        out = out_dir / f"cell_{arm}_{stage}.json"
        log = out_dir / f"cell_{arm}_{stage}.log"
        rc = V.sh([v["python"], "-X", "faulthandler", "-u", str(HERE / "s12b3_triton_cell.py"), "--checkout", str(ck),
                   "--stage", stage, "--out", str(out)], log, timeout = 900, cwd = str(cache),
                  env = V.clean_env({"TRITON_CACHE_DIR": str(cache), "PYTHONDONTWRITEBYTECODE": "1"}))
        st: dict = {"rc": rc}
        if out.is_file():
            st.update(json.loads(out.read_text(encoding = "utf-8")))
        else:
            st["missing_output"] = True
        st["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-3000:] if log.is_file() else ""
        rec["stages"][stage] = st
        save()
