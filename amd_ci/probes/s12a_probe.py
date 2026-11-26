#!/usr/bin/env python3
"""A12-2 probe: zoo Triton grouped GEMM vs torch._grouped_mm's per-expert loop for MoE LoRA on gfx1151, one state.
Observes only; JSON via --out.

base state: builds the stack (Linux: zoo_venv.build_venv, AMD gfx1151 torch; Windows: the workflow's AMD ROCm torch
            venv + head deps + triton-windows), builds the shared model dir, runs base bf16 + fp16 once each and
            saves their LoRA grads as the parity references.
head state: the timing matrix, interleaved ABBA on one process per run so drift cancels:
            base, head, head, base, base, head (bf16; base arms import states/base, head arms states/head),
            then head with UNSLOTH_MOE_GROUPED_TRITON=0 (in-tree loop control) and head fp16 (parity vs base fp16).
Every run is its own process with its own UNSLOTH_COMPILE_LOCATION / TRITON_CACHE_DIR / inductor cache.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zoo_venv  # noqa: E402

CELL = HERE / "s12a_train.py"
WIN = sys.platform == "win32"

ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = sys.executable)
ap.add_argument("--unsloth-ref", required = True)
ap.add_argument("--torch-pin", default = "torch==2.11.0+rocm7.13.0")
ap.add_argument("--transformers", default = "5.17.0")
ap.add_argument("--trl-peft", default = "trl==1.18.0,peft==0.21.3")
ap.add_argument("--steps", type = int, default = 16)
ap.add_argument("--warmup", type = int, default = 4)
ap.add_argument("--timeout", type = int, default = 2400)
a = ap.parse_args()
a.out = a.out.resolve()
ck = str(Path(a.checkout).resolve())
CK = {"base": str(Path(ck).parent / "base"), "head": str(Path(ck).parent / "head")}
out_dir = a.out.parent / f"{a.state}_artifacts"
out_dir.mkdir(parents = True, exist_ok = True)
root = Path(os.environ.get("AMD_CI_WORK") or out_dir.parent)
shared = root / "s12a_shared"
shared.mkdir(parents = True, exist_ok = True)
model_dir = shared / "model"
obs: dict = {"state": a.state, "checkout": ck, "platform": sys.platform, "checkouts": CK}


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


if a.state not in ("base", "head"):
    obs["_skipped"] = ["everything"]
    save()
    sys.exit(0)
obs["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = ck, capture_output = True, text = True).stdout.strip()


def pipi(py, *specs, log, timeout = 2400, no_deps = False, reinstall = False):
    uv = shutil.which("uv")
    if uv and not WIN:
        cmd = [uv, "pip", "install", "--python", py, "--index-strategy", "unsafe-best-match",
               "--extra-index-url", zoo_venv.PYPI]
        if reinstall:
            cmd.append("--reinstall")
    else:
        cmd = [py, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
        if reinstall:
            cmd.append("--force-reinstall")
    if no_deps:
        cmd.append("--no-deps")
    return zoo_venv.sh(cmd + list(specs), log, timeout = timeout)


def pip_list(py):
    pl = subprocess.run([py, "-m", "pip", "list"], capture_output = True, text = True)
    return [ln for ln in (pl.stdout or "").splitlines() if ln.split(" ")[0].lower() in
            ("torch", "triton", "triton-windows", "pytorch-triton-rocm", "transformers", "peft", "trl", "unsloth",
             "unsloth-zoo", "unsloth_zoo", "accelerate")]


# ---- stack, built once at the base state and reused at the head ----
mark = shared / "stack.json"
if not mark.is_file():
    log = out_dir / "stack.log"
    st: dict = {}
    if WIN:
        py = a.python
        st["deps"] = pipi(py, CK["head"], "triton-windows>=3.6,<3.7", log = log)
    else:
        v = zoo_venv.build_venv(shared / "venv", CK["head"], a.torch_pin, "amd", a.transformers, out_dir)
        st["venv"] = v
        if v.get("error"):
            mark.write_text(json.dumps(st, default = str), encoding = "utf-8")
        py = v.get("python")
    if not st.get("venv", {}).get("error"):
        st["trl_peft"] = pipi(py, *a.trl_peft.split(","), f"transformers=={a.transformers}", log = log)
        st["unsloth"] = pipi(py, f"unsloth @ git+https://github.com/unslothai/unsloth@{a.unsloth_ref}", log = log,
                             no_deps = True, reinstall = True)
        st["python"] = py
        st["pip"] = pip_list(py)
    st["log_tail"] = log.read_text(encoding = "utf-8", errors = "replace")[-2500:] if log.is_file() else ""
    mark.write_text(json.dumps(st, default = str), encoding = "utf-8")
obs["stack"] = json.loads(mark.read_text(encoding = "utf-8"))
save()
py = obs["stack"].get("python")
if not py:
    obs["error"] = "stack build failed"
    save()
    sys.exit(0)

CLEAR = ("UNSLOTH_MOE_GROUPED_TRITON", "UNSLOTH_DISABLE_MOE_TRITON", "UNSLOTH_MOE_GROUPED_TRITON_MAX_ROWS",
         "UNSLOTH_MOE_BACKEND", "UNSLOTH_COMPILE_LOCATION", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")


def run(tag: str, arm: str, dtype: str, extra_args: list, env_extra: dict | None = None, build = False) -> dict:
    cache = shared / "caches" / tag
    cache.mkdir(parents = True, exist_ok = True)
    env = {k: v for k, v in zoo_venv.state_env(CK[arm]).items() if k not in CLEAR}
    env.update({"UNSLOTH_COMPILE_LOCATION": str(cache / "unsloth_compiled_cache"),
                "TRITON_CACHE_DIR": str(cache / "triton"), "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
                "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "PYTHONUNBUFFERED": "1"})
    env.update(env_extra or {})
    out = out_dir / f"run_{tag}.json"
    lg = out_dir / f"run_{tag}.log"
    cmd = [py, "-u", str(CELL), "--out", str(out), "--model-dir", str(model_dir), "--arm", arm, "--dtype", dtype,
           "--steps", str(a.steps), "--warmup", str(a.warmup)] + (["--build"] if build else []) + extra_args
    with open(lg, "ab") as fh:
        fh.write(("$ " + " ".join(cmd) + "\n").encode())
    rc = zoo_venv.sh(cmd, lg, timeout = a.timeout, env = env, cwd = str(cache))
    rec: dict = {"rc": rc, "tag": tag, "arm": arm, "checkout": CK[arm]}
    if out.is_file():
        try:
            rec.update(json.loads(out.read_text(encoding = "utf-8")))
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = repr(e)
    else:
        rec["missing_output"] = True
    rec["log_tail"] = lg.read_text(encoding = "utf-8", errors = "replace")[-2500:] if lg.is_file() else ""
    return rec


grads = shared / "grads"
grads.mkdir(exist_ok = True)
obs["runs"] = {}
if a.state == "base":
    if not (model_dir / "config.json").is_file():
        obs["build"] = run("build", "base", "bf16", [], build = True)
        save()
    obs["model_built"] = (model_dir / "config.json").is_file()
    plan = [("base_ref_bf16", "base", "bf16", ["--grads-save", str(grads / "base_bf16.pt"), "--profile"], None),
            ("base_ref_fp16", "base", "fp16", ["--grads-save", str(grads / "base_fp16.pt")], None)]
else:
    rb, rf = ["--grads-ref", str(grads / "base_bf16.pt")], ["--grads-ref", str(grads / "base_fp16.pt")]
    plan = [("t1_base", "base", "bf16", rb, None), ("t1_head", "head", "bf16", rb + ["--profile"], None),
            ("t2_head", "head", "bf16", rb, None), ("t2_base", "base", "bf16", rb, None),
            ("t3_base", "base", "bf16", rb, None), ("t3_head", "head", "bf16", rb, None),
            ("ctl_head_off", "head", "bf16", rb, {"UNSLOTH_MOE_GROUPED_TRITON": "0"}),
            ("head_fp16", "head", "fp16", rf, None)]
for tag, arm, dtype, extra, env_extra in plan:
    obs["runs"][tag] = run(tag, arm, dtype, extra, env_extra)
    save()
save()
