#!/usr/bin/env python3
"""Gate 18 A/B for zoo #2849 x #2850 on gfx1151: one MoE LoRA config (the A12-2 / #2850 shape), six arms on zoo main.

  a  main                      b  main + #2849 (the head state)
  c  main + #2850              d  main + #2849 + #2850
  e  d, UNSLOTH_MOE_BACKEND=unsloth_triton        f  d, UNSLOTH_MOE_GROUPED_TRITON=1
base state: builds the stack + model, runs arm a once (grad reference). head state: builds arm c / d checkouts by
applying the shipped patches to a copy of the base checkout, then A/A (a, a) and three interleaved rounds
(a b c d e f, f e d c b a, a b c d e f), one process per run, own caches each. Observes only; JSON via --out.
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

CELL = HERE / "abx_train.py"
PATCHES = HERE.parent / "patches"
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
ap.add_argument("--timeout", type = int, default = 3600)
a = ap.parse_args()
a.out = a.out.resolve()
ck = str(Path(a.checkout).resolve())
CK = {"a": str(Path(ck).parent / "base"), "b": str(Path(ck).parent / "head")}
out_dir = a.out.parent / f"{a.state}_artifacts"
out_dir.mkdir(parents = True, exist_ok = True)
root = Path(os.environ.get("AMD_CI_WORK") or out_dir.parent)
shared = root / "abx_shared"
shared.mkdir(parents = True, exist_ok = True)
model_dir = shared / "model"
obs: dict = {"state": a.state, "checkout": ck, "platform": sys.platform}


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
        cmd = [uv, "pip", "install", "--python", py, "--index-strategy", "unsafe-best-match", "--extra-index-url", zoo_venv.PYPI]
        if reinstall:
            cmd.append("--reinstall")
    else:
        cmd = [py, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
        if reinstall:
            cmd.append("--force-reinstall")
    if no_deps:
        cmd.append("--no-deps")
    return zoo_venv.sh(cmd + list(specs), log, timeout = timeout)


mark = shared / "stack.json"
if not mark.is_file():
    log = out_dir / "stack.log"
    st: dict = {}
    if WIN:
        py = a.python
        st["deps"] = pipi(py, CK["b"], "triton-windows>=3.6,<3.7", log = log)
    else:
        v = zoo_venv.build_venv(shared / "venv", CK["b"], a.torch_pin, "amd", a.transformers, out_dir)
        st["venv"] = v
        py = v.get("python")
    if not st.get("venv", {}).get("error"):
        st["trl_peft"] = pipi(py, *a.trl_peft.split(","), f"transformers=={a.transformers}", log = log)
        st["bitsandbytes"] = pipi(py, "bitsandbytes", log = log)
        st["unsloth"] = pipi(py, f"unsloth @ git+https://github.com/unslothai/unsloth@{a.unsloth_ref}", log = log,
                             no_deps = True, reinstall = True)
        st["python"] = py
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
         "UNSLOTH_MOE_BACKEND", "UNSLOTH_COMPILE_LOCATION", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
         "UNSLOTH_MOE_GROUPED_MM_PROBE")
ARM_ENV = {"e": {"UNSLOTH_MOE_BACKEND": "unsloth_triton"}, "f": {"UNSLOTH_MOE_GROUPED_TRITON": "1"}}
ARM_TREE = {"a": "a", "b": "b", "c": "c", "d": "d", "e": "d", "f": "d"}


def run(tag: str, arm: str, extra_args: list, build = False) -> dict:
    cache = shared / "caches" / tag
    cache.mkdir(parents = True, exist_ok = True)
    tree = CK[ARM_TREE[arm]]
    env = {k: v for k, v in zoo_venv.state_env(tree).items() if k not in CLEAR}
    env.update({"UNSLOTH_COMPILE_LOCATION": str(cache / "unsloth_compiled_cache"),
                "TRITON_CACHE_DIR": str(cache / "triton"), "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
                "UNSLOTH_DISABLE_AUTO_UPDATES": "1", "PYTHONUNBUFFERED": "1"})
    env.update(ARM_ENV.get(arm, {}))
    out = out_dir / f"run_{tag}.json"
    lg = out_dir / f"run_{tag}.log"
    cmd = [py, "-u", str(CELL), "--out", str(out), "--model-dir", str(model_dir), "--arm", arm, "--dtype", "bf16",
           "--steps", str(a.steps), "--warmup", str(a.warmup)] + (["--build"] if build else []) + extra_args
    with open(lg, "ab") as fh:
        fh.write(("$ " + " ".join(cmd) + "\n").encode())
    rc = zoo_venv.sh(cmd, lg, timeout = a.timeout, env = env, cwd = str(cache))
    rec: dict = {"rc": rc, "tag": tag, "arm": arm, "checkout": tree}
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
ref = ["--grads-ref", str(grads / "a_bf16.pt"), "--oracle-ref", str(grads / "oracle.pt")]
if a.state == "base":
    if not (model_dir / "config.json").is_file():
        obs["build"] = run("build", "a", [], build = True)
        save()
    obs["model_built"] = (model_dir / "config.json").is_file()
    plan = [("ref_a", "a", ["--grads-save", str(grads / "a_bf16.pt"), "--profile", "--save-init", str(grads / "init_a.pt")])]
else:
    # Arm c / d trees: a copy of the base checkout with the shipped patch applied (git apply, checked).
    obs["trees"] = {}
    for arm, patch in (("c", "abx_2850.diff"), ("d", "abx_both.diff")):
        dst = shared / f"tree_{arm}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(CK["a"], dst, ignore = shutil.ignore_patterns("__pycache__"))
        p = subprocess.run(["git", "apply", "--whitespace=nowarn", str(PATCHES / patch)], cwd = dst,
                           capture_output = True, text = True)
        head = subprocess.run(["git", "diff", "--stat"], cwd = dst, capture_output = True, text = True).stdout[-400:]
        obs["trees"][arm] = {"rc": p.returncode, "err": (p.stderr or "")[-800:], "stat": head}
        CK[arm] = str(dst)
    save()
    if any(t["rc"] for t in obs["trees"].values()):
        obs["error"] = "patch apply failed"
        save()
        sys.exit(0)
    R = ["a", "b", "c", "d", "e", "f"]
    plan = [("aa1_a", "a", ref), ("aa2_a", "a", ref)]
    for i, order in enumerate((R, R[::-1], R), 1):
        for arm in order:
            plan.append((f"r{i}_{arm}", arm, ref + (["--profile"] if i == 1 else [])))
for tag, arm, extra in plan:
    obs["runs"][tag] = run(tag, arm, extra)
    save()
if a.state == "base" and (grads / "init_a.pt").is_file():
    # fp64 oracle on CPU: plain transformers + PEFT from arm a's LoRA config and pre-step-0 tensors (no unsloth).
    obs["oracle"] = run("oracle", "a", ["--oracle", str(grads / "oracle.pt"), "--init", str(grads / "init_a.pt")])
    save()
save()
