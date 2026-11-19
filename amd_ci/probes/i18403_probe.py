#!/usr/bin/env python3
"""Probe for unsloth #18403 (ROCm GQA SDPA dispatch faults: QLoRA on gfx1200 aborts; bisected to #12740 / #14158).
Observes only; JSON to --out, never stdout. Per state:
  0. versions, unsloth importable from this checkout, bitsandbytes 4-bit usable (decided once, shared by states).
  A. kernel matrix (once per job, cached under $AMD_CI_WORK and embedded in every state's observation: the torch
     build is the same for both states): FLASH / EFFICIENT / MATH x 3 Unsloth layouts x fp16 / bf16, each cell its
     own process with AMD_SERIALIZE_KERNEL=3, plus the per-segment layout in fp16 without it.
  B. route cell: which backend each SDPA call of a packed fp16 forward + backward takes, verdict API, parity of the
     default route vs MATH-only.
  C. training cells: reporter settings, with AMD_SERIALIZE_KERNEL=3 ("serial") and without ("plain"); head also
     "check_off" (UNSLOTH_ROCM_SDPA_CHECK=0, expected to behave like the base).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = sys.executable)
ap.add_argument("--steps", type = int, default = 6)
a = ap.parse_args()
ck = Path(a.checkout).resolve()
obs: dict = {"state": a.state, "checkout": str(ck), "platform": sys.platform}
root = Path(os.environ.get("AMD_CI_WORK") or tempfile.gettempdir())
work = Path(tempfile.mkdtemp(prefix = f"i18403_{a.state}_", dir = str(root)))
shared = root / "i18403_shared"
shared.mkdir(exist_ok = True)
model_dir = shared / "qwen3_tiny_hd128"


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


def env_for(extra = None):
    pp = [str(ck)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    e = dict(os.environ, PYTHONPATH = os.pathsep.join(pp), UNSLOTH_DISABLE_AUTO_UPDATES = "1",
             PYTHONUNBUFFERED = "1", PYTHONIOENCODING = "utf-8", PYTHONUTF8 = "1", HIP_VISIBLE_DEVICES = "0")
    for k in ("UNSLOTH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "PYTEST_ADDOPTS", "CI", "UNSLOTH_FORCE_FLOAT32",
              "UNSLOTH_SDPA_PACKED_SEGMENTS", "UNSLOTH_DISABLE_AUTO_PADDING_FREE", "UNSLOTH_ROCM_SDPA_CHECK",
              "AMD_SERIALIZE_KERNEL", "HIP_LAUNCH_BLOCKING", "UNSLOTH_RETURN_LOGITS"):
        e.pop(k, None)
    e.update({k: v for k, v in (extra or {}).items() if v is not None})
    return e


def run(cmd, timeout, **kw):
    try:
        p = subprocess.run(cmd, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                           timeout = timeout, **kw)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as e:
        return "timeout", str(e.stdout or "")[-2000:], str(e.stderr or "")[-2000:]


def load(p: Path, rc, so, se):
    try:
        c = json.loads(p.read_text(encoding = "utf-8")) if p.exists() else {"stage": "no_output"}
    except Exception as e:  # noqa: BLE001
        c = {"stage": "bad_output", "error": repr(e)}
    c.update({"rc": rc, "stderr_tail": se[-3000:], "stdout_tail": so[-1200:]})
    blob = (se + so)
    c["hip_invalid_value"] = "hipErrorInvalidValue" in blob or "invalid argument" in blob
    c["illegal_memory_access"] = "illegal memory access" in blob
    return c


for key, rev in (("commit", "HEAD"), ("tree", "HEAD^{tree}")):
    try:
        obs[key] = subprocess.run(["git", "rev-parse", rev], cwd = ck, capture_output = True, text = True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        obs[key] = f"ERR {e}"

# 0. environment
rc, so, se = run([a.python, "-m", "pip", "list", "--format=json"], 300)
if rc != 0:
    rc, so, se = run(["uv", "pip", "list", "--python", a.python, "--format=json"], 300)
try:
    want = ("torch", "transformers", "trl", "peft", "unsloth", "unsloth_zoo", "unsloth-zoo", "accelerate", "triton",
            "pytorch-triton-rocm", "bitsandbytes", "flash-attn", "flash_attn", "xformers")
    obs["pip"] = {d["name"].lower(): d["version"] for d in json.loads(so) if d["name"].lower() in want}
except Exception as e:  # noqa: BLE001
    obs["pip"] = {"error": f"{e!r} {se[-300:]}"}
rc, so, se = run([a.python, "-c", "import unsloth; print('OK', unsloth.__file__)"], 900, env = env_for(), cwd = str(work))
obs["unsloth_importable"] = rc == 0 and "OK" in so
obs["unsloth_import_tail"] = (so + se)[-1500:]
bnb_file = shared / "bnb.json"
if not bnb_file.exists():
    code = ("import torch, bitsandbytes as bnb\n"
            "l = bnb.nn.Linear4bit(256, 256, compute_dtype=torch.float16, quant_type='nf4').cuda()\n"
            "y = l(torch.randn(4, 256, device='cuda', dtype=torch.float16)); torch.cuda.synchronize()\n"
            "assert torch.isfinite(y).all(); print('BNB_OK', bnb.__version__)")
    rc, so, se = run([a.python, "-c", code], 600, env = env_for(), cwd = str(work))
    bnb_file.write_text(json.dumps({"ok": rc == 0 and "BNB_OK" in so, "tail": (so + se)[-800:], "decided_by": a.state}),
                        encoding = "utf-8")
obs["bnb"] = json.loads(bnb_file.read_text(encoding = "utf-8"))
BNB = 1 if obs["bnb"].get("ok") else 0
save()

# A. kernel matrix, once per job
matrix_file = shared / "matrix.json"
if not matrix_file.exists():
    cells = {}
    plan = [(b, l, d, "3") for b in ("FLASH", "EFFICIENT", "MATH")
            for l in ("seg_view_expanded_causal", "gqa_causal_contig", "dense_mask_expanded")
            for d in ("float16", "bfloat16")]
    plan += [(b, "seg_view_expanded_causal", "float16", None) for b in ("FLASH", "EFFICIENT", "MATH")]
    for b, l, d, ser in plan:
        name = f"{b}|{l}|{d}|{'serial' if ser else 'async'}"
        out = work / f"k_{len(cells)}.json"
        r, so, se = run([a.python, "-u", str(HERE / "i18403_kernel.py"), "--backend", b, "--layout", l, "--dtype", d,
                         "--out", str(out)], 300, cwd = str(work),
                        env = env_for({"AMD_SERIALIZE_KERNEL": ser, "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}))
        c = load(out, r, so, se)
        if c.get("stage") != "done":
            c["status"] = "crash"
        cells[name] = {k: c.get(k) for k in ("status", "stage", "rc", "max_rel", "errs", "error", "versions",
                                               "hip_invalid_value", "illegal_memory_access")}
        if c["status"] in ("crash", "error"):
            cells[name]["stderr_tail"] = c.get("stderr_tail", "")[-1200:]
    matrix_file.write_text(json.dumps({"cells": cells, "measured_in": a.state}, indent = 1), encoding = "utf-8")
obs["matrix"] = json.loads(matrix_file.read_text(encoding = "utf-8"))
save()

if not obs["unsloth_importable"]:
    save()
    sys.exit(0)

# model, once per job
if not (model_dir / "config.json").exists():
    bo = work / "build.json"
    r, so, se = run([a.python, "-u", str(HERE / "i18403_train.py"), "--build", "--model-dir", str(model_dir),
                     "--out", str(bo)], 1200, cwd = str(work), env = env_for())
    obs["build"] = load(bo, r, so, se)
    save()

# B. route
ro = work / "route.json"
cw = work / "w_route"
cw.mkdir(exist_ok = True)
r, so, se = run([a.python, "-u", str(HERE / "i18403_train.py"), "--mode", "route", "--model-dir", str(model_dir),
                 "--bnb", str(BNB), "--out", str(ro), "--work", str(cw)], 900, cwd = str(cw),
                env = env_for({"UNSLOTH_COMPILE_LOCATION": str(cw / "ucache"), "UNSLOTH_RETURN_LOGITS": "1"}))
obs["route"] = load(ro, r, so, se)
save()
if a.state != "base":
    # Head with the self-check failing (gfx1200 stand-in): calls must leave the fused kernels and keep numerics.
    ro2, cw2 = work / "route_refused.json", work / "w_route_refused"
    cw2.mkdir(exist_ok = True)
    r, so, se = run([a.python, "-u", str(HERE / "i18403_train.py"), "--mode", "route", "--model-dir", str(model_dir),
                     "--bnb", str(BNB), "--out", str(ro2), "--work", str(cw2)], 900, cwd = str(cw2),
                    env = env_for({"UNSLOTH_COMPILE_LOCATION": str(cw2 / "ucache"), "UNSLOTH_RETURN_LOGITS": "1",
                                   "I18403_FORCE_REFUSE": "1"}))
    obs["route_refused"] = load(ro2, r, so, se)
    save()

# C. training
obs["train"] = {}
cells = [("serial", {"AMD_SERIALIZE_KERNEL": "3"}), ("plain", {})]
if a.state != "base":
    cells.append(("check_off", {"UNSLOTH_ROCM_SDPA_CHECK": "0"}))
    cells.append(("refused", {"I18403_FORCE_REFUSE": "1", "AMD_SERIALIZE_KERNEL": "3"}))
for name, extra in cells:
    cw = work / f"w_{name}"
    cw.mkdir(exist_ok = True)
    co = work / f"train_{name}.json"
    r, so, se = run([a.python, "-u", str(HERE / "i18403_train.py"), "--mode", "train", "--model-dir", str(model_dir),
                     "--bnb", str(BNB), "--steps", str(a.steps), "--work", str(cw), "--out", str(co)], 1200,
                    cwd = str(cw), env = env_for({"UNSLOTH_COMPILE_LOCATION": str(cw / "ucache"), **extra}))
    obs["train"][name] = load(co, r, so, se)
    save()
save()
