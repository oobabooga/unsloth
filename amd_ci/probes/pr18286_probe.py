#!/usr/bin/env python3
"""Probe for unsloth #18286 (QAT LoRA: fake quantize each frozen base weight once). Observes only.

The head under test is a local commit (be911c33f7) that is not on GitHub: merge base 40cbeb7c5f + the PR commit + a
fix. So each state's checkout is pinned here, on both OSes, whatever states.py resolved:
  base  = 40cbeb7c5f exactly
  head  = 40cbeb7c5f + amd_ci/probes/pr18286_head_be911c3.diff, applied to the INDEX (`git apply --cached`, LF blobs,
          so a CRLF Windows checkout cannot break it) and written out with checkout-index
and the resulting `git write-tree` is recorded; the criteria compare it with the trees of 40cbeb7c5f / be911c33f7.

Per state:
  1. deps: torchao, peft, unsloth_zoo, trl, transformers, datasets importable by --python; missing ones are pip
     installed once per venv with torch held at its installed version (Windows venvs carry only ROCm torch)
  2. pytest tests/test_qat_weight_memo.py (the head's file, copied into the base too) with junit, per-test outcome
  3. pr18286_cell.py: tiny Qwen3 LoRA + int4 QAT, 6 SFT steps, eval forward; head also with UNSLOTH_QAT_WEIGHT_MEMO=0
     (kill switch). The head probe copies the base cell's losses (obs_base.json next to --out) as `base_ref`.
JSON to --out, never stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_SHA = "40cbeb7c5f03b559e564b2d41f514e67388a8dc9"
PATCH = HERE / "pr18286_head.diff"
TEST = "tests/test_qat_weight_memo.py"
NEED = ("torchao", "peft", "unsloth_zoo", "trl", "transformers", "datasets", "accelerate", "bitsandbytes")
PIP_SPECS = ["torchao>=0.15.0", "unsloth_zoo>=2026.11.6", "transformers", "trl", "peft", "accelerate", "datasets",
             "sentencepiece", "protobuf", "hf_transfer", "safetensors", "typer", "rich", "pydantic", "pyyaml", "tyro",
             "nest-asyncio", "diffusers", "bitsandbytes"]

ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = sys.executable)
ap.add_argument("--timeout", type = int, default = 2400)
a = ap.parse_args()
ck = Path(a.checkout).resolve()
work = Path(tempfile.mkdtemp(prefix = f"p18286_{a.state}_"))
obs: dict = {"state": a.state, "checkout": str(ck), "os": os.name}


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


def env_for(extra = None):
    e = dict(os.environ, PYTHONPATH = os.pathsep.join([str(ck)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
             UNSLOTH_DISABLE_AUTO_UPDATES = "1", PYTHONUNBUFFERED = "1", PYTHONIOENCODING = "utf-8", PYTHONUTF8 = "1")
    for k in ("PYTEST_ADDOPTS", "CI", "UNSLOTH_QAT_WEIGHT_MEMO", "UNSLOTH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE",
              "UNSLOTH_GC_KEEP_LAYERS"):
        e.pop(k, None)
    e.update(extra or {})
    return e


def git(*args):
    p = subprocess.run(["git", *args], cwd = ck, capture_output = True, text = True, encoding = "utf-8", errors = "replace")
    return p.returncode, (p.stdout + p.stderr).strip()


def pin():
    steps = []
    rc, msg = git("cat-file", "-e", f"{BASE_SHA}^{{commit}}")
    if rc != 0:
        steps.append(("fetch", *git("fetch", "-q", "origin", BASE_SHA)))
    steps.append(("checkout", *git("-c", "advice.detachedHead=false", "checkout", "-q", "-f", "--detach", BASE_SHA)))
    steps.append(("clean", *git("clean", "-fdq", "--", "unsloth", "tests")))
    if a.state == "head":
        # actions/checkout on Windows converts the branch's .diff to CRLF; the index blobs are LF.
        lf = work / PATCH.name
        lf.write_bytes(PATCH.read_bytes().replace(b"\r\n", b"\n"))
        steps.append(("apply_cached", *git("apply", "--cached", "--whitespace=nowarn", str(lf))))
        rc, files = git("apply", "--numstat", str(lf))
        paths = [ln.split("\t")[-1] for ln in files.splitlines() if "\t" in ln]
        steps.append(("checkout_index", *git("checkout-index", "-f", "--", *paths)))
    rc, tree = git("write-tree")
    return {"tree": tree if rc == 0 else None, "steps": [(n, r, m[-400:]) for n, r, m in steps],
            "head_commit": git("rev-parse", "HEAD")[1], "memo_file_exists": (ck / "unsloth/models/_qat_weight_memo.py").is_file()}


def probe_imports():
    code = ("import importlib, json, sys\nr = {}\n"
            f"for n in {list(NEED)!r} + ['torch']:\n"
            "    try:\n        m = importlib.import_module(n); r[n] = getattr(m, '__version__', 'ok')\n"
            "    except Exception as e:\n        r[n] = None; r[n + '_error'] = repr(e)[:300]\n"
            "try:\n    import torchao.quantization.qat; r['torchao_qat'] = True\n"
            "except Exception as e:\n    r['torchao_qat'] = False; r['torchao_qat_error'] = repr(e)[:300]\n"
            "try:\n    import torch; r['hip'] = torch.version.hip; r['gpu'] = torch.cuda.is_available()\n"
            "except Exception:\n    pass\n"
            "open(sys.argv[1], 'w', encoding='utf-8').write(json.dumps(r))\n")
    f = work / "imports.json"
    p = subprocess.run([a.python, "-c", code, str(f)], cwd = work, capture_output = True, text = True, encoding = "utf-8",
                       errors = "replace", env = env_for(), timeout = 900)
    return json.loads(f.read_text(encoding = "utf-8")) if f.is_file() else {"error": (p.stdout + p.stderr)[-800:]}


def deps():
    before = probe_imports()
    rec = {"before": before}
    missing = [n for n in NEED if not before.get(n)] + ([] if before.get("torchao_qat") else ["torchao_qat"])
    rec["missing"] = missing
    if missing and before.get("torch"):
        cons = work / "constraints.txt"
        cons.write_text(f"torch=={before['torch']}\n", encoding = "utf-8")
        specs = list(PIP_SPECS)
        if os.name == "nt":
            specs.append("triton-windows>=3.8.0.post29")
        p = subprocess.run([a.python, "-m", "pip", "install", "-q", "-c", str(cons), *specs], capture_output = True,
                           text = True, encoding = "utf-8", errors = "replace", timeout = 2400)
        rec["pip_rc"] = p.returncode
        rec["pip_tail"] = (p.stdout + p.stderr)[-2500:]
        if p.returncode != 0 and shutil.which("uv"):
            p = subprocess.run(["uv", "pip", "install", "-q", "--python", a.python, "-c", str(cons), *specs],
                               capture_output = True, text = True, encoding = "utf-8", errors = "replace", timeout = 2400)
            rec["uv_rc"] = p.returncode
            rec["uv_tail"] = (p.stdout + p.stderr)[-2500:]
        rec["after"] = probe_imports()
    return rec


def run_tests():
    junit = work / "junit.xml"
    cmd = [a.python, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", "-rA", "--color=no",
           f"--junitxml={junit}", "-o", "junit_family=xunit1", TEST]
    try:
        p = subprocess.run(cmd, cwd = ck, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                           env = env_for({"UNSLOTH_COMPILE_LOCATION": str(work / "ucache_t")}), timeout = a.timeout)
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = "timeout", f"TIMEOUT {e}"
    cases = {}
    try:
        for case in ET.parse(junit).getroot().iter("testcase"):
            tid = f"{case.get('classname')}::{case.get('name')}"
            kids = {c.tag: c for c in case}
            oc = next((o for k, o in (("error", "error"), ("failure", "failed"), ("skipped", "skipped")) if k in kids), "passed")
            el = kids.get({"error": "error", "failed": "failure", "skipped": "skipped"}.get(oc, ""), None)
            msg = "" if el is None else ((el.get("message") or "") + " " + (el.text or ""))[:700]
            cases[tid] = {"outcome": oc, "message": msg}
    except Exception as e:  # noqa: BLE001
        out += f"\nJUNIT {e!r}"
    return {"rc": rc, "cases": cases, "tail": out[-4000:]}


def cell(tag, extra):
    cw = work / f"cell_{tag}"
    cw.mkdir(exist_ok = True)
    f = cw / "cell.json"
    try:
        p = subprocess.run([a.python, "-u", str(HERE / "pr18286_cell.py"), "--work", str(cw), "--out", str(f)], cwd = cw,
                           capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                           env = env_for({"UNSLOTH_COMPILE_LOCATION": str(cw / "ucache"), **extra}), timeout = a.timeout)
        r = json.loads(f.read_text(encoding = "utf-8")) if f.is_file() else {"stage": "no_output"}
        r["rc"] = p.returncode
        r["stderr_tail"] = (p.stderr or "")[-2500:]
    except subprocess.TimeoutExpired:
        r = {"stage": "timeout"}
    return r


obs["pin"] = pin()
save()
# The head's test file in every state, so the base runs the same tests (and fails them: no module there).
shutil.copy(HERE / "pr18286_tests" / "test_qat_weight_memo.py", ck / TEST)
obs["deps"] = deps()
save()
obs["test"] = run_tests()
save()
obs["cells"] = {"default": cell("default", {})}
save()
# Control for test_compile_trace_unchanged: the same fullgraph trace with no memo (base) / memo off and on (head).
obs["compile_ctl"] = {}
for tag, extra in ((("nomemo", {}),) if a.state == "base" else (("memo_off", {"UNSLOTH_QAT_WEIGHT_MEMO": "0"}), ("memo_auto", {}))):
    f = work / f"ctl_{tag}.json"
    try:
        p = subprocess.run([a.python, str(HERE / "pr18286_compile_ctl.py"), str(f)], cwd = work, capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace", env = env_for({"UNSLOTH_COMPILE_LOCATION": str(work / "ucache_c"), **extra}),
                           timeout = 900)
        r = json.loads(f.read_text(encoding = "utf-8")) if f.is_file() else {"error": (p.stdout + p.stderr)[-1500:]}
        r["rc"] = p.returncode
    except subprocess.TimeoutExpired:
        r = {"error": "timeout"}
    obs["compile_ctl"][tag] = r
save()
if a.state == "head":
    obs["cells"]["memo_off"] = cell("memo_off", {"UNSLOTH_QAT_WEIGHT_MEMO": "0"})
    bf = a.out.parent / "obs_base.json"
    try:
        b = json.loads(bf.read_text(encoding = "utf-8"))
        obs["base_ref"] = {"losses": ((b.get("cells") or {}).get("default") or {}).get("losses"),
                           "tree": (b.get("pin") or {}).get("tree")}
    except Exception as e:  # noqa: BLE001
        obs["base_ref"] = {"error": repr(e)[:300]}
save()
