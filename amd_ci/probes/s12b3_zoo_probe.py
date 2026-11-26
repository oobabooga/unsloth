#!/usr/bin/env python3
"""zoo PR 2858 probe, one state: a refused hipHostRegister / hipHostUnregister must not leave its error pending for the
next kernel launch. Observes only; JSON via --out.

Per state: shared minimal ROCm torch venv (torch + pytest, nothing of zoo's deps: the test loads block_swap*.py by
path), the head's tests/test_block_swap_cuda_error.py (shipped byte-exact in s12b3_data/, sha256 pinned; copied into
a state that lacks it, compared where present), the files the test module really loaded, then pytest per GPU test
(own process each) and one run of the CPU tests. Repo conftest is not loaded (--noconftest): it pulls in
unsloth_zoo's package imports, which need transformers; the test file itself is standalone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s12b3_venv as V  # noqa: E402

TEST_REL = "tests/test_block_swap_cuda_error.py"
TEST_SRC = HERE / "s12b3_data" / "test_block_swap_cuda_error.py"
TEST_SHA = "d218630ee73868cb96ed18c8ad1deb3b2629fff5c1f603e1734faa890bd12d45"
GPU_TESTS = ["test_the_next_kernel_after_a_refused_arena_registration_runs",
             "test_the_next_kernel_after_a_refused_disk_buffer_registration_runs",
             "test_the_next_kernel_after_a_refused_unregistration_runs"]

ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--torch-pin", default = "torch==2.11.0+rocm7.13.0")
a = ap.parse_args()
a.out = a.out.resolve()
ck = Path(a.checkout).resolve()
out_dir = a.out.parent / f"{a.state}_artifacts"
out_dir.mkdir(parents = True, exist_ok = True)
root = Path(os.environ.get("AMD_CI_WORK") or out_dir.parent)
obs: dict = {"state": a.state, "checkout": str(ck)}


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


obs["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = ck, capture_output = True, text = True).stdout.strip()
save()

# ---- venv (shared by all states) ----
venv = root / "s12b3_shared" / ("venv_" + a.torch_pin.replace("=", "_").replace("+", "_"))
venv.parent.mkdir(parents = True, exist_ok = True)
obs["venv"] = V.build(venv, [a.torch_pin], out_dir / "venv_build.log")
save()
if obs["venv"].get("error"):
    sys.exit(0)
py = obs["venv"]["python"]
obs["env"] = obs["venv"].get("env")

# ---- the test file ----
dest = ck / TEST_REL
tf: dict = {"overlay_sha256": sha(TEST_SRC), "expected_sha256": TEST_SHA}
if dest.is_file():
    tf["present_in_tree"] = True
    tf["tree_sha256"] = sha(dest)
else:
    tf["present_in_tree"] = False
    dest.parent.mkdir(parents = True, exist_ok = True)
    shutil.copyfile(TEST_SRC, dest)
    tf["copied_from_overlay"] = True
tf["used_sha256"] = sha(dest)
obs["test_file"] = tf
save()

env = V.clean_env({"PYTHONDONTWRITEBYTECODE": "1"})

# ---- which files the test module loaded at this state ----
IMPORT_CODE = (
    "import json, runpy, sys\n"
    "g = runpy.run_path(sys.argv[1])\n"
    "d, m = g['_dmod'], g['_mod']\n"
    "print('AMDCI' + json.dumps({'block_swap_disk': d.__file__, 'block_swap': m.__file__,\n"
    "    'disk_has_consume_cuda_error': hasattr(d, 'consume_cuda_error'),\n"
    "    'block_swap_has__consume_cuda_error': hasattr(m, '_consume_cuda_error'),\n"
    "    'disk_tier_is_loaded_module': m._disk() is d, 'torch': g['torch'].__file__}))\n"
)
p = subprocess.run([py, "-c", IMPORT_CODE, str(dest)], cwd = ck, env = env, capture_output = True, text = True,
                   timeout = 600)
line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
obs["imported"] = json.loads(line[5:]) if line else {"error": ((p.stdout or "") + (p.stderr or ""))[-2500:]}
save()

# ---- raw runtime behaviour (diagnostic only, tree independent) ----
DIAG = (
    "import json, torch\n"
    "c = torch.cuda.cudart(); torch.ones(1, device='cuda').sum().item()\n"
    "b = torch.empty(1 << 20, dtype=torch.uint8)\n"
    "r1 = int(c.cudaHostRegister(b.data_ptr(), b.numel(), 1))\n"
    "r2 = int(c.cudaHostRegister(b.data_ptr(), b.numel(), 1))\n"
    "launch = 'ok'\n"
    "try:\n"
    "    (torch.ones(16, device='cuda') * 2).sum().item()\n"
    "except Exception as e:\n"
    "    launch = repr(e)[:600]\n"
    "u1 = int(c.cudaHostUnregister(b.data_ptr()))\n"
    "print('AMDCI' + json.dumps({'first_register_rc': r1, 'second_register_rc': r2, 'launch_after_refusal': launch,\n"
    "    'unregister_rc': u1, 'cudart_has_register': hasattr(c, 'cudaHostRegister')}))\n"
)
if a.state == "base":
    p = subprocess.run([py, "-c", DIAG], cwd = str(out_dir), env = env, capture_output = True, text = True, timeout = 600)
    line = next((ln for ln in (p.stdout or "").splitlines() if ln.startswith("AMDCI")), None)
    obs["raw_runtime"] = json.loads(line[5:]) if line else {"error": ((p.stdout or "") + (p.stderr or ""))[-2500:]}
    save()


def pytest_run(tag: str, sel: list, kexpr: str | None = None) -> dict:
    junit = out_dir / f"junit_{tag}.xml"
    cmd = [py, "-m", "pytest", "-v", "-rfEs", "-p", "no:cacheprovider", "--noconftest", "-o", "addopts=",
           "--timeout", "600", f"--junitxml={junit}", *sel] + (["-k", kexpr] if kexpr else [])
    rec: dict = {"cmd": " ".join(cmd)}
    try:
        r = subprocess.run(cmd, cwd = ck, env = env, capture_output = True, text = True, timeout = 1200)
        rec["rc"], out, err = r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        rec["rc"], out, err = "timeout", "", "TimeoutExpired"
    (out_dir / f"pytest_{tag}.log").write_text(out + "\n--- stderr ---\n" + err, encoding = "utf-8")
    rec["tail"], rec["stderr_tail"] = out[-3000:], err[-1500:]
    cases: dict = {}
    if junit.is_file():
        try:
            for tc in ET.parse(junit).getroot().iter("testcase"):
                kinds = {c.tag: c for c in tc}
                k = kinds.get("failure", kinds.get("error"))
                if k is not None:
                    res = "failed" if "failure" in kinds else "error"
                    msg = ((k.get("message") or "") + "\n" + (k.text or "")[-3500:])[:4500]
                elif "skipped" in kinds:
                    res, msg = "skipped", (kinds["skipped"].get("message") or "")[:400]
                else:
                    res, msg = "passed", ""
                cases[tc.get("name")] = {"result": res, "msg": msg}
        except Exception as e:  # noqa: BLE001
            rec["junit_error"] = repr(e)
    rec["cases"] = cases
    return rec


obs["gpu"] = {}
for t in GPU_TESTS:
    obs["gpu"][t] = pytest_run(f"gpu_{t}", [f"{TEST_REL}::{t}"])
    save()
obs["cpu"] = pytest_run("cpu", [TEST_REL], "not next_kernel")
save()
