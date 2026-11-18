#!/usr/bin/env python3
"""Probe for unsloth #18287 (FSDP2 replicated-table gather off the GPU; torchrun ranks > 0 re-apply allocator settings
late). Observes only. Per state:
  1. the PR's tests (tests/test_parallel_env_allocator.py, tests/test_fsdp2_sharded_gather_peak.py), from
     amd_ci/probes/pr18287_tests (the head's versions incl. the follow-up flip tests) copied into the checkout,
     so the base runs the same test file; pytest -o addopts= with junit, per-test outcome + message
  2. head only: the tests run first on the PR head as pushed (`pr`), then again after the follow-up fix
     (amd_ci/probes/pr18287_fix.patch, `git apply`), which is the head that is judged
  3. pr18287_cell.py per state (and for `pr`): `import unsloth` in its own process with CUDA/HIP started first (what a
     rank > 0 does) and not, with and without a user PYTORCH_HIP_ALLOC_CONF=expandable_segments:True; setter calls,
     FutureWarnings, import errors, expandable segments. Skipped when unsloth_zoo is not importable (Windows venv).
JSON to --out, never stdout."""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys, tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = ["tests/test_parallel_env_allocator.py", "tests/test_fsdp2_sharded_gather_peak.py"]
ap = argparse.ArgumentParser()
ap.add_argument("--state", required = True)
ap.add_argument("--checkout", required = True)
ap.add_argument("--out", required = True, type = Path)
ap.add_argument("--python", default = sys.executable)
ap.add_argument("--timeout", type = int, default = 1500)
a = ap.parse_args()
ck = Path(a.checkout).resolve()
work = Path(tempfile.mkdtemp(prefix = f"p18287_{a.state}_"))
obs: dict = {"state": a.state, "checkout": str(ck), "os": os.name}
ALLOC = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")


def save():
    a.out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")


def env_for(extra = None):
    e = dict(os.environ, PYTHONPATH = os.pathsep.join([str(ck)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
             UNSLOTH_DISABLE_AUTO_UPDATES = "1", PYTHONUNBUFFERED = "1", PYTHONIOENCODING = "utf-8", PYTHONUTF8 = "1")
    for k in ALLOC + ("PYTEST_ADDOPTS", "UNSLOTH_VLLM_STANDBY", "CI"):
        e.pop(k, None)
    e.update(extra or {})
    return e


def git(*args):
    p = subprocess.run(["git", *args], cwd = ck, capture_output = True, text = True, encoding = "utf-8", errors = "replace")
    return p.returncode, (p.stdout + p.stderr).strip()


def own_module():
    code = ("import importlib.util, json, sys\n"
            "p = sys.argv[1] + '/unsloth/parallel_env.py'\n"
            "s = importlib.util.spec_from_file_location('_pe18287', p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
            "src = open(p, encoding='utf-8').read()\n"
            "print(json.dumps({'file': m.__file__, 'has_apply': hasattr(m, 'apply_late_allocator_env'),\n"
            "                  'has_fix': 'catch_warnings' in src}))\n")
    p = subprocess.run([a.python, "-c", code, str(ck)], capture_output = True, text = True, encoding = "utf-8", errors = "replace", env = env_for())
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"error": (p.stdout + p.stderr)[-600:]}


def run_tests(tag):
    junit = work / f"{tag}.xml"
    cmd = [a.python, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", "-rA", "--color=no",
           f"--junitxml={junit}", "-o", "junit_family=xunit1", *TESTS]
    try:
        p = subprocess.run(cmd, cwd = ck, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                           env = env_for(), timeout = a.timeout)
        rc, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out = -1, f"TIMEOUT {e}"
    cases = {}
    try:
        for case in ET.parse(junit).getroot().iter("testcase"):
            tid = f"{case.get('classname')}::{case.get('name')}"
            kids = {c.tag: c for c in case}
            if "error" in kids:
                oc, el = "error", kids["error"]
            elif "failure" in kids:
                oc, el = "failed", kids["failure"]
            elif "skipped" in kids:
                oc, el = "skipped", kids["skipped"]
            else:
                oc, el = "passed", None
            msg = "" if el is None else ((el.get("message") or "") + " " + (el.text or ""))[:600]
            cases[tid] = {"outcome": oc, "message": msg}
    except Exception as e:  # noqa: BLE001
        cases = {}
        out += f"\nJUNIT {e}"
    return {"rc": rc, "cases": cases, "tail": out[-3000:]}


def cells(tag):
    try:
        import importlib.util as iu  # noqa: F401
        has_zoo = subprocess.run([a.python, "-c", "import unsloth_zoo"], capture_output = True, env = env_for(), timeout = 600).returncode == 0
    except Exception:  # noqa: BLE001
        has_zoo = False
    res = {"zoo_importable": has_zoo}
    if not has_zoo:
        return res
    for name, pre, extra in (("rank_gt0", "1", {}), ("single", "0", {}),
                             ("rank_gt0_user_hip_expandable", "1", {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"})):
        f = work / f"{tag}_{name}.json"
        try:
            p = subprocess.run([a.python, str(HERE / "pr18287_cell.py"), str(f), pre], cwd = work, capture_output = True,
                               text = True, encoding = "utf-8", errors = "replace", env = env_for(extra), timeout = 900)
            res[name] = json.loads(f.read_text(encoding = "utf-8")) if f.is_file() else {"error": (p.stdout + p.stderr)[-1500:]}
            res[name]["rc"] = p.returncode
        except Exception as e:  # noqa: BLE001
            res[name] = {"error": repr(e)}
    return res


obs["commit"] = git("rev-parse", "HEAD")[1]
for t in TESTS:
    shutil.copy(HERE / "pr18287_tests" / Path(t).name, ck / t)
save()
if a.state == "head":
    obs["pr_module"] = own_module()
    obs["pr_test"] = run_tests("pr")
    obs["pr_cells"] = cells("pr")
    save()
    if not obs["pr_module"].get("has_fix"):
        rc, msg = git("apply", "--whitespace=nowarn", str(HERE / "pr18287_fix.patch"))
        obs["fix_apply"] = {"rc": rc, "msg": msg[-800:]}
obs["module"] = own_module()
obs["test"] = run_tests(a.state)
obs["cells"] = cells(a.state)
save()
