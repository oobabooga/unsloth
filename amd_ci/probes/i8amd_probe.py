#!/usr/bin/env python3
"""Probe: Studio's Triton INT8 attention on gfx1151, base vs head, against comfy-kitchen's HIP INT8 kernel (observes
only; judged by criteria/i8amd_timing.py).

Both states: the INT8 attention test files in the checkout (pytest, junit ids), then (with --e2e-spec) one end-to-end
render through diffusion_bench/amd/diffusion_probe.py (Studio in process, native vs INT8 attention, LPIPS vs native of
the same state; the model is fetched once, token-free, into a dir both states share).
Head state only: the base and head kernel files side by side (``i8_base`` / ``i8_head``; ``i8_nospec`` = head with the
sequence lengths out of the attention kernel's compile key, as on NVIDIA), comfy-kitchen installed --no-deps into a
private dir, then probes/i8amd/driver.py: acc (bits base vs head, error vs fp64 for base / head / ck), variants
(compiled attention variants over every shape + the device AMDGCN of each), ck:N (rounds of base, head, ck, ck, head,
base, one fresh process per arm), nospec:N, rocprof (kernel trace when rocprofv3 exists). ROW lines -> ``rows``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ("diffusion_int8_attention.py", "diffusion_int8_attention_cuda.py", "diffusion_int8_attention_triton.py",
         "diffusion_triton_prefetch.py", "rocm_bf16.py")
TESTS = ("tests/test_diffusion_int8_attention_triton_fallback.py", "tests/test_diffusion_int8_attention.py",
         "tests/test_diffusion_int8_attention_rocm.py", "tests/test_diffusion_int8_attention_dispatch.py",
         "tests/test_diffusion_int8_attention_triton_compile_key.py",
         "tests/test_diffusion_int8_attention_triton_device.py",
         "tests/test_diffusion_int8_attention_triton_offsets.py", "tests/test_diffusion_int8_attention_cache.py",
         "tests/test_diffusion_int8_attention_long_probe.py")
CK = ["comfy-kitchen==0.2.41"]
NOSPEC_FROM = '["n_heads"] if triton is not None and _rocm_build() else ["sq", "sk", "n_heads"]'
NOSPEC_TO = '["sq", "sk", "n_heads"]'
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")


def pytest(checkout: Path, python: str, out: Path) -> dict:
    wd = checkout / "studio" / "backend"
    junit = out.with_name(out.stem + ".junit.xml")
    junit.unlink(missing_ok = True)
    tests = [t for t in TESTS if (wd / t).is_file()]
    cmd = [python, "-m", "pytest", "-q", "-rfEs", "--color=no", "-o", "addopts=", "-p", "no:cacheprovider",
           f"--junitxml={junit}", "-o", "junit_family=xunit1", *tests]
    t0 = time.time()
    r = subprocess.run(cmd, cwd = wd, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                       timeout = 3600)
    res = {"rc": r.returncode, "seconds": round(time.time() - t0, 1), "files": tests,
           "tail": (r.stdout + r.stderr)[-4000:], "cases": {}}
    if junit.is_file():
        for tc in ET.parse(junit).getroot().iter("testcase"):
            name = f"{tc.get('file') or tc.get('classname')}::{tc.get('name')}"
            kind = "passed"
            for child in tc:
                if child.tag in ("failure", "error"):
                    kind = "failed"
                elif child.tag == "skipped":
                    kind = "skipped"
            res["cases"][name] = kind
    for kind in ("passed", "failed", "skipped"):
        res[kind] = sum(v == kind for v in res["cases"].values())
    return res


def stage(work: Path, base: Path, head: Path) -> Path:
    s = work / "stage"
    shutil.rmtree(s, ignore_errors = True)
    shutil.copytree(HERE / "i8amd", s)
    for name, tree in (("i8_base", base), ("i8_head", head), ("i8_nospec", head)):
        (s / name).mkdir()
        for f in FILES:
            shutil.copy2(tree / "studio" / "backend" / "core" / "inference" / f, s / name / f)
        (s / name / "__init__.py").write_text("", encoding = "utf-8")
    t = s / "i8_nospec" / "diffusion_int8_attention_triton.py"
    text = t.read_text(encoding = "utf-8")
    if NOSPEC_FROM not in text:
        raise RuntimeError("the head's compile-key line moved; i8_nospec cannot be built")
    t.write_text(text.replace(NOSPEC_FROM, NOSPEC_TO), encoding = "utf-8")
    return s


def install_ck(pkg_dir: Path, log: list) -> str | None:
    if (pkg_dir / ".done").is_file():
        return None
    pkg_dir.mkdir(parents = True, exist_ok = True)
    for cmd in ([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(pkg_dir),
                 "--index-url", "https://pypi.org/simple", *CK],
                ["uv", "pip", "install", "-q", "--no-deps", "--target", str(pkg_dir), "--python", sys.executable,
                 "--index-url", "https://pypi.org/simple", *CK]):
        try:
            r = subprocess.run(cmd, capture_output = True, text = True, timeout = 1200)
        except Exception as exc:  # noqa: BLE001
            log.append({"cmd": " ".join(cmd), "error": repr(exc)[:300]})
            continue
        log.append({"cmd": " ".join(cmd), "rc": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]})
        if r.returncode == 0:
            (pkg_dir / ".done").write_text("ok", encoding = "utf-8")
            return None
    return "comfy-kitchen install failed"


def e2e(a, checkout: Path, out: Path) -> dict:
    """One diffusion_bench pass over the spec's cells for this state (its own process; tokens never reach it)."""
    root = Path(os.environ.get("GITHUB_WORKSPACE") or HERE.parent.parent)
    probe = root / "diffusion_bench" / "amd" / "diffusion_probe.py"
    work = Path(os.environ.get("AMD_CI_WORK") or out.parent) / "i8amd_e2e"
    obs_path = out.with_name(out.stem + ".e2e.json")
    cmd = [sys.executable, "-u", str(probe), "--state", a.state, "--checkout", str(checkout), "--out", str(obs_path),
           "--spec", str(root / a.e2e_spec), "--work", str(work / a.state), "--models-dir", str(work / "models"),
           "--budget-min", str(a.e2e_budget_min), "--install", "auto"]
    for pat in a.e2e_only.split(","):
        cmd += ["--only", pat]
    # CI=true makes torch's custom-op aliasing check an error on the first compiled call (resources README section
    # 10 / 2): measurement arms run with it unset, as a user's process does
    env = {k: v for k, v in os.environ.items() if k not in TOKEN_VARS and k != "CI"}
    log = out.with_name(out.stem + ".e2e.log")
    t0 = time.time()
    try:
        with open(log, "w", encoding = "utf-8") as fh:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env,
                                timeout = int(a.e2e_budget_min * 60 + 3600)).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    try:
        found = json.loads(obs_path.read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        found = {"error": f"no observation: {type(exc).__name__}: {exc}"[:300]}
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "obs": found}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--steps", default = "acc,variants,ck:2,nospec:2,rocprof")
    ap.add_argument("--shapes", default = "")
    ap.add_argument("--long-shapes", default = "")
    ap.add_argument("--head-extra", default = "", help = "k=v,k=v or JSON merged into the head's tables (arm i8_headx)")
    ap.add_argument("--configs", default = "", help = "sweep table file name inside probes/i8amd")
    ap.add_argument("--sweep-shapes", default = "wan22_a14b_720p,wan22_5b_720p,qwen21_2048")
    ap.add_argument("--driver-timeout", type = int, default = 4 * 3600)
    ap.add_argument("--e2e-spec", default = "", help = "diffusion_bench spec, relative to the branch root")
    ap.add_argument("--e2e-only", default = "i8e_*")
    ap.add_argument("--e2e-budget-min", type = float, default = 60.0)
    a = ap.parse_args()
    for k in TOKEN_VARS:
        os.environ.pop(k, None)
    out = Path(a.out).resolve()
    checkout = Path(a.checkout).resolve()
    obs: dict = {"state": a.state, "python": sys.executable}
    try:
        import torch
        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        obs["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        obs["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if obs["gpu"] else None
        import triton
        obs["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)[:500]
    obs["pytest"] = pytest(checkout, sys.executable, out)
    if a.state == "head":
        # the base ran first (differential.py probes base, then head): its ids let the criteria name new failures
        try:
            obs["base_pytest_cases"] = json.loads(out.with_name("obs_base.json").read_text(encoding = "utf-8"))[
                "pytest"]["cases"]
        except Exception as exc:  # noqa: BLE001
            obs["base_pytest_cases_error"] = repr(exc)[:300]
    out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    if a.e2e_spec:
        obs["e2e"] = e2e(a, checkout, out)
        out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    if a.state != "head":
        return 0
    work = out.parent / "i8amd_work"
    work.mkdir(parents = True, exist_ok = True)
    base = checkout.parent / "base"
    if not (base / "studio").is_dir():
        obs["driver_error"] = f"no base checkout at {base}"
        out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
        return 0
    try:
        s = stage(work, base, checkout)
    except Exception as exc:  # noqa: BLE001
        obs["driver_error"] = f"stage: {exc}"
        out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
        return 0
    obs["ck_log"] = []
    pkg_dir = Path(os.environ.get("RUNNER_TEMP") or work) / "i8amd_ck"
    obs["ck_install_error"] = install_ck(pkg_dir, obs["ck_log"])
    env = dict(os.environ, PYTHONPATH = os.pathsep.join([str(pkg_dir), os.environ.get("PYTHONPATH", "")]).rstrip(
        os.pathsep))
    env.pop("CI", None)  # measurement arms as a user's process runs them (see e2e)
    rows = work / "rows.jsonl"
    rows.unlink(missing_ok = True)
    cmd = [sys.executable, "-u", str(s / "driver.py"), "--python", sys.executable, "--out", str(rows), "--steps",
           a.steps, "--work", str(work)]
    if a.shapes:
        cmd += ["--shapes", a.shapes]
    if a.long_shapes:
        cmd += ["--long-shapes", a.long_shapes]
    if a.configs:
        cmd += ["--configs", str(s / a.configs)]
    cmd += ["--sweep-shapes", a.sweep_shapes]
    if a.head_extra:
        extra = a.head_extra
        if not extra.lstrip().startswith("{"):
            # k=v,k=v (no JSON quoting on a Windows command line); a dotted key nests: long.bm=256
            def val(x):
                try:
                    return json.loads(x)
                except ValueError:
                    return x
            d: dict = {}
            for kv in (x for x in extra.split(",") if x):
                k, v = kv.split("=", 1)
                *path, leaf = k.split(".")
                t = d
                for part in path:
                    t = t.setdefault(part, {})
                t[leaf] = val(v)
            extra = json.dumps(d)
        obs["head_extra"] = extra
        cmd += ["--head-extra", extra]
    log = out.with_name(out.stem + ".driver.log")
    t0 = time.time()
    try:
        with open(log, "w", encoding = "utf-8") as fh:
            obs["driver_rc"] = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env,
                                              timeout = a.driver_timeout).returncode
    except subprocess.TimeoutExpired:
        obs["driver_rc"] = "timeout"
    obs["driver_seconds"] = round(time.time() - t0, 1)
    obs["rows"] = [json.loads(x[4:]) for x in rows.read_text(encoding = "utf-8").splitlines()
                   if x.startswith("ROW ")] if rows.is_file() else []
    out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
