#!/usr/bin/env python3
"""Probe: Studio's Triton INT8 attention, base vs head, on this AMD GPU (observes only; judged by
criteria/i8ts_timing.py).

Both states: the four INT8 attention test files in the checkout (pytest, junit counts and failing ids).
Head state only: the base and head copies of the kernel files side by side (``i8_base`` / ``i8_head`` packages, the
base taken from the sibling ``base`` checkout), then probes/i8ts/driver.py: acc (both packages in one process: bit
diff, rel L1 and worst-row cosine vs fp64), profile (torch.profiler per kernel), sweep (head configs), cold / warm
first call on an empty TRITON_CACHE_DIR, and ABBA timing rounds (one fresh process per arm). Every ROW goes into the
observation as ``rows``.
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
         "tests/test_diffusion_int8_attention_rocm.py", "tests/test_diffusion_int8_attention_dispatch.py")


def pytest(checkout: Path, python: str, out: Path) -> dict:
    wd = checkout / "studio" / "backend"
    junit = out.with_name(out.stem + ".junit.xml")
    junit.unlink(missing_ok = True)
    cmd = [python, "-m", "pytest", "-q", "-rfEs", "--color=no", "-o", "addopts=", "-p", "no:cacheprovider",
           f"--junitxml={junit}", "-o", "junit_family=xunit1", *TESTS]
    t0 = time.time()
    r = subprocess.run(cmd, cwd = wd, capture_output = True, text = True, timeout = 3600)
    res = {"rc": r.returncode, "seconds": round(time.time() - t0, 1), "tail": (r.stdout + r.stderr)[-4000:],
           "cases": {}}
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
    shutil.copytree(HERE / "i8ts", s)
    for name, tree in (("i8_base", base), ("i8_head", head)):
        (s / name).mkdir()
        for f in FILES:
            shutil.copy2(tree / "studio" / "backend" / "core" / "inference" / f, s / name / f)
        (s / name / "__init__.py").write_text("", encoding = "utf-8")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--steps", default = "acc,profile,sweep,cold,abba:2")
    ap.add_argument("--shapes", default = "")
    ap.add_argument("--sweep-shapes", default = "s1k_h24,s2k_h56,qwen21_1024,ltx25_480p,wan22_5b_480p,wan22_5b_720p,"
                                                "s1k_d64,s2k_d64,ltx25_audio_5s")
    ap.add_argument("--profile-shapes", default = "s1k_h24,qwen21_1024,wan22_5b_720p,wan22_a14b_720p,s1k_d64,"
                                                  "ltx25_audio_5s")
    ap.add_argument("--driver-timeout", type = int, default = 4 * 3600)
    ap.add_argument("--head-extra", default = "", help = "JSON candidate merged into the head's tables (arm i8_headx)")
    a = ap.parse_args()
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
    out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    if a.state != "head":
        return 0
    work = Path(os.environ.get("AMD_CI_WORK") or out.parent) / "i8ts"
    work.mkdir(parents = True, exist_ok = True)
    base = checkout.parent / "base"
    if not (base / "studio").is_dir():
        obs["driver_error"] = f"no base checkout at {base}"
        out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
        return 0
    s = stage(work, base, checkout)
    rows = work / "rows.jsonl"
    rows.unlink(missing_ok = True)
    cmd = [sys.executable, "-u", str(s / "driver.py"), "--python", sys.executable, "--out", str(rows), "--steps",
           a.steps, "--sweep-shapes", a.sweep_shapes, "--profile-shapes", a.profile_shapes, "--work", str(work)]
    if a.shapes:
        cmd += ["--shapes", a.shapes]
    if a.head_extra:
        extra = a.head_extra
        if not extra.lstrip().startswith("{"):
            # k=v,k=v (no JSON quoting on a Windows command line): values parse as JSON, else stay strings
            def val(x):
                try:
                    return json.loads(x)
                except ValueError:
                    return x
            extra = json.dumps({k: val(v) for k, v in (kv.split("=", 1) for kv in extra.split(",") if kv)})
        obs["head_extra"] = extra
        cmd += ["--head-extra", extra]
    log = out.with_name(out.stem + ".driver.log")
    t0 = time.time()
    try:
        with open(log, "w", encoding = "utf-8") as fh:
            obs["driver_rc"] = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT,
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
