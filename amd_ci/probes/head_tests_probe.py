#!/usr/bin/env python3
"""Probe: run the HEAD's versions of the selected test files against EVERY state.

Observes only; criteria/head_tests_differential.py judges.

The generic pytest probe skips a test file absent at the base, which is right for a
regression question and useless for a differential one: a PR whose new tests encode
its fix can only show "the base exhibits the defect" if those tests run against the
base's code. So the head's copies of the selected files (plus every non-test helper
and conftest under the same test directories) are overlaid onto each non-head
checkout before pytest runs. Recorded per test id from junit XML, never by count.

Also records, per state, two host observations the change touches (both read from
that state's own code, so base and head are compared on the same host):
  --observe-detection   Studio's GPU detection (utils.hardware) answers
  --observe-h3          MiniMax-H3's host-RAM guard at this host's real free VRAM /
                        available RAM, and the picker fit tiers (head only has them)

Text I/O names utf-8 everywhere: Path.read_text() is cp1252 on the Windows runners.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DETECTION_SCRIPT = r"""
import json, sys
out = {}
def grab(name, fn):
    try:
        out[name] = json.loads(json.dumps(fn(), default=str))
    except BaseException as exc:
        out[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
try:
    from utils import hardware as hw
    grab("detect_hardware", lambda: str(hw.detect_hardware()))
    grab("is_rocm", lambda: bool(getattr(hw.hardware, "IS_ROCM", None)) if hasattr(hw, "hardware") else None)
    grab("gpu_summary", hw.get_gpu_summary)
    grab("backend_visible_gpu_info", hw.get_backend_visible_gpu_info)
    grab("visible_gpu_count", hw.get_visible_gpu_count)
    grab("physical_gpu_count", hw.get_physical_gpu_count)
except BaseException as exc:
    out["import_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(out, indent=2, default=str))
"""

H3_SCRIPT = r"""
import inspect, json, sys
out = {}
try:
    import psutil
    vm = psutil.virtual_memory()
    out["host_total_gb"] = round(vm.total / 1e9, 2)
    out["host_available_gb"] = round(vm.available / 1e9, 2)
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            out["vram_free_gb"] = round(free / 1e9, 2)
            out["vram_total_gb"] = round(total / 1e9, 2)
    except BaseException as exc:
        out["vram_error"] = f"{type(exc).__name__}: {exc}"[:300]
    from core.inference import video_minimax_h3 as v
    from core.inference.video_minimax_h3_te import h3_te_resident_gb
    avail_vram = float(out.get("vram_free_gb") or 0.0)
    # The configuration THIS state's loader would run: a slab-arena pin holds one host copy
    # (transformer_streamed False, as video.py passes it), and the conditioner streams only
    # where the state has that switch on. The base has neither, so both default off/doubled.
    try:
        from core.inference.diffusion_pinned_arena import pin_arena_enabled
        single_copy = bool(pin_arena_enabled())
    except BaseException:
        single_copy = False
    try:
        from core.inference.video_minimax_h3_te import h3_te_stream_enabled
        te_streamed = bool(h3_te_stream_enabled())
    except BaseException:
        te_streamed = False
    out["pin_arena"] = single_copy
    out["te_streamed"] = te_streamed
    kw = dict(
        text_encoder_gb=h3_te_resident_gb("int8", bf16_gb=v.H3_TEXT_ENCODER_BF16_GB),
        transformer_gb=v.h3_transformer_resident_gb("int8"),
        transformer_streamed=not single_copy,
        text_encoder_streamed=te_streamed,
    )
    params = inspect.signature(v.estimate_h3_diffusers_host_ram_gb).parameters
    kw = {k: val for k, val in kw.items() if k in params}
    out["guard_kwargs"] = kw
    out["required_host_gb"] = round(float(v.estimate_h3_diffusers_host_ram_gb(avail_vram, **kw)), 2)
    if hasattr(v, "h3_host_ram_shortfall"):
        out["guard_source"] = "h3_host_ram_shortfall"
        sig = inspect.signature(v.h3_host_ram_shortfall).parameters
        out["refusal"] = v.h3_host_ram_shortfall(avail_vram, **{k: val for k, val in kw.items() if k in sig})
    else:
        # The base's inline guard in video.py, restated: available + process RSS against the floor.
        out["guard_source"] = "inline (base video.py)"
        cap = (psutil.virtual_memory().available + psutil.Process().memory_info().rss) / 1e9
        req = out["required_host_gb"]
        out["refusal"] = (
            f"MiniMax-H3 needs about {req:.0f} GB available system RAM at this VRAM tier; "
            f"{cap:.1f} GB is available. Load the GGUF artifact instead." if cap + 0.5 < req else None)
    out["fit_tiers"] = v.h3_diffusers_fit_tiers() if hasattr(v, "h3_diffusers_fit_tiers") else "absent at this state"
    out["ok"] = True
except BaseException as exc:
    import traceback
    out["ok"] = False
    out["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    out["tail"] = traceback.format_exc().strip().splitlines()[-6:]
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(out, indent=2, default=str))
"""


def overlay(head_dir: Path, workdir: Path, tests: list[str]) -> dict:
    """Copy the head's selected test files, and every non-test helper / conftest in their
    directories, onto this checkout. Returns what was added and what was replaced."""
    added, replaced, missing = [], [], []
    files: set[Path] = set()
    dirs: set[Path] = set()
    for t in tests:
        rel = Path(t.split("::")[0])
        files.add(rel)
        dirs.add(rel.parent)
    for d in dirs:
        src_dir = head_dir / d
        if not src_dir.is_dir():
            continue
        for p in src_dir.iterdir():
            if p.is_file() and p.suffix == ".py" and not p.name.startswith("test_"):
                files.add(d / p.name)
    for rel in sorted(files):
        src, dst = head_dir / rel, workdir / rel
        if not src.is_file():
            missing.append(str(rel))
            continue
        if dst.is_file():
            if dst.read_bytes() == src.read_bytes():
                continue
            replaced.append(str(rel))
        else:
            added.append(str(rel))
        dst.parent.mkdir(parents = True, exist_ok = True)
        shutil.copyfile(src, dst)
    return {"added": added, "replaced": replaced, "missing_at_head": missing}


def parse_junit(path: Path) -> dict:
    res = {"passed": [], "failed": [], "errors": [], "skipped": [], "messages": {}}
    if not path.is_file():
        return res
    root = ET.fromstring(path.read_text(encoding = "utf-8"))
    for tc in root.iter("testcase"):
        cls = tc.get("classname") or ""
        name = tc.get("name") or ""
        tid = f"{cls}::{name}" if cls else name
        outcome = "passed"
        msg = None
        for child in tc:
            if child.tag == "failure":
                outcome, msg = "failed", child.get("message")
            elif child.tag == "error":
                outcome, msg = "errors", child.get("message")
            elif child.tag == "skipped":
                outcome, msg = "skipped", child.get("message")
        res[outcome].append(tid)
        if msg and outcome != "passed":
            res["messages"][tid] = msg[:300]
    return res


def run_script(python: str, script: str, workdir: Path, out_file: Path, timeout: int = 300) -> dict:
    try:
        p = subprocess.run([python, "-c", script, str(out_file)], cwd = workdir,
                           capture_output = True, text = True, timeout = timeout,
                           env = {**os.environ, "PYTHONPATH": str(workdir)})
        rec: dict = {"rc": p.returncode}
        if out_file.is_file():
            rec.update(json.loads(out_file.read_text(encoding = "utf-8")))
        else:
            rec["stderr_tail"] = (p.stderr or "")[-1500:]
        return rec
    except subprocess.TimeoutExpired:
        return {"rc": -1, "error": "TimeoutExpired"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--subdir", default = "studio/backend")
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 1800)
    ap.add_argument("--tests-from", default = None,
                    help = "checkout whose test files are run everywhere (default: the sibling 'head' state)")
    ap.add_argument("--observe-detection", action = "store_true")
    ap.add_argument("--observe-h3", action = "store_true")
    ap.add_argument("--tests", nargs = "+", required = True)
    args = ap.parse_args()
    # Absolute: pytest and the helper scripts run with cwd = the checkout.
    args.out = args.out.resolve()

    checkout = Path(args.checkout)
    head_root = Path(args.tests_from) if args.tests_from else checkout.parent / "head"
    workdir = checkout / args.subdir
    head_dir = head_root / args.subdir
    obs: dict = {"state": args.state, "workdir": str(workdir), "tests": args.tests,
                 "tests_from": str(head_root)}

    def write() -> None:
        args.out.parent.mkdir(parents = True, exist_ok = True)
        args.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    if not workdir.is_dir() or not head_dir.is_dir():
        obs["error"] = f"missing directory: {workdir if not workdir.is_dir() else head_dir}"
        write()
        return 0

    is_head = workdir.resolve() == head_dir.resolve()
    obs["overlay"] = {"added": [], "replaced": [], "missing_at_head": []} if is_head \
        else overlay(head_dir, workdir, args.tests)
    obs["is_head_source"] = is_head
    present = [t for t in args.tests if (workdir / t.split("::")[0]).exists()]
    obs["selected"] = present
    write()

    if args.observe_detection:
        obs["detection"] = run_script(args.python, DETECTION_SCRIPT, workdir,
                                      args.out.parent / f"detection_{args.state}.json")
        write()
    if args.observe_h3:
        obs["h3_guard"] = run_script(args.python, H3_SCRIPT, workdir,
                                     args.out.parent / f"h3_{args.state}.json")
        write()

    # The Linux GPU job's Studio venv carries no pytest (only the suites job installs it).
    # Installed here, once, recorded; never silently assumed.
    if subprocess.run([args.python, "-c", "import pytest, pytest_asyncio"], capture_output = True).returncode != 0:
        pkgs = ["pytest", "pytest-asyncio", "pytest-timeout"]
        r = subprocess.run([args.python, "-m", "pip", "install", "-q", *pkgs], capture_output = True, text = True)
        if r.returncode != 0:
            subprocess.run([args.python, "-m", "ensurepip", "-q"], capture_output = True)
            r = subprocess.run([args.python, "-m", "pip", "install", "-q", *pkgs], capture_output = True, text = True)
        obs["pytest_install"] = {"rc": r.returncode, "stderr_tail": (r.stderr or "")[-500:]}
        write()
    have_timeout = subprocess.run([args.python, "-c", "import pytest_timeout"],
                                  capture_output = True).returncode == 0
    junit = args.out.parent / f"junit_{args.state}.xml"
    cmd = [args.python, "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider",
           "--continue-on-collection-errors", f"--junitxml={junit}"]
    if have_timeout:
        cmd += ["--timeout", "600"]
    cmd += present
    obs["cmd"] = " ".join(cmd)
    env = {**os.environ, "UNSLOTH_SETTLE_DELAY_S": "0", "PYTHONIOENCODING": "utf-8"}
    try:
        p = subprocess.run(cmd, cwd = workdir, capture_output = True, text = True,
                           encoding = "utf-8", errors = "replace",
                           timeout = args.timeout, env = env)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc = -1
        out = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        err = "TimeoutExpired"
    obs["rc"] = rc
    obs["tail"] = out[-6000:]
    obs["stderr_tail"] = err[-2000:]
    res = parse_junit(junit)
    obs.update({k: sorted(v) if isinstance(v, list) else v for k, v in res.items()})
    obs["n_passed"] = len(res["passed"])
    obs["n_failed"] = len(res["failed"])
    obs["n_errors"] = len(res["errors"])
    obs["n_skipped"] = len(res["skipped"])
    write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
