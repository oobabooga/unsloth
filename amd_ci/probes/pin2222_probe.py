#!/usr/bin/env python3
"""Probe: real pinned host memory for unsloth-zoo PR 2222 (counted pinned cap), at one state.

OBSERVES only (verdicts live in criteria/pin2222_cap.py). Runs on the runner's plain python3
(--no-studio) and builds its own venv once per job: ROCm torch >= 2.13 from download.pytorch.org
(repo.amd.com's gfx1151 index stops at 2.11, below the PR's counted mode), plus the few packages
`import unsloth_zoo` needs. `unsloth` itself is a stub package so the zoo's presence check passes;
gradient_checkpointing does not import it.

Per arm and scenario a fresh process runs probes/pin2222/pin_sim.py with the arm's checkout first
on PYTHONPATH, so `allocated_bytes.peak` from torch.cuda.host_memory_stats() covers that scenario
only. `--extra-arms` (head state only) adds intermediate PR commits extracted with git archive from
the same clone, so base, the intermediates and head run on the same host and torch.

Also records whether torch honours the 2.13+ host allocator knobs here (round threshold, max cached
size), measured behaviourally, so the criteria can call the run INCONCLUSIVE when they are not.
Result JSON goes to --out, never stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN_SIM = HERE / "pin2222" / "pin_sim.py"
TORCH = "2.14.1+rocm7.14"
ROCM_INDEX = "https://download.pytorch.org/whl/rocm7.14"
DEPS = ["transformers==5.21.0", "accelerate", "peft", "psutil", "filelock", "packaging", "numpy"]
CAP_MIB = {"1": 10, "1b": 10, "2": 10, "3": 11}
CONF = {"1": "pinned_max_cached_size_mb:4", "1b": "pinned_max_cached_size_mb:4",
        "2": "pinned_max_round_threshold_mb:1", "3": "pinned_max_round_threshold_mb:1"}

KNOB = r"""
import json, os, sys, torch
MIB = 1 << 20
mode = sys.argv[1]
res = {"mode": mode, "conf": os.environ.get("PYTORCH_ALLOC_CONF"), "torch": torch.__version__,
       "hip": torch.version.hip, "cuda_available": torch.cuda.is_available()}
g = getattr(torch._C, "_accelerator_getAllocatorSettings", None)
res["has_getter"] = g is not None
try: res["settings"] = g() if g else None
except Exception as e: res["settings"] = f"error: {type(e).__name__}: {e}"
try:
    res["arch"] = torch.cuda.get_device_properties(0).gcnArchName
    res["arch_list"] = torch.cuda.get_arch_list()
    def cur(): return int(torch.cuda.host_memory_stats()["allocated_bytes.current"])
    if mode in ("round_default", "round_1"):
        t = torch.empty(3 * MIB, dtype = torch.uint8, pin_memory = True)
        res["pinned"] = bool(t.is_pinned()); res["held_after_3mib"] = cur()
    else:
        t = torch.empty(6 * MIB, dtype = torch.uint8, pin_memory = True)
        res["pinned"] = bool(t.is_pinned()); res["held_live_6mib"] = cur()
        del t; torch.cuda.synchronize()
        u = torch.empty(1, dtype = torch.uint8, pin_memory = True); del u
        res["held_after_free_6mib"] = cur()
    res["stats_keys"] = sorted(k for k in torch.cuda.host_memory_stats() if k.startswith("allocated_bytes"))
except Exception as e:
    res["error"] = f"{type(e).__name__}: {str(e)[:800]}"
print("KNOB " + json.dumps(res), flush = True)
"""


def sh(cmd: list[str], log: list[str], timeout: int = 1800, env: dict | None = None) -> int:
    t0 = time.time()
    p = subprocess.run(cmd, capture_output = True, text = True, timeout = timeout, env = env)
    log.append(f"$ {' '.join(cmd)}  (rc {p.returncode}, {time.time() - t0:.0f}s)\n{p.stdout[-1500:]}{p.stderr[-2500:]}")
    print(log[-1], flush = True)
    return p.returncode


def ensure_venv(root: Path, log: list[str]) -> tuple[Path | None, str]:
    """One venv per job, shared by every state's probe call (they run serially)."""
    v = root / "pin2222_venv"
    py = v / "bin" / "python"
    done = v / ".ready"
    if done.is_file():
        return py, "reused"
    uv = shutil.which("uv")
    if uv:
        rc = sh([uv, "venv", "-q", str(v), "--python", "3.12"], log)
        if rc:
            return None, "uv venv failed"
        env = dict(os.environ)
        for k in ("UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_EXTRA_INDEX_URL", "UV_EXCLUDE_NEWER"):
            env.pop(k, None)
        rc = sh([uv, "pip", "install", "-q", "--python", str(py), f"torch=={TORCH}", *DEPS,
                 "--index-url", "https://pypi.org/simple", "--extra-index-url", ROCM_INDEX,
                 "--index-strategy", "unsafe-best-match"], log, env = env)
    else:
        rc = sh([sys.executable, "-m", "venv", str(v)], log)
        if rc:
            return None, "python -m venv failed"
        rc = sh([str(py), "-m", "pip", "install", "-q", f"torch=={TORCH}", "--index-url", ROCM_INDEX], log)
        if not rc:
            rc = sh([str(py), "-m", "pip", "install", "-q", *DEPS, "--index-url", "https://pypi.org/simple"], log)
    if rc:
        return None, f"install failed rc {rc}"
    done.write_text("ok\n", encoding = "utf-8")
    return py, "built"


def child_env(arm_dir: str | None, stub: Path, conf: str | None, cap_mib: float | None, work: Path) -> dict:
    env = dict(os.environ)
    for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTHONPATH"):
        env.pop(k, None)
    if conf: env["PYTORCH_ALLOC_CONF"] = conf
    if cap_mib is not None: env["UNSLOTH_GC_PINNED_MAX_GB"] = str(cap_mib / 1024)
    if arm_dir: env["PYTHONPATH"] = f"{arm_dir}{os.pathsep}{stub}"
    env["UNSLOTH_IS_PRESENT"] = "1"
    env["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
    env["UNSLOTH_COMPILE_LOCATION"] = str(work / "ucc")
    env["HF_HUB_OFFLINE"] = "1"
    return env


def tagged(stdout: str, tag: str) -> dict | None:
    line = next((l for l in stdout.splitlines() if l.startswith(tag + " ")), None)
    return json.loads(line[len(tag) + 1:]) if line else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--extra-arms", default = "", help = "comma list name=sha, run only at the head state")
    ap.add_argument("--scenarios", default = "1,1b,2,3")
    ap.add_argument("--timeout", type = int, default = 600)
    args = ap.parse_args()

    root = Path(os.environ.get("AMD_CI_WORK") or tempfile.gettempdir()).resolve()
    work = root / "pin2222"
    work.mkdir(parents = True, exist_ok = True)
    stub = work / "stub"
    (stub / "unsloth").mkdir(parents = True, exist_ok = True)
    (stub / "unsloth" / "__init__.py").write_text("", encoding = "utf-8")

    log: list[str] = []
    obs: dict = {"state": args.state, "checkout": args.checkout}
    head_sha = subprocess.run(["git", "-C", args.checkout, "rev-parse", "HEAD"],
                              capture_output = True, text = True).stdout.strip()
    obs["sha"] = head_sha
    try:
        py, how = ensure_venv(root, log)
    except Exception as e:  # noqa: BLE001
        py, how = None, f"{type(e).__name__}: {e}"
    obs["venv"] = how
    if py is None:
        obs["setup_error"] = how
        obs["setup_log"] = "\n".join(log)[-6000:]
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    # Host facts: do the 2.13+ knobs and host_memory_stats work on this ROCm torch?
    knobs = {}
    for mode, conf in (("round_default", None), ("round_1", "pinned_max_round_threshold_mb:1"),
                       ("cached_default", None), ("cached_4", "pinned_max_cached_size_mb:4")):
        p = subprocess.run([str(py), "-c", KNOB, mode], capture_output = True, text = True, timeout = 300,
                           env = child_env(None, stub, conf, None, work), cwd = str(work))
        knobs[mode] = tagged(p.stdout, "KNOB") or {"error": f"rc {p.returncode}: {p.stderr[-1500:]}"}
    obs["knobs"] = knobs

    arms = {args.state: args.checkout}
    if args.state == "head" and args.extra_arms:
        for item in args.extra_arms.split(","):
            name, _, sha = item.partition("=")
            dest = work / "arms" / name
            if not (dest / "unsloth_zoo").is_dir():
                dest.mkdir(parents = True, exist_ok = True)
                ar = subprocess.run(["git", "-C", args.checkout, "archive", sha, "unsloth_zoo"], capture_output = True)
                if ar.returncode:
                    obs.setdefault("arm_errors", {})[name] = ar.stderr.decode(errors = "replace")[-800:]
                    continue
                subprocess.run(["tar", "-x", "-C", str(dest)], input = ar.stdout, check = True)
            arms[name] = str(dest)
    obs["arms"] = arms

    results = []
    for arm, arm_dir in arms.items():
        for s in args.scenarios.split(","):
            env = child_env(arm_dir, stub, CONF[s], CAP_MIB[s], work)
            t0 = time.time()
            try:
                p = subprocess.run([str(py), str(PIN_SIM), s, arm], capture_output = True, text = True,
                                   timeout = args.timeout, env = env, cwd = str(work))
                res = tagged(p.stdout, "RESULT")
                if res is None:
                    res = {"scenario": s, "arm": arm, "error": f"rc {p.returncode}: {p.stderr[-2500:]}"}
                res["stderr_tail"] = p.stderr[-1200:]
            except subprocess.TimeoutExpired:
                res = {"scenario": s, "arm": arm, "error": f"timeout {args.timeout}s"}
            res["arm_dir"] = arm_dir
            res["seconds"] = round(time.time() - t0, 1)
            results.append(res)
            print(f"{arm:>12} S{s:<2} peak {res.get('peak_mib')} cap {res.get('cap_mib')} "
                  f"over {res.get('over_cap')} dec {res.get('decisions')} err {str(res.get('error', ''))[:200]}",
                  flush = True)
    obs["results"] = results
    obs["setup_log"] = "\n".join(log)[-3000:]
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
