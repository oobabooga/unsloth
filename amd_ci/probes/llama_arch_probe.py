#!/usr/bin/env python3
"""Probe: KV preemption across architectures on this GPU, for one state.

Observes only. Builds llama-server (HIP, else Vulkan) for the llama.cpp tree the state maps to
(--map state=url@sha, cached by sha), generates random-weight fixtures with the head tree's
test-llama-archs once (shared by every state, so every state serves identical files), fetches
three small real GGUFs, and runs arch_stress.py cells. base runs only the arms a stock server has.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = [("llama", "llama-dense"), ("qwen3", "qwen3-dense"), ("qwen35", "qwen35-dense"),
            ("qwen35moe", "qwen35moe-moe"), ("gpt-oss", "gpt-oss-moe"), ("lfm2moe", "lfm2moe-moe"),
            ("deepseek2", "deepseek2-moe"), ("deepseek4", "deepseek4-moe"), ("gemma3", "gemma3-dense")]
REAL = [("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
        ("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q8_0.gguf"),
        ("unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-UD-IQ2_M.gguf")]


def sh(cmd, cwd=None, timeout=3600, env=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def backend_args():
    hip = "/opt/rocm/bin/hipcc" if os.path.exists("/opt/rocm/bin/hipcc") else shutil.which("hipcc")
    if hip:
        args = ["-DGGML_HIP=ON", "-DGPU_TARGETS=gfx1151", "-DAMDGPU_TARGETS=gfx1151"]
        if os.path.exists("/opt/rocm/llvm/bin/clang"):
            args += ["-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang", "-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++",
                     "-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++"]
        return "hip", args
    if shutil.which("glslc"):
        return "vulkan", ["-DGGML_VULKAN=ON"]
    return None, None


def build(url, sha, root, targets, obs):
    src = root / f"src_{sha[:12]}"
    bld = src / "build"
    obs["backend"] = backend_args()[0]
    if all((bld / "bin" / t).exists() for t in targets):
        obs.setdefault("build_cached", []).append(sha[:12])
        return bld / "bin"
    if not src.exists():
        src.mkdir(parents=True)
        for cmd in (["git", "init", "-q"], ["git", "fetch", "-q", "--depth", "1", url, sha], ["git", "checkout", "-q", "FETCH_HEAD"]):
            for attempt in range(4):
                rc, out = sh(cmd, cwd=src, timeout=1800)
                if not rc:
                    break
                time.sleep(15 * (attempt + 1))
            if rc:
                obs["clone_error"] = out[-1500:]
                return None
    name, args = backend_args()
    if not name:
        obs["error"] = "no GPU toolchain (hipcc or glslc) on this runner"
        return None
    obs["backend"] = name
    rc, out = sh(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", *args], cwd=src, timeout=1800)
    if rc:
        obs["configure_error"] = out[-1500:]
        return None
    t0 = time.time()
    rc, out = sh(["cmake", "--build", "build", "-j", str(os.cpu_count() or 8), "--target", *targets], cwd=src, timeout=7200)
    obs.setdefault("build_seconds", {})[sha[:12]] = round(time.time() - t0)
    if rc or not all((bld / "bin" / t).exists() for t in targets):
        obs["build_error"] = out[-2500:]
        return None
    return bld / "bin"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--map", action="append", default=[])
    ap.add_argument("--root", default=os.environ.get("AMD_CI_WORK", os.getcwd()))
    args = ap.parse_args()
    root = Path(args.root) / "arch_probe"
    root.mkdir(parents=True, exist_ok=True)
    obs = {"state": args.state}
    mapping = dict(m.split("=", 1) for m in args.map)
    url, sha = (mapping.get(args.state) or mapping["head"]).rsplit("@", 1)
    obs["llama_ref"] = sha

    def done():
        args.out.write_text(json.dumps(obs, indent=2), encoding="utf-8")
        return 0

    # merge maps to the same sha as head: the binary and the fixtures are identical, so reuse head's runs
    head_runs = args.out.parent / "runs_head.jsonl"
    if args.state != "head" and mapping.get("head", "").endswith("@" + sha) and head_runs.exists():
        prev = json.loads((args.out.parent / "obs_head.json").read_text(encoding="utf-8"))
        obs.update({k: v for k, v in prev.items() if k != "state"}, reused_from="head")
        return done()

    bindir = build(url, sha, root, ["llama-server"], obs)
    if not bindir:
        return done()
    exe = bindir / "llama-server"
    env = dict(os.environ, LD_LIBRARY_PATH=str(bindir))
    obs["has_preempt_ram"] = "--preempt-ram" in sh([str(exe), "--help"], env=env)[1]
    devices = sh([str(exe), "--list-devices"], env=env)[1]
    obs["devices"] = devices[-800:]
    gpu = next((d for d in ("ROCm0", "Vulkan0", "CUDA0") if d in devices), None)
    obs["device"] = gpu

    # fixtures: always from the head tree's generator, so every state serves the same files
    fixdir = root / "fixtures"
    if not all((fixdir / f"{n}.gguf").exists() for _, n in FIXTURES):
        hurl, hsha = mapping["head"].rsplit("@", 1)
        hbin = build(hurl, hsha, root, ["llama-server", "test-llama-archs"], obs)
        if not hbin:
            obs["fixture_error"] = "head generator did not build"
            return done()
        fixdir.mkdir(exist_ok=True)
        for arch, _ in FIXTURES:
            rc, out = sh([str(hbin / "test-llama-archs"), "-a", f"^{arch}$", "-s", "2981215097", "-o", str(fixdir)],
                         env=dict(os.environ, LD_LIBRARY_PATH=str(hbin)), timeout=900)
            if rc:
                obs.setdefault("fixture_errors", {})[arch] = out[-500:]
    models = [(n, fixdir / f"{n}.gguf", 256, 64) for _, n in FIXTURES]
    mdir = root / "models"
    mdir.mkdir(exist_ok=True)
    for repo, f in REAL:
        dst = mdir / f
        if not dst.exists():
            try:
                tmp = dst.with_suffix(".part")
                urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/main/{f}", tmp)
                tmp.rename(dst)
            except Exception as e:  # noqa: BLE001
                obs.setdefault("model_errors", {})[f] = repr(e)[:300]
                continue
        models.append((f.replace(".gguf", ""), dst, 4096, 512))

    if args.state == "base":
        cells = [("S1", "base", "fcfs"), ("S3", "base", "fcfs")]
    else:
        cells = [("S1", "swap", "fcfs"), ("S1", "swap", "cost"), ("S1", "recomp", "fcfs"), ("S3", "off", "fcfs"),
                 ("S3", "swap", "fcfs"), ("S5", "swap", "fcfs"), ("S8", "swap", "fcfs"), ("S10", "swap", "fcfs"),
                 ("S10", "recomp", "fcfs"), ("S4", "swap", "fcfs")]
    runs = args.out.parent / f"runs_{args.state}.jsonl"
    if runs.exists():
        runs.unlink()
    obs["cells"] = []
    for tag, path, ctx, ub in models:
        if not path.exists():
            continue
        for scen, arm, pol in cells:
            cmd = [sys.executable, str(HERE / "arch_stress.py"), "--bin", str(exe), "--model", str(path), "--scenario", scen,
                   "--arm", arm, "--policy", pol, "--ctx", str(ctx), "--np", "4", "--ngl", "99", "--threads", "4",
                   "--batch", str(max(256, ub)), "--ubatch", str(ub), "--tag", tag, "--logdir", str(args.out.parent / f"logs_{args.state}"),
                   "--out", str(runs), "--duration", "30", "--start-timeout", "900"]
            if gpu:
                cmd += ["--", "--device", gpu]
            t0 = time.time()
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
                rc = p.returncode
                tail = (p.stdout + p.stderr)[-300:] if rc else ""
            except subprocess.TimeoutExpired:
                rc, tail = -9, "timeout"
            obs["cells"].append({"model": tag, "scenario": scen, "arm": arm, "policy": pol, "rc": rc, "s": round(time.time() - t0), "tail": tail})
    obs["runs"] = [json.loads(l) for l in runs.read_text(encoding="utf-8").splitlines() if l.strip()] if runs.exists() else []
    return done()


if __name__ == "__main__":
    raise SystemExit(main())
