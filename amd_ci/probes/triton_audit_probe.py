#!/usr/bin/env python3
"""Triton kernel performance audit probe (unsloth + unsloth_zoo kernels), one state. Observes only.

Runs the audit bench suite (probes/triton_audit_bench/run_all.py, one fresh process per module) against the
state's unsloth checkout, with unsloth_zoo and upstream fla cloned at pinned SHAs. Writes the bench rows
(jsonl) next to --out and a summary JSON to --out. Never judges.
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "triton_audit_bench"
ZOO_SHA = "21e66eb539353e84a3c26c92cdae5a784680fbb6"
FLA_SHA = "9d478facb282410dc150051d5f57205cd3812f31"
KERNEL_FILES = ["unsloth/kernels/" + f for f in (
    "cross_entropy_loss.py", "rope_embedding.py", "rms_layernorm.py", "layernorm.py", "swiglu.py", "geglu.py",
    "qk_norm_rope.py", "large_head_attention.py", "nf4.py", "nf4_gemv.py", "int4_packed.py", "fp8.py", "nvfp4.py",
    "utils.py", "moe/grouped_gemm/kernels/forward.py", "moe/grouped_gemm/kernels/backward.py")]


def _clone(url, sha, dest):
    if not (dest / ".git").is_dir():
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", url, str(dest)], check = True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", sha], check = True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--reps", type = int, default = 1)
    ap.add_argument("--dtypes", default = "bf16")
    ap.add_argument("--quick", action = "store_true")
    ap.add_argument("--timeout", type = int, default = 900)
    a = ap.parse_args()
    out_dir = a.out.parent
    work = Path(os.environ.get("AMD_CI_WORK") or os.environ.get("RUNNER_TEMP") or out_dir) / "triton_audit_deps"
    work.mkdir(parents = True, exist_ok = True)
    obs = {"state": a.state, "checkout": a.checkout, "python": a.python}
    try:
        _clone("https://github.com/unslothai/unsloth-zoo", ZOO_SHA, work / "unsloth-zoo")
        _clone("https://github.com/fla-org/flash-linear-attention", FLA_SHA, work / "fla_latest")
        obs["deps"] = {"zoo": ZOO_SHA, "fla": FLA_SHA}
    except Exception as exc:  # recorded, judged by criteria
        obs["deps_error"] = repr(exc)
    pip = subprocess.run([a.python, "-m", "pip", "install", "-q", "einops", "requests", "packaging"],
                         capture_output = True, text = True)
    obs["pip_rc"] = pip.returncode
    probe = subprocess.run([a.python, "-c", "import json,torch,triton;print(json.dumps({'torch':torch.__version__,"
                            "'hip':torch.version.hip,'triton':triton.__version__,'gpu':torch.cuda.get_device_name(0),"
                            "'arch':torch.cuda.get_device_properties(0).gcnArchName}))"], capture_output = True, text = True)
    try:
        obs["env"] = json.loads(probe.stdout.strip().splitlines()[-1])
    except Exception:
        obs["env_error"] = (probe.stdout + probe.stderr)[-1500:]
    obs["kernel_sha1"] = {}
    for f in KERNEL_FILES:
        p = Path(a.checkout) / f
        obs["kernel_sha1"][f] = hashlib.sha1(p.read_bytes()).hexdigest() if p.is_file() else None
    rows = out_dir / f"triton_audit_rows_{a.state}.jsonl"
    env = dict(os.environ, AUDIT_UNSLOTH_SRC = a.checkout, AUDIT_ZOO_SRC = str(work / "unsloth-zoo"),
               AUDIT_FLA_LATEST = str(work / "fla_latest"), PYTHONUNBUFFERED = "1",
               TORCHINDUCTOR_CACHE_DIR = str(work / f"inductor_{a.state}"),
               TRITON_CACHE_DIR = str(work / f"triton_{a.state}"))
    cmd = [a.python, "-u", str(BENCH / "run_all.py"), "--out", str(rows), "--reps", str(a.reps),
           "--dtypes", a.dtypes, "--timeout", str(a.timeout), "--tag", "rocm_" + a.state]
    if a.quick:
        cmd.append("--quick")
    t0 = time.time()
    p = subprocess.run(cmd, env = env, capture_output = True, text = True)
    obs["bench_rc"] = p.returncode
    obs["bench_secs"] = time.time() - t0
    (out_dir / f"triton_audit_{a.state}.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding = "utf-8")
    kinds, per = {}, {}
    if rows.is_file():
        for line in rows.read_text(encoding = "utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
            if r.get("kind") == "time" and r.get("arm") == "ours":
                per[f"{r.get('bench')}|{r.get('case')}|{r.get('dtype')}|{r.get('phase')}"] = r.get("ms")
    obs["row_kinds"] = kinds
    obs["ours_ms"] = per
    obs["rows_file"] = str(rows)
    a.out.write_text(json.dumps(obs), encoding = "utf-8")


if __name__ == "__main__":
    main()
