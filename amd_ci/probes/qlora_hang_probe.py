#!/usr/bin/env python3
"""Probe for issue unslothai/unsloth#11498: does a long QLoRA run hang or fault the GPU?

Observes only; criteria/qlora_hang.py judges. One state maps to one ARM through
--arm-map (issue mode: every state is the same checkout, the arm differs):

  main_default  Unsloth from the checkout, default env (vendored fla on)
                + a fla_stress pass when the training arm stayed healthy
  main_nofla    same with UNSLOTH_DISABLE_VENDORED_FLA=1
  peft          Transformers + bnb NF4 + PEFT, no unsloth import
  hist          the reporter's release, from $AMD_CI_HIST_SITE prepended to PYTHONPATH

Each arm runs qlora_hang_worker.py in its own process group under a heartbeat
watchdog: differential.py has no per-probe timeout, so a hung GPU would otherwise
hold the job until the workflow timeout. An arm result is cached in --cache-dir so
a second differential over the same arm reuses it instead of training again.
Once a GPU reset or an unusable GPU is seen, later arms are not run (status
"skipped"), because they would measure a recovering device.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = HERE / "qlora_hang_worker.py"

FAULT_PATTERNS = [
    r"Memory access fault", r"HW Exception", r"GPU Hang", r"hipErrorIllegalAddress",
    r"illegal memory access", r"hipErrorLaunchFailure", r"HSA_STATUS_ERROR", r"Page not present",
    r"hipErrorNoDevice", r"unspecified launch failure", r"device-side assert",
]
DMESG_PATTERNS = [r"amdgpu", r"gfxhub", r"MES", r"ring .* timeout", r"GPU reset", r"VRAM is lost"]

ARMS = {
    "main_default": ("unsloth", {}),
    "main_nofla": ("unsloth", {"UNSLOTH_DISABLE_VENDORED_FLA": "1"}),
    "peft": ("peft", {}),
    "hist": ("unsloth", {"__HIST__": "1"}),
}


def read_beats(path: Path) -> list[dict]:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding = "utf-8", errors = "replace").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    return out


def kernel_log() -> dict:
    """amdgpu kernel lines, if this unprivileged runner may read them at all."""
    for cmd in (["dmesg", "-T"], ["journalctl", "-k", "-n", "2000", "--no-pager"]):
        try:
            r = subprocess.run(cmd, capture_output = True, text = True, timeout = 30)
        except Exception:  # noqa: BLE001
            continue
        if r.returncode == 0 and r.stdout:
            lines = [ln for ln in r.stdout.splitlines() if any(re.search(p, ln) for p in DMESG_PATTERNS)]
            return {"readable": True, "source": cmd[0], "amdgpu_lines": lines[-120:]}
    return {"readable": False}


def gpu_healthy(python: str, env: dict) -> dict:
    code = ("import torch; x = torch.ones(1024, device='cuda'); y = (x @ x).item(); "
            "torch.cuda.synchronize(); print('OK', y)")
    try:
        r = subprocess.run([python, "-c", code], capture_output = True, text = True,
                           timeout = 180, env = env)
        ok = r.returncode == 0 and "OK" in r.stdout
        return {"ok": ok, "rc": r.returncode, "tail": (r.stdout + r.stderr)[-600:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "rc": None, "tail": "health check timed out after 180 s"}


def run_worker(python, worker_arm, env, wd: Path, tag, args) -> dict:
    hb = wd / f"heartbeat_{tag}.jsonl"
    log = wd / f"worker_{tag}.log"
    hb.unlink(missing_ok = True)
    cmd = [python, "-u", str(WORKER), "--arm", worker_arm, "--heartbeat", str(hb),
           "--model", args.model, "--max-steps", str(args.max_steps), "--ga", str(args.ga),
           "--seq", str(args.seq), "--minutes", str(args.stress_minutes if worker_arm == "fla_stress" else args.minutes)]
    t0 = time.time()
    with open(log, "wb") as fh:
        proc = subprocess.Popen(cmd, stdout = fh, stderr = subprocess.STDOUT, env = env,
                                start_new_session = True)
        status, last_change, n_seen, first_micro = None, time.time(), 0, False
        while True:
            try:
                rc = proc.wait(timeout = 10)
                break
            except subprocess.TimeoutExpired:
                pass
            beats = read_beats(hb)
            if len(beats) != n_seen:
                n_seen, last_change = len(beats), time.time()
                first_micro = first_micro or any(b.get("phase") == "micro" for b in beats)
            # Before the first micro-step: download, load, compile, autotune. After: steady state.
            limit = args.stall_s if first_micro else args.warmup_s
            if time.time() - last_change > limit:
                status = "hang"
                break
            if time.time() - t0 > args.hard_s:
                status = "hard_timeout"
                break
        if status is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
            try:
                rc = proc.wait(timeout = 120)
            except subprocess.TimeoutExpired:
                rc = None  # stuck in the driver (D state): itself evidence of a wedged GPU
    elapsed = time.time() - t0
    beats = read_beats(hb)
    text = log.read_text(encoding = "utf-8", errors = "replace")
    faults = sorted({m.group(0) for p in FAULT_PATTERNS for m in re.finditer(p, text)})
    micro = [b for b in beats if b.get("phase") == "micro"]
    steps = [b for b in beats if b.get("phase") == "step"]
    engagement = next((b["engagement"] for b in reversed(beats) if b.get("phase") == "engagement"), None)
    loaded = next((b for b in beats if b.get("phase") == "loaded"), {})
    done = any(b.get("phase") == "done" for b in beats)
    losses = [b.get("loss") for b in micro if b.get("loss") is not None]
    nonfinite = any(isinstance(x, float) and not (x == x and abs(x) != float("inf")) for x in losses) \
        or any(b.get("finite") is False for b in micro)
    if status is None:
        status = "ok" if (rc == 0 and done) else ("fault" if faults else "crash")
    elif faults:
        status = f"{status}+fault"
    return {
        "worker_arm": worker_arm, "status": status, "rc": rc, "elapsed_s": round(elapsed, 1),
        "micro_steps": len(micro), "optimizer_steps": len(steps),
        "last_beat": beats[-1] if beats else None,
        "losses_step": [round(b["loss"], 4) for b in steps if b.get("loss") is not None][:5]
        + ["..."] + [round(b["loss"], 4) for b in steps if b.get("loss") is not None][-5:],
        "grad_norms_step": [round(b.get("grad_norm", 0), 4) for b in steps][-5:],
        "peak_gib": max([b.get("peak_gib", 0) for b in steps] or [0]),
        "max_len_seen": max([b.get("len", 0) for b in micro] or [0]),
        "nonfinite": nonfinite, "fault_lines": faults,
        "log_tail": text[-4000:], "versions": loaded.get("versions"), "engagement": engagement,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--arm-map", required = True, help = "state=arm,state=arm")
    ap.add_argument("--cache-dir", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--model", default = "unsloth/Qwen3.5-2B")
    ap.add_argument("--max-steps", type = int, default = 120)
    ap.add_argument("--ga", type = int, default = 16)
    ap.add_argument("--seq", type = int, default = 1024)
    ap.add_argument("--minutes", type = float, default = 35)
    ap.add_argument("--stress-minutes", type = float, default = 10)
    ap.add_argument("--warmup-s", type = int, default = 1500)
    ap.add_argument("--stall-s", type = int, default = 300)
    ap.add_argument("--hard-s", type = int, default = 3600)
    args = ap.parse_args()

    arm = dict(kv.split("=", 1) for kv in args.arm_map.split(","))[args.state]
    worker_arm, extra_env = ARMS[arm]
    args.cache_dir.mkdir(parents = True, exist_ok = True)
    cached = args.cache_dir / f"arm_{arm}.json"
    reset_marker = args.cache_dir / "GPU_UNHEALTHY"
    obs: dict = {"state": args.state, "arm": arm}

    if cached.is_file():
        obs.update(json.loads(cached.read_text(encoding = "utf-8")))
        obs["reused"] = True
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0
    if reset_marker.is_file():
        obs.update({"status": "skipped", "why": reset_marker.read_text(encoding = "utf-8")})
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    env = dict(os.environ)
    env.update({k: v for k, v in extra_env.items() if not k.startswith("__")})
    env["UNSLOTH_COMPILE_LOCATION"] = str(args.cache_dir / f"compiled_{arm}")
    if "__HIST__" in extra_env:
        site = os.environ.get("AMD_CI_HIST_SITE", "")
        if not site or not Path(site).is_dir():
            obs.update({"status": "error", "why": f"AMD_CI_HIST_SITE missing: {site!r}"})
            args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
            return 0
        env["PYTHONPATH"] = site + os.pathsep + env.get("PYTHONPATH", "")
    obs["env_delta"] = {k: env[k] for k in ("UNSLOTH_DISABLE_VENDORED_FLA", "PYTHONPATH") if k in env}

    obs["kernel_log_before"] = kernel_log()
    before = set(obs["kernel_log_before"].get("amdgpu_lines", []))
    wd = args.cache_dir
    obs.update(run_worker(args.python, worker_arm, env, wd, arm, args))
    obs["health_after"] = gpu_healthy(args.python, env)
    if arm == "main_default" and obs["status"] == "ok" and obs["health_after"]["ok"]:
        obs["fla_stress"] = run_worker(args.python, "fla_stress", env, wd, "fla_stress", args)
        obs["fla_stress"].pop("log_tail", None) if obs["fla_stress"]["status"] == "ok" else None
        obs["health_after"] = gpu_healthy(args.python, env)
    after = kernel_log()
    obs["kernel_log_after"] = {**after, "new_amdgpu_lines": [ln for ln in after.get("amdgpu_lines", []) if ln not in before]} \
        if after.get("readable") else after
    if not obs["health_after"]["ok"]:
        reset_marker.write_text(f"GPU unusable after arm {arm}: {obs['health_after']['tail'][-300:]}",
                                encoding = "utf-8")

    cached.write_text(json.dumps({k: v for k, v in obs.items() if k not in ("state",)}, indent = 2),
                      encoding = "utf-8")
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
