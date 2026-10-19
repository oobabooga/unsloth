#!/usr/bin/env python3
"""k6p_plan.py: the run-4 matrix, identical on Linux and Windows (calls k6p_bench.py once per step, one log per step under
K6P_DIR/../out or --out). Arms (bins under K6P_DIR/bins): rel / rel2 = release bundle (A/A), rs = the release source built with the
same toolchain as k6, ub = plain upstream b11721 (no --moe-cache-mib), k6 / k6b (A/A), k6nf = k6 with GGML_CUDA_DISABLE_FUSION=1.
Spill cells add k6 kill-switch arms (fix 2 lend rule off / k5 ubatch rule / prefetch depth 2) and rs at -ub 4096."""
import os, subprocess, sys, time
J = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["K6P_DIR"], "plan_logs")
TBO_OPS = sys.argv[2] if len(sys.argv) > 2 else ""
os.makedirs(OUT, exist_ok=True)
BASE = ["rel:rel", "rel2:rel", "rs:rs", "ub:ub:K6P_NO_MOE=1", "k6:k6", "k6b:k6", "k6nf:k6:GGML_CUDA_DISABLE_FUSION=1"]
SPILLX = ["k6l0:k6:LLAMA_MOE_PREFILL_UB_SHRINK_LEND=0", "k6k5:k6:LLAMA_MOE_PREFILL_UB_SHRINK=1", "k6d2:k6:LLAMA_MOE_PREFETCH_DEPTH_SPILL=2",
          "rsu4:rs:K6P_ARGS=-ub 4096", "rsu2:rs:K6P_ARGS=-ub 2048"]
ALL = ",".join(BASE + SPILLX)


def step(name, *args, arms=None):
    env = dict(os.environ, ARMS=ALL, PYTHONIOENCODING="utf-8")
    argv = [sys.executable, "-u", os.path.join(J, "k6p_bench.py")] + list(args) + (["--arms", arms] if arms else [])
    t = time.time()
    with open(os.path.join(OUT, name + ".log"), "w", encoding="utf-8") as fh:
        rc = subprocess.run(argv, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    print("STEP %-28s rc=%s %4.0f s" % (name, rc, time.time() - t), flush=True)
    with open(os.path.join(OUT, name + ".log"), encoding="utf-8", errors="replace") as fh:
        for l in fh:
            if not l.startswith("ROW") and any(k in l for k in ("_r", "KLD", "TBO", "DEVINFO", "fit-target")):
                print("   ", l.rstrip()[:260], flush=True)


B = "rel,rel2,rs,ub,k6,k6b,k6nf"
SX = B + ",k6l0,k6k5,k6d2,rsu4"
step("dl", "dl", "--models", "q35,lfm2,gptoss")
step("devinfo", "devinfo", arms="rel,rs,ub,k6")
step("time_q35_full", "time", "--cell", "q35:full", "--reps", "3", arms=B)
step("time_lfm2_full", "time", "--cell", "lfm2:full", "--reps", "3", arms=B)
step("time_lfm2_spill1536", "time", "--cell", "lfm2:spill1536", "--reps", "3", arms=SX)
step("time_gptoss_full", "time", "--cell", "gptoss:full", "--reps", "3", arms=B)
step("time_gptoss_spill6144", "time", "--cell", "gptoss:spill6144", "--reps", "3", arms=SX)
# KLD, reference = the release bundle all on the device (ub 512); then the same-toolchain pair with rs as the reference
step("kld_q35", "kld", "--model", "q35", "--modes", "full", arms="rel,rel2,rs,ub,k6,k6nf")
step("kld_q35_ub1", "kld", "--model", "q35", "--modes", "full_ub1", arms="rel,rs,k6")
step("kld_lfm2", "kld", "--model", "lfm2", "--modes", "full,spill1536", arms="rel,rel2,rs,ub,k6,k6nf")
step("kld_lfm2_rs", "kld", "--model", "lfm2", "--modes", "full,spill1536,full_ub1", arms="rs,k6")
step("kld_gptoss", "kld", "--model", "gptoss", "--modes", "full,spill6144", arms="rel,rel2,rs,ub,k6")
step("kld_gptoss_spillx", "kld", "--model", "gptoss", "--modes", "spill6144", arms="rel,rsu2,k6l0,k6k5,k6d2")
if TBO_OPS:
    step("tbo", "tbo", "--bin", "k6", "--ops", TBO_OPS, "--timeout", "2400")
print("=== PLAN DONE ===")
