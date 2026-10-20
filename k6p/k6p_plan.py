#!/usr/bin/env python3
"""k6p_plan.py (k9 platforms): the same matrix on Linux and Windows gfx1151 (calls k6p_bench.py once per step, one log per step
under the --out dir). Arms (bins under K6P_DIR/bins): rel = release b11734-mix-c208e5f bundle (its weekly TheRock pin), k8 / k9 =
b11721 + k8 / k9 patch built with the release recipe and pin, k9b = k9 A/A, kill-switch arms k9lm0 (LLAMA_MOE_LAYER_MAJOR=0),
k9df0 (LLAMA_MOE_CACHE_DEVSEED_FALLBACK=0), k9gs0 (GGML_GALLOCR_SPARSE_RESET=0).
Cells: q35 full (dense), gptoss spill6144 (6 GB budget), q3m spill8192 with built-in MTP (--spec-type draft-mtp
--spec-draft-n-max 2): layer-major auto on there in k9."""
import os, subprocess, sys, time
J = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["K6P_DIR"], "plan_logs")
os.makedirs(OUT, exist_ok=True)
ALL = ",".join(["rel:rel", "k8:k8", "k8b:k8", "k9:k9", "k9b:k9", "k9lm0:k9:LLAMA_MOE_LAYER_MAJOR=0",
                "k9df0:k9:LLAMA_MOE_CACHE_DEVSEED_FALLBACK=0", "k9gs0:k9:GGML_GALLOCR_SPARSE_RESET=0"])
MTP = "--spec-type draft-mtp --spec-draft-n-max 2"


def step(name, *args, arms=None, extra=""):
    env = dict(os.environ, ARMS=ALL, PYTHONIOENCODING="utf-8", K6P_EXTRA=extra)
    argv = [sys.executable, "-u", os.path.join(J, "k6p_bench.py")] + list(args) + (["--arms", arms] if arms else [])
    t = time.time()
    with open(os.path.join(OUT, name + ".log"), "w", encoding="utf-8") as fh:
        rc = subprocess.run(argv, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    print("STEP %-28s rc=%s %4.0f s" % (name, rc, time.time() - t), flush=True)
    with open(os.path.join(OUT, name + ".log"), encoding="utf-8", errors="replace") as fh:
        for l in fh:
            if not l.startswith("ROW") and any(k in l for k in ("_r", "KLD", "TBO", "DEVINFO", "fit-target")):
                print("   ", l.rstrip()[:300], flush=True)


step("dl", "dl", "--models", "q35,gptoss,q3m")
step("devinfo", "devinfo", arms="rel,k8,k9")
step("time_q35_full", "time", "--cell", "q35:full", "--reps", "3", arms="rel,k8,k9,k9b")
step("time_q3m_spill8192_mtp", "time", "--cell", "q3m:spill8192", "--reps", "3", arms="rel,k8,k8b,k9,k9b,k9lm0,k9gs0", extra=MTP)
step("time_gptoss_spill6144", "time", "--cell", "gptoss:spill6144", "--reps", "3", arms="rel,k8,k9,k9b,k9df0")
print("=== PLAN DONE ===")
