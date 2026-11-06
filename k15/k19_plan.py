#!/usr/bin/env python3
"""k19_plan.py (k19-uma copy: arms uma / i3 / pf / up, steps archs, exact, tiers, bench): the k19-int3 AMD matrix on one gfx1151 host (Linux or Windows), one k15_bench.py call per step, one log per step.

Bins (K6P_DIR/bins): i3 = head (k19-int3 eddd38eef9a + HIP build fix 058773ab6c3), up = upstream ff5a2337 (the base: the tree the AMD CI
has built in every llama run so far), pf = k15 7f0111d448c (pf89, the live stack's predecessor; Linux only: md5 / tier reference, and
the base of the cap cells, since upstream has no GGML_CUDA_FAKE_FREE_MIB), poly = PolyStrata f472d85a (Linux only, when it built).
K19_STEPS (comma list, default all): archs, ops, exe, exact, bench, tiers, lfm, poly, tbofull.
Exactness pins X (k15 gate rule + the k19 non-exact defaults): CPU misses off, fetch overlap off, VERIFY_REASSOC off, route tail skip off,
a fixed --moe-cache-mib where a cache runs. Studio argv (default arms): -fa on -np 1 --fit on --moe-cache-mib auto (+ MTP draft-mtp 2).
"""
import json, os, subprocess, sys, time

J = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["K6P_DIR"], "plan_logs")
os.makedirs(OUT, exist_ok=True)
WIN = os.name == "nt"
STEPS = set((os.environ.get("K19_STEPS") or "archs,exact,tiers,bench").split(","))
HAS = lambda b: os.path.isdir(os.path.join(os.environ["K6P_DIR"], "bins", b))

X = "LLAMA_MOE_CPU_MISS=0;LLAMA_MOE_CACHE_FETCH_OVERLAP=0;GGML_CUDA_VERIFY_REASSOC=0;LLAMA_MOE_ROUTE_TAIL_SKIP=0;LLAMA_MTP_EXACT=1"
# upstream: no GGML_CUDA_FAKE_FREE_MIB, so a cap cell gives it --fit-target free - N; and its --moe-cache-mib takes an integer only
# (default 0 = no cache; "auto" is a usage error there), so its Studio-argv arms run without the flag: upstream's own default
NF = "K6P_NOFAKE=1;K6P_NO_MOE=1"
# k19-uma: uma = k19-uma (k19-int3 e18702b7f46 + the small-plan reserve + the unified-memory pricing), i3 = k19-int3 e18702b7f46 (the
# base), pf = k15 7f0111d448c (test-llama-context attribution only), up = upstream ff5a2337 (reference at the capped tiers, 1 launch)
ARMS = [
    "i3:i3", "uma:uma", "up:up:" + NF,
    # exactness arms (pinned, LLAMA_MTP_EXACT=1 GGML_CUDA_VERIFY_REASSOC=0 + the k19 pins); the cache cell adds a fixed 4096 MiB cache
    "xi3:i3:" + X, "xuma:uma:" + X, "ci3:i3:%s;K6P_MOE=4096" % X, "cuma:uma:%s;K6P_MOE=4096" % X,
    # the head's kill switches: CPU misses back on, the unified-memory no-cache rule off, the small-plan reserve off; the base without a cache
    "uma_cm1:uma:LLAMA_MOE_CPU_MISS=1", "uma_mf0:uma:LLAMA_MOE_UMA_MIN_FRAC=0", "uma_sp0:uma:LLAMA_SMALL_PLAN_MAX=0", "i3_nc:i3:K6P_MOE=0",
    "i3_cm0:i3:LLAMA_MOE_CPU_MISS=0",
    # test-llama-context arms
    "t_i3:i3", "t_uma:uma", "t_pf:pf", "t_up:up",
]
MTP = "--spec-type draft-mtp --spec-draft-n-max 2"
PLAN = []
DEADLINE = float(os.environ.get("K15_DEADLINE") or "inf")


def step(name, *args, arms=None, extra="", timeout=None, p8k=0):
    rem = DEADLINE - time.time()
    if rem < 180:
        PLAN.append({"step": name, "rc": "skipped-budget", "s": 0})
        print("STEP %-28s SKIPPED: %.0f s of the job budget left" % (name, rem), flush=True)
        return
    timeout = min(timeout or rem, rem - 60)
    if arms:   # arms whose bin is absent on this host (pf / Windows) are dropped, not failed
        arms = ",".join(x for x in arms.split(",") if HAS(dict(a.split(":")[:2] for a in ARMS)[x]))
    env = dict(os.environ, ARMS=",".join(a for a in ARMS if HAS(a.split(":")[1])), PYTHONIOENCODING="utf-8", K6P_EXTRA=extra, K6P_P8K=str(p8k))
    argv = [sys.executable, "-u", os.path.join(J, "k15_bench.py")] + list(args) + (["--arms", arms] if arms else [])
    t = time.time()
    with open(os.path.join(OUT, name + ".log"), "w", encoding="utf-8") as fh:
        try:
            rc = subprocess.run(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "step-timeout"
            stop_child()
    PLAN.append({"step": name, "rc": rc, "s": round(time.time() - t)})
    print("STEP %-28s rc=%s %5.0f s" % (name, rc, time.time() - t), flush=True)
    with open(os.path.join(OUT, name + ".log"), encoding="utf-8", errors="replace") as fh:
        for l in fh:
            if not l.startswith("ROW") and any(k in l for k in ("_r", "TBO", "EXE", "ARCHS", "POLY", "DEVINFO", "no model")):
                print("   ", l.rstrip()[:300], flush=True)


def stop_child():
    f = os.path.join(os.environ["K6P_DIR"], "child.pid")
    try:
        with open(f, encoding="utf-8") as fh:
            pid = int(fh.read().strip())
    except Exception:  # noqa: BLE001
        return
    if WIN:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
    print("    stopped the overrun step's child %d" % pid, flush=True)


T0 = time.time()
step("dl", "dl", "--models", os.environ.get("K19_DL_MODELS", "q35,q36,lfm"), timeout=7200)
step("devinfo", "devinfo", arms="uma,i3,up")
R3 = os.environ.get("K19_REPS", "3")

# 1. test-llama-context on ROCm0 (GGML_SCHED_DEBUG_REALLOC=1 as the ctest): head, base (aborted in rf on model 1 in r3), k15 (attribution),
#    upstream (reference)
if "archs" in STEPS:
    for b in ("t_uma", "t_i3", "t_pf", "t_up"):
        step("archs_" + b, "archs", "--bin", b, "--timeout", "1800")
# 2. md5 under the exact contract, head vs base (both pinned): every change of the head must be text-inert there
if "exact" in STEPS:
    step("exact_q35", "time", "--cell", "q35:full", "--reps", "1", arms="xi3,xuma", timeout=1800)
    step("exact_q36_mtp", "time", "--cell", "q36:full", "--reps", "1", arms="xi3,xuma", extra=MTP, timeout=3600)
    step("exact_q36_spill8g", "time", "--cell", "q36:spill8192", "--reps", "1", arms="ci3,cuma", extra=MTP, timeout=3600)
    step("exact_lfm_full", "time", "--cell", "lfm:full", "--reps", "1", arms="xi3,xuma", timeout=1800)
# 3. the unified-memory tiers (emulated device, Studio argv + MTP where the model has it), base vs head 3 launches each rotating,
#    upstream and the base without a cache once, the head's kill switches once
if "tiers" in STEPS:
    step("tier_q36_4g", "time", "--cell", "q36:cap4096", "--reps", R3, arms="i3,uma", extra=MTP, p8k=7900, timeout=5400)
    step("tier_q36_4g_ref", "time", "--cell", "q36:cap4096", "--reps", "1", arms="up,i3_nc,uma_mf0", extra=MTP, p8k=7900, timeout=3600)
    step("tier_q36_8g", "time", "--cell", "q36:cap8192", "--reps", R3, arms="i3,uma", extra=MTP, p8k=7900, timeout=5400)
    step("tier_q36_8g_ref", "time", "--cell", "q36:cap8192", "--reps", "1", arms="up,i3_cm0,uma_cm1", extra=MTP, p8k=7900, timeout=3600)
    step("tier_lfm_4g", "time", "--cell", "lfm:cap4096", "--reps", R3, arms="i3,uma", timeout=3600)
    step("tier_lfm_4g_ref", "time", "--cell", "lfm:cap4096", "--reps", "1", arms="up,i3_cm0,uma_cm1", timeout=3600)
    step("tier_q36_16g", "time", "--cell", "q36:cap16384", "--reps", R3, arms="i3,uma", extra=MTP, p8k=7900, timeout=5400)
# 4. the full card (what an uncapped 64-128 GB Strix Halo runs): no MoE cache there, the small-plan reserve engages
if "bench" in STEPS:
    step("bench_q36_full", "time", "--cell", "q36:full", "--reps", R3, arms="i3,uma", extra=MTP, timeout=5400)
    step("bench_lfm_full", "time", "--cell", "lfm:full", "--reps", R3, arms="i3,uma", timeout=3600)
    step("bench_q35_full", "time", "--cell", "q35:full", "--reps", R3, arms="i3,uma,uma_sp0", timeout=1800)
    step("bench_q36_spill8g", "time", "--cell", "q36:spill8192", "--reps", R3, arms="i3,uma", extra=MTP, timeout=5400)
with open(os.path.join(OUT, "plan.json"), "w", encoding="utf-8") as fh:
    json.dump({"steps": PLAN, "wall_s": round(time.time() - T0), "win": WIN}, fh)
print("=== PLAN DONE %.0f s ===" % (time.time() - T0))
