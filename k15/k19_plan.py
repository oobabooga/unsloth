#!/usr/bin/env python3
"""k19_plan.py: the k19-int3 AMD matrix on one gfx1151 host (Linux or Windows), one k15_bench.py call per step, one log per step.

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
STEPS = set((os.environ.get("K19_STEPS") or "archs,ops,exe,exact,bench,tiers,lfm,poly,tbofull").split(","))
HAS = lambda b: os.path.isdir(os.path.join(os.environ["K6P_DIR"], "bins", b))

X = "LLAMA_MOE_CPU_MISS=0;LLAMA_MOE_CACHE_FETCH_OVERLAP=0;GGML_CUDA_VERIFY_REASSOC=0;LLAMA_MOE_ROUTE_TAIL_SKIP=0"
# upstream: no GGML_CUDA_FAKE_FREE_MIB, so a cap cell gives it --fit-target free - N; and its --moe-cache-mib takes an integer only
# (default 0 = no cache; "auto" is a usage error there), so its Studio-argv arms run without the flag: upstream's own default
NF = "K6P_NOFAKE=1;K6P_NO_MOE=1"
ARMS = [
    "i3:i3", "i3b:i3", "up:up:" + NF, "upb:up:" + NF, "pf:pf",
    # exactness arms (pinned); the q36 cache cell adds a fixed 4096 MiB cache
    "xi3:i3:" + X, "xi3b:i3:" + X, "xup:up:%s;%s" % (X, NF), "xpf:pf:" + X,
    "ci3:i3:%s;K6P_MOE=4096" % X, "ci3b:i3:%s;K6P_MOE=4096" % X, "cup:up:%s;K6P_NOFAKE=1;K6P_MOE=4096" % X, "cpf:pf:%s;K6P_MOE=4096" % X,
    # kill switches of the int3 changes that run on this host: fit tiers, host-bound placement, AVX-512 K-quant / IQ3_S dots (Zen 5),
    # the blocking device wait of the host team (the HIP build fix's path), CPU misses
    "i3_p8o:i3:LLAMA_FIT_PLACE8=0", "i3_pmo:i3:LLAMA_FIT_PLACEMID=0", "i3_hbo:i3:LLAMA_FIT_HOSTBOUND=0",
    "i3_kq0:i3:GGML_CPU_KQ_AVX512=0;GGML_CPU_IQ_AVX512=0", "i3_blk0:i3:LLAMA_MOE_SINGLE_WAIT_BLOCKING=0", "i3_cm0:i3:LLAMA_MOE_CPU_MISS=0",
    # test-backend-ops / exe arms
    "t_i3:i3", "t_up:up", "t_pf:pf",
    # r3: the head with no MoE cache (--moe-cache-mib 0, upstream's own default): the 4 GiB rows of r2 (Windows) had the head 35% (q36)
    # / 15% (LFM2.5) below upstream; this arm splits "cache + hybrid gate / CPU-miss path" from the rest of the head
    "i3_nc:i3:K6P_MOE=0",
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
step("devinfo", "devinfo", arms="i3,up,pf")

# 1. tiny-model correctness on the device: every generated arch (dense / MoE / MTP) through test-llama-context, head and base
if "archs" in STEPS:
    for b in ("t_i3", "t_up"):
        step("archs_" + b, "archs", "--bin", b, "--timeout", "1800")
# 2. targeted test-backend-ops on ROCm0 (failing IDs compared head vs base); DFLASH2_CONV exists only in the head
if "ops" in STEPS:
    for b in ("t_i3", "t_up"):
        step("ops_" + b, "tbo", "--bin", b, "--ops", "MUL_MAT_ID;MUL_MAT;FLASH_ATTN_EXT;GATED_DELTA_NET;DFLASH2_CONV;SSM_CONV", "--timeout", "3600")
# 3. the head's new exactness tests on the device (they compare against the CPU / kill switches inside)
if "exe" in STEPS:
    for e in ("test-fa-kvlen-exact", "test-fa-kvlen-rows", "test-moe-bw-exact", "test-dflash2-conv-exact", "test-moe-host-ffn",
              "test-mtp-meta", "test-rs-snap", "test-fit-small-reserve"):
        step("exe_" + e, "exe", "--bin", "t_i3", "--exe", e, "--timeout", "900")
# 4. md5 identity where the int3 changes should be inert on HIP (pinned exact): tiny dense, real MoE full card (with and without MTP),
#    real MoE with a fixed 4096 MiB cache at a forced 8 GiB budget (fit-target, cache + host experts engaged)
if "exact" in STEPS:
    step("exact_q35", "time", "--cell", "q35:full", "--reps", "1", arms="xi3,xi3b,xpf,xup", timeout=1800)
    step("exact_q36_mtp", "time", "--cell", "q36:full", "--reps", "1", arms="xi3,xi3b,xpf,xup", extra=MTP, timeout=3600)
    step("exact_q36_nomtp", "time", "--cell", "q36:full", "--reps", "1", arms="xi3,xpf,xup", timeout=3600)
    step("exact_q36_spill8g", "time", "--cell", "q36:spill8192", "--reps", "1", arms="ci3,ci3b,cpf,cup", extra=MTP, timeout=3600)
# 5. speed, Studio argv, A/A on both sides, rotating order: full card and a forced 8 GiB offload (MoE cache + CPU misses)
if "bench" in STEPS:
    step("bench_q36_full", "time", "--cell", "q36:full", "--reps", "3", arms="up,upb,i3,i3b", extra=MTP, timeout=5400)
    step("bench_q36_spill8g", "time", "--cell", "q36:spill8192", "--reps", "3", arms="up,upb,i3,i3b", extra=MTP, timeout=5400)
    step("bench_q35", "time", "--cell", "q35:full", "--reps", "2", arms="up,i3", timeout=1800)
# 6. --fit tiers on an emulated device (GGML_CUDA_FAKE_FREE_MIB N, strict): 4 GiB small device, 8 GiB k19-place8, 12 / 16 GiB
#    k19-placemid; head, head with the tier's kill switch, pf (k15, no tiers), up (fit-target free - N), Studio argv + MTP
if "tiers" in STEPS:
    step("tier_q36_4g", "time", "--cell", "q36:cap4096", "--reps", "1", arms="i3,i3b,i3_nc,i3_p8o,i3_blk0,pf,up", extra=MTP, p8k=7900, timeout=3600)
    step("tier_q36_8g", "time", "--cell", "q36:cap8192", "--reps", "1", arms="i3,i3b,i3_nc,i3_p8o,i3_kq0,i3_cm0,pf,up", extra=MTP, p8k=7900, timeout=5400)
    step("tier_q36_12g", "time", "--cell", "q36:cap12288", "--reps", "1", arms="i3,i3_pmo,pf,up", extra=MTP, p8k=7900, timeout=3600)
    step("tier_q36_16g", "time", "--cell", "q36:cap16384", "--reps", "1", arms="i3,i3b,i3_pmo,pf,up", extra=MTP, p8k=7900, timeout=3600)
# 7. LFM2.5-24B-A2B (no drafter): host-bound small device + host team (blocking device wait) at 4 GiB, full card speed
if "lfm" in STEPS:
    step("lfm_4g", "time", "--cell", "lfm:cap4096", "--reps", "1", arms="i3,i3b,i3_nc,i3_hbo,i3_blk0,i3_kq0,i3_cm0,pf,up", timeout=3600)
    step("lfm_full", "time", "--cell", "lfm:full", "--reps", "2", arms="up,i3", timeout=2400)
# 8. PolyStrata f472d85a on the same Qwen3.6 GGUF (its defaults), full card
if "poly" in STEPS and HAS("poly"):
    step("poly_q36", "poly", "--model", "q36", "--reps", "2", timeout=3600)
# 9. full test-backend-ops head / base, failing IDs compared (pf only to attribute a head failure)
if "tbofull" in STEPS:
    for b in ("t_i3", "t_up"):
        step("tbofull_" + b, "tbo", "--bin", b, "--ops", "ALL", "--timeout", "7200", timeout=7500)
with open(os.path.join(OUT, "plan.json"), "w", encoding="utf-8") as fh:
    json.dump({"steps": PLAN, "wall_s": round(time.time() - T0), "win": WIN}, fh)
print("=== PLAN DONE %.0f s ===" % (time.time() - T0))
