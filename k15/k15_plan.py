#!/usr/bin/env python3
"""k15_plan.py: the k15 AMD matrix on one gfx1151 host (Linux or Windows), one k15_bench.py call per step, one log per step.

Bins (K6P_DIR/bins): k15 = upstream ff5a2337 + k10.patch + k15d.patch (1b1b9369adf), k10 = ff5a2337 + k10.patch (a162dce854, the last
head that ran on AMD), up = stock ff5a2337 (upstream merge base; Linux only, used only to attribute a full test-backend-ops failure).
Pins for every exactness arm (k15 gate rule): LLAMA_MOE_CPU_MISS=0, LLAMA_MOE_CACHE_FETCH_OVERLAP=0, GGML_CUDA_VERIFY_REASSOC=0 and a
fixed --moe-cache-mib (never auto) except where the test list asks for --moe-cache-mib 0.
K15_STEPS (comma list, default all): ops, gst, mmid, eager, gdn, race, mailbox, replay, ple, dguard, bench, tbofull, tbopad, stock.
"""
import json, os, subprocess, sys, time

J = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["K6P_DIR"], "plan_logs")
os.makedirs(OUT, exist_ok=True)
WIN = os.name == "nt"
STEPS = set((os.environ.get("K15_STEPS") or "ops,gst,mmid,eager,gdn,race,mailbox,replay,ple,dguard,bench,tbofull,tbopad,stock").split(","))

PIN = "LLAMA_MOE_CPU_MISS=0;LLAMA_MOE_CACHE_FETCH_OVERLAP=0;GGML_CUDA_VERIFY_REASSOC=0"
C1 = "K6P_MOE=1024"                     # pinned MoE cache for q3m (covers the 2-token MTP window + tail; rule: never auto / 0)
FAULT = "GGML_CUDA_GRAPH_UPDATE_FAULT=7;GGML_CUDA_GRAPH_INSTANTIATE_FAULT=1"
KS5 = ("GGML_CUDA_DISABLE_CONV_CAT_FUSION=1;GGML_CUDA_DISABLE_QSA_POOL_FUSION=1;GGML_CUDA_DISABLE_QSA_KEY_PACK_FUSION=1;"
       "GGML_CUDA_DISABLE_UNARY_SRC_CONT_FUSION=1;GGML_CUDA_QSA_SEL_GRID=0")
ARMS = [
    "k15:k15", "k15b:k15", "k10:k10", "k10b:k10", "up:up",
    # test-backend-ops switch arms
    "k15_cc:k15:GGML_CUDA_DISABLE_CONV_CAT_FUSION=1", "k15_qp:k15:GGML_CUDA_DISABLE_QSA_POOL_FUSION=1",
    "k15_qk:k15:GGML_CUDA_DISABLE_QSA_KEY_PACK_FUSION=1", "k15_gc:k15:GGML_CUDA_DISABLE_UNARY_SRC_CONT_FUSION=1",
    "k15_sg0:k15:GGML_CUDA_QSA_SEL_GRID=0", "k15_ks5:k15:" + KS5,
    "k15_ip0:k15:GGML_CUDA_GDN_STATE_IN_PLACE=0", "k15_gg0:k15:GGML_CUDA_GDN_GATES=0",
    "k15_q80:k15:GGML_CUDA_GATED_Q8_1=0", "k15_glu1:k15:GGML_CUDA_DISABLE_GLU_Q8_1=1",
    "k15_mk0:k15:GGML_CPU_MMID_MULTI_K=0",
    "k15_p0:k15:K6P_PAD=0", "k15_p37:k15:K6P_PAD=37", "k15_p4096:k15:K6P_PAD=4096",
    "k10_p0:k10:K6P_PAD=0", "k10_p37:k10:K6P_PAD=37", "k10_p4096:k10:K6P_PAD=4096",
    "k15_gst:k15:GGML_CUDA_GRAPH_WHY=40", "k15_gstp:k15:GGML_CUDA_GRAPH_WHY=40;K6P_PAD=4096",
    # exactness arms, pinned (q3m arms add C1 below; q35 is dense, no cache)
    "e_g:k15:%s;GGML_CUDA_DISABLE_GRAPHS=1" % PIN, "e_d:k15:%s" % PIN, "e_d2:k15:%s" % PIN,
    "e_f1:k15:%s;%s" % (PIN, FAULT), "e_f2:k15:%s;%s;GGML_CUDA_GRAPH_DESTROY_FAULT=1" % (PIN, FAULT),
    "e_fks:k15:%s;%s;GGML_CUDA_GRAPH_EAGER_FALLBACK=0" % (PIN, FAULT),
    "m_g:k15:%s;%s;GGML_CUDA_DISABLE_GRAPHS=1" % (PIN, C1), "m_d:k15:%s;%s" % (PIN, C1), "m_d2:k15:%s;%s" % (PIN, C1),
    "m_f1:k15:%s;%s;%s" % (PIN, C1, FAULT), "m_f2:k15:%s;%s;%s;GGML_CUDA_GRAPH_DESTROY_FAULT=1" % (PIN, C1, FAULT),
    "m_fks:k15:%s;%s;%s;GGML_CUDA_GRAPH_EAGER_FALLBACK=0" % (PIN, C1, FAULT),
    "m_a0:k15:%s;%s;GGML_CUDA_ADD_RMS_FUSE=0" % (PIN, C1), "m_a02:k15:%s;%s;GGML_CUDA_ADD_RMS_FUSE=0" % (PIN, C1),
    "m_ip0:k15:%s;%s;GGML_CUDA_GDN_STATE_IN_PLACE=0" % (PIN, C1), "m_dbg:k15:%s;%s;GGML_CUDA_GDN_STATE_IN_PLACE_DEBUG=1" % (PIN, C1),
    # spilled tier, --moe-cache-mib 0 (the GDN warm-up crash path); k10 at the same flags = reference only
    "n_d:k15:%s;K6P_MOE=0" % PIN, "n_d2:k15:%s;K6P_MOE=0" % PIN, "n_ip0:k15:%s;K6P_MOE=0;GGML_CUDA_GDN_STATE_IN_PLACE=0" % PIN,
    "n_dbg:k15:%s;K6P_MOE=0;GGML_CUDA_GDN_STATE_IN_PLACE_DEBUG=1" % PIN, "n_k10:k10:%s;K6P_MOE=0" % PIN,
    # r2 (1b1b9369adf): mailbox handshake fast path / one-region host experts kill switches at the m_ pins (default = m_d / m_d2), and
    # LLAMA_MOE_SINGLE=1 forced: HIP has no mapped mailbox (ggml_backend_moe_mailbox_alloc is CUDA only), so the split path must stay
    "m_mbf0:k15:%s;%s;GGML_MOE_MAILBOX_FAST=0" % (PIN, C1), "m_sf0:k15:%s;%s;LLAMA_MOE_SINGLE_FUSED=0" % (PIN, C1),
    "m_s1:k15:%s;%s;LLAMA_MOE_SINGLE=1" % (PIN, C1),
    # r2: replay-commit fallback (1811f0c7468) at a 4 GB budget + MTP: default x2, every commit graph forced through the reserved fallback
    # buffer, and forced with the reserve off (the old behaviour: a commit that cannot allocate aborts; reference only)
    "r4_d:k15:%s" % PIN, "r4_d2:k15:%s" % PIN, "r4_ff:k15:%s;LLAMA_RS_REPLAY_FALLBACK_FORCE=1" % PIN,
    "r4_fb0:k15:%s;LLAMA_RS_REPLAY_FALLBACK_FORCE=1;LLAMA_RS_REPLAY_FALLBACK=0" % PIN,
    # r2: PLE background read-in, gemma-4 E2B with a lazy PLE table: default (Linux: made resident; Windows: left lazy) vs LLAMA_PLE_RESIDENT=0
    "ple_d:k15:K6P_ARGS=--lazy-mode on", "ple_0:k15:K6P_ARGS=--lazy-mode on;LLAMA_PLE_RESIDENT=0", "ple_d2:k15:K6P_ARGS=--lazy-mode on",
    # draft guard: the default k15 MTP launches must keep the draft; a forced floor shows the check runs on HIP (non-vacuity)
    "dg_force:k15:LLAMA_DRAFT_MIN_FREE_MIB=1000000",
]
MTP = "--spec-type draft-mtp --spec-draft-n-max 2"
PLAN = []


DEADLINE = float(os.environ.get("K15_DEADLINE") or "inf")   # epoch seconds: the job's step budget, set by linux_gpu.sh / win_gpu.ps1


def step(name, *args, arms=None, extra="", timeout=None):
    rem = DEADLINE - time.time()
    if rem < 180:
        PLAN.append({"step": name, "rc": "skipped-budget", "s": 0})
        print("STEP %-28s SKIPPED: %.0f s of the job budget left" % (name, rem), flush=True)
        return
    timeout = min(timeout or rem, rem - 60)
    env = dict(os.environ, ARMS=",".join(ARMS), PYTHONIOENCODING="utf-8", K6P_EXTRA=extra, K6P_P8K="0")
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
            if not l.startswith("ROW") and any(k in l for k in ("_r", "TBO", "EXE", "DEVINFO", "no model")):
                print("   ", l.rstrip()[:300], flush=True)


def stop_child():
    """a step killed on its budget leaves k15_bench.py's server / test process (own session) running: stop it by its recorded pid"""
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


def rows():
    p = os.path.join(os.environ.get("K6P_OUT") or os.path.join(os.environ["K6P_DIR"], "out"), "rows.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


T0 = time.time()
step("dl", "dl", "--models", "q35,q3m,g4e2b", timeout=7200)
step("devinfo", "devinfo", arms="k15,k10" + ("" if WIN else ",up"))

# 2. fusion cases: default, each disable switch, all five; GDN (default / in-place off / gates off, + k10's GATED_DELTA_NET);
#    GATED_MMVQ / GATED_Q8_1 (default / GATED_Q8_1=0 / DISABLE_GLU_Q8_1=1). New cases exist only in k15's test-backend-ops,
#    so their base arm is k15 with the switch (the pre-fusion path); k10 runs the cases it has.
if "ops" in STEPS:
    FUS = "CONV_CAT_STATE;QSA_POOL_KEYS;QSA_KEY_PACK;GATE_CONT;QSA_SEL_FA"
    for b in ("k15", "k15_cc", "k15_qp", "k15_qk", "k15_gc", "k15_sg0", "k15_ks5"):
        step("ops_fus_" + b, "tbo", "--bin", b, "--ops", FUS, "--timeout", "1800")
    for b in ("k15", "k15_ip0", "k15_gg0", "k10"):
        step("ops_gdn_" + b, "tbo", "--bin", b, "--ops", "GATED_DELTA_NET;", "--timeout", "1800")
    for b in ("k15", "k15_q80", "k15_glu1"):
        step("ops_gq_" + b, "tbo", "--bin", b, "--ops", "GATED_MMVQ;GATED_Q8_1", "--timeout", "1800")
# test-graph-src-type (k15 only: the test is new), plain and with the 4096-byte environment pad
if "gst" in STEPS:
    for b in ("k15_gst", "k15_gstp"):
        step("gst_" + b, "exe", "--bin", b, "--exe", "test-graph-src-type", "--timeout", "900")
# 7. MUL_MAT_ID at environment pads 0 / 37 / 4096, k15 vs k10; 9 (GPU side). k15 with GGML_CPU_MMID_MULTI_K=0 (the CPU reference
#    side of the comparison on the stock loop)
if "mmid" in STEPS:
    for b in ("k15_p0", "k15_p37", "k15_p4096", "k10_p0", "k10_p37", "k10_p4096", "k15_mk0"):
        step("mmid_" + b, "tbo", "--bin", b, "--ops", "MUL_MAT_ID;", "--timeout", "3600")
# 3. eager fallback on HIP graphs: graphs off / default x2 / faults / faults + destroy fault / faults with the fallback off (the old
#    behaviour: must abort, else the fault never reached the path), dense GDN q35 full and MoE GDN q3m 8 GB + MTP, pinned cache
if "eager" in STEPS:
    step("eager_q35", "time", "--cell", "q35:full", "--reps", "1", arms="e_g,e_d,e_d2,e_f1,e_f2,e_fks", timeout=3600)
# 3 + 5 + 4 (pinned cache): one q3m matrix, every arm the same flags but its switch
if STEPS & {"eager", "race", "gdn", "mailbox"}:
    step("exact_q3m", "time", "--cell", "q3m:spill8192", "--reps", "1",
         arms="m_g,m_d,m_d2,m_f1,m_f2,m_fks,m_a0,m_a02,m_ip0,m_dbg,m_mbf0,m_sf0,m_s1", extra=MTP, timeout=7200)
# r2: replay-commit fallback at a 4 GB-like budget (spill4096: --fit-target = free - 4096 MiB), MTP, shipping cache size
if "replay" in STEPS:
    step("replay_q3m_4g", "time", "--cell", "q3m:spill4096", "--reps", "1", arms="r4_d,r4_d2,r4_ff,r4_fb0", extra=MTP, timeout=3600)
# r2: PLE read-in on (Linux) / off by construction (Windows), same text as LLAMA_PLE_RESIDENT=0
if "ple" in STEPS:
    step("ple_g4e2b", "time", "--cell", "g4e2b:full", "--reps", "1", arms="ple_d,ple_0,ple_d2", timeout=1800)
# 4 + 8: --moe-cache-mib 0, in-place off vs default, kept-gather count with the debug log, k10 reference
if "gdn" in STEPS:
    step("gdn_nc_q3m", "time", "--cell", "q3m:spill8192", "--reps", "1", arms="n_d,n_d2,n_ip0,n_dbg,n_k10", extra=MTP, timeout=5400)
# 10. draft guard: forced floor (the k15 default MTP launches of every q3m step are the "draft kept" evidence)
if "dguard" in STEPS:
    step("dguard_q3m", "time", "--cell", "q3m:spill8192", "--reps", "1", arms="dg_force", extra=MTP, timeout=1800)
# 11. no-regression bench (before the full suites: the shortest item that needs the exclusive card), A/A (k10 / k10b, k15 / k15b), rotating order, unpinned shipping defaults (as k10: cache auto)
if "bench" in STEPS:
    step("bench_q3m", "time", "--cell", "q3m:spill8192", "--reps", "3", arms="k10,k10b,k15,k15b", extra=MTP, timeout=7200)
    step("bench_q35", "time", "--cell", "q35:full", "--reps", "3", arms="k10,k10b,k15,k15b", timeout=3600)
# 2 + 6. full test-backend-ops, k15 / k10, then k15 with the 4096-byte pad; stock only when a fork bin fails something (attribution)
if "tbofull" in STEPS:
    for b in ("k15", "k10"):
        step("tbofull_" + b, "tbo", "--bin", b, "--ops", "ALL", "--timeout", "7200", timeout=7500)
if "tbopad" in STEPS:
    step("tbofull_k15_p4096", "tbo", "--bin", "k15_p4096", "--ops", "ALL", "--timeout", "7200", timeout=7500)
if "stock" in STEPS and not WIN:
    full = [r for r in rows() if r.get("kind") == "tbo" and r.get("op") == "ALL"]
    if any(r.get("fail_ids") for r in full) and os.path.isdir(os.path.join(os.environ["K6P_DIR"], "bins", "up")):
        step("tbofull_up", "tbo", "--bin", "up", "--ops", "ALL", "--timeout", "7200", timeout=7500)
    else:
        print("STEP tbofull_up skipped: no full-suite failure on k15 / k10 to attribute", flush=True)
with open(os.path.join(OUT, "plan.json"), "w", encoding="utf-8") as fh:
    json.dump({"steps": PLAN, "wall_s": round(time.time() - T0), "win": WIN}, fh)
print("=== PLAN DONE %.0f s ===" % (time.time() - T0))
