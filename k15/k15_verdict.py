#!/usr/bin/env python3
"""k15_verdict.py DIR OUT_DIR: judge the k15 AMD run from its artifacts (downloaded under DIR), stdlib only.

The probes (k15_bench.py via k15_plan.py / linux_cpu.sh) observe; this module judges. One line per test-list item and OS:
  PASS / FAIL (a real finding) / VOID (the base or kill-switch arm did not show what makes the comparison meaningful, an A/A pair
  differed, or the new path provably did not run) / INCONCLUSIVE (rows missing: the step did not run or timed out).
Writes OUT_DIR/VERDICT.md + verdict.json. Exit 0 only when every item is PASS (or NO_REGRESSION for the bench); 1 otherwise.
"""
import glob, json, os, re, statistics, sys

D, OD = sys.argv[1], sys.argv[2]
os.makedirs(OD, exist_ok=True)
RES, MD = [], []
NEEDS = ["rocm", "gpu", "exclusive_gpu", "linux", "windows", "windows_rocm_wddm", "nvidia", "multi_gpu", "discrete_gpu"]


def rd(p):
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def load(artifact):
    """rows.jsonl of one job's artifact (k15-linux-gpu / k15-windows-gpu / k15-linux-cpu), written incrementally by k15_bench.py"""
    out = []
    for p in glob.glob(os.path.join(D, "**", artifact, "**", "rows.jsonl"), recursive=True):
        with open(p, encoding="utf-8", errors="replace") as fh:
            out += [json.loads(l) for l in fh if l.strip().startswith("{")]
    return out


def add(item, os_, verdict, why):
    RES.append({"item": item, "os": os_, "verdict": verdict, "why": why})


def text_key(r):
    if not r or not r.get("ok") or r.get("err"):
        return None
    return ((r.get("dec") or {}).get("md5"), (r.get("p2k") or {}).get("md5"))


def srv(rows, arm, model=None):
    return [r for r in rows if r.get("kind") == "server" and r.get("arm") == arm and (model is None or r.get("model") == model)]


def one(rows, arm, model=None):
    x = srv(rows, arm, model)
    return x[-1] if x else None


def died(r):
    return r is None or not r.get("ok") or bool(r.get("err")) or (r.get("stats") or {}).get("abort") or r.get("died_before_kill")


def tbo(rows, b, op):
    x = [r for r in rows if r.get("kind") == "tbo" and r.get("bin") == b and r.get("op") == op]
    return x[-1] if x else None


def tbo_ok(r):
    """(ok, passed, total): every selected case passed on the device under test, at least one ran, no abort / timeout"""
    if r is None or r.get("err"):
        return None, 0, 0
    p = r.get("passed") or [0, 0]
    a, b = int(p[0]), int(p[1])
    return (r.get("rc") == 0 and a == b and b > 0 and not r.get("aborted")), a, b


# ---------------------------------------------------------------- 1. build
def item1():
    for n in ("k15", "k10", "up"):
        f = glob.glob(os.path.join(D, "**", "SRC_CHECK_" + n), recursive=True)
        if not f:
            if n != "up":   # the stock bin is only used to attribute a full-suite failure (item 2 says when it was needed)
                add("1 build " + n, "linux", "INCONCLUSIVE", "no SRC_CHECK_%s (hosted build did not finish)" % n)
            continue
        s = rd(f[0])
        m = re.search(r"tree \S+ (\S+) expect (\S+)", s)
        tree_ok = bool(m and m.group(1) == m.group(2))
        brc = re.search(r"^build %s rc=(\d+)" % n, s, re.M)
        trc = re.search(r"tests build %s rc=(\d+)" % n, s)
        hip = re.search(r"mapping lines: (\d+)", s)
        graphs = re.search(r"GGML_HIP_GRAPHS:BOOL=(\w+)", s)
        ok = tree_ok and brc and brc.group(1) == "0" and trc and trc.group(1) == "0"
        if n == "k15":
            ok = ok and hip and hip.group(1) == "1" and graphs and graphs.group(1) == "ON"
        elif hip and hip.group(1) != "0":
            ok = False
        add("1 build " + n, "linux", "PASS" if ok else "FAIL",
            "tree %s, build rc %s, tests rc %s, hip.h cudaErrorInvalidValue mapping %s, GGML_HIP_GRAPHS %s" % (
                "match" if tree_ok else "MISMATCH", brc and brc.group(1), trc and trc.group(1), hip and hip.group(1), graphs and graphs.group(1)))
    # hosted CPU ctest: k15 failing tests must be a subset of k10's (same hosted image, no GPU)
    fl = {}
    for n in ("k15", "k10"):
        f = glob.glob(os.path.join(D, "**", "ctest_%s.log" % n), recursive=True)
        if f:
            fl[n] = set(re.findall(r"^\s*\d+ - (\S+) \((?:Failed|Timeout|SEGFAULT|Subprocess aborted|Exception)", rd(f[0]), re.M))
    if len(fl) == 2:
        new = sorted(fl["k15"] - fl["k10"])
        add("1 ctest -L main (hosted CPU)", "linux", "FAIL" if new else "PASS", "k15 failing %s, k10 failing %s, new %s" % (
            sorted(fl["k15"]), sorted(fl["k10"]), new))
    else:
        add("1 ctest -L main (hosted CPU)", "linux", "INCONCLUSIVE", "ctest logs missing for %s" % sorted({"k15", "k10"} - set(fl)))
    f = glob.glob(os.path.join(D, "**", "win_diag.txt"), recursive=True)
    if not f:
        add("1 build", "windows", "INCONCLUSIVE", "no win_diag.txt")
        return
    s = rd(f[0])
    for n in ("k15", "k10"):
        m = re.search(r"^%s tree (\S+) expect (\S+)" % n, s, re.M)
        b = re.search(r"^%s build rc=(\d+)" % n, s, re.M)
        h = re.search(r"^%s hip.h cudaErrorInvalidValue mapping lines: (\d+)" % n, s, re.M)
        g = re.search(r"^%s cache: GGML_HIP_GRAPHS:BOOL=(\w+)" % n, s, re.M)
        ok = bool(m and m.group(1) == m.group(2) and b and b.group(1) == "0")
        if n == "k15":
            ok = ok and bool(h and h.group(1) == "1" and g and g.group(1) == "ON")
        add("1 build " + n, "windows", "PASS" if ok else ("INCONCLUSIVE" if not b else "FAIL"),
            "tree %s, build rc %s, hip.h mapping %s, GGML_HIP_GRAPHS %s" % (
                "match" if m and m.group(1) == m.group(2) else "MISMATCH/missing", b and b.group(1), h and h.group(1), g and g.group(1)))


# ---------------------------------------------------------------- 2 / 6 / 7. test-backend-ops
def item2(rows, os_):
    h, b = tbo(rows, "k15", "ALL"), tbo(rows, "k10", "ALL")
    hf = h
    st = tbo(rows, "up", "ALL")
    if h is None or b is None:
        add("2 test-backend-ops full", os_, "INCONCLUSIVE", "full-suite row missing: k15 %s, k10 %s" % (h is not None, b is not None))
    else:
        hp, bp = h.get("passed") or [0, 0], b.get("passed") or [0, 0]
        if int(hp[1]) == 0:
            add("2 test-backend-ops full", os_, "VOID", "ROCm0 ran no case (per backend %s)" % h.get("per_backend"))
        else:
            new = sorted(set(h.get("fail_ids") or []) - set(b.get("fail_ids") or []))
            att = ""
            if new and st is not None:
                up = set(st.get("fail_ids") or [])
                att = "; of the new, also failing on stock ff5a2337: %s" % [x for x in new if x in up]
            ok = not new and h.get("rc") in (0, 1) and not h.get("aborted") and h.get("rc") != "timeout"
            add("2 test-backend-ops full", os_, "PASS" if ok else "FAIL", "k15 %s/%s rc %s, k10 %s/%s rc %s; failing only on k15: %s%s" % (
                hp[0], hp[1], h.get("rc"), bp[0], bp[1], b.get("rc"), new[:20], att))
    g = [r for r in rows if r.get("kind") == "exe" and r.get("exe") == "test-graph-src-type"]
    for r in g:
        m = r.get("match")
        ok = r.get("rc") == 0 and m and m[0] == m[1] and int(m[1]) > 0 and not r.get("skipped")
        add("2 test-graph-src-type " + r["bin"], os_, "PASS" if ok else ("FAIL" if m or r.get("skipped") else "INCONCLUSIVE"),
            "rc %s, %s evaluations match, skipped %s, src-type recaptures %s" % (r.get("rc"), m, r.get("skipped"), r.get("src_type_recaptures")))
    if not g:
        add("2 test-graph-src-type", os_, "INCONCLUSIVE", "no row")
    groups = [("2 fusion cases", ("k15", "k15_cc", "k15_qp", "k15_qk", "k15_gc", "k15_sg0", "k15_ks5"),
               ("CONV_CAT_STATE", "QSA_POOL_KEYS", "QSA_KEY_PACK", "GATE_CONT", "QSA_SEL_FA")),
              ("2 GATED_DELTA_NET", ("k15", "k15_ip0", "k15_gg0", "k10"), ("GATED_DELTA_NET",)),
              ("2 GATED_MMVQ / GATED_Q8_1", ("k15", "k15_q80", "k15_glu1"), ("GATED_MMVQ", "GATED_Q8_1"))]
    for name, bins, ops in groups:
        cells, bad, missing, empty = [], [], [], []
        for bn in bins:
            for op in ops:
                r = tbo(rows, bn, op)
                ok, a, t = tbo_ok(r)
                cells.append("%s/%s %s/%s" % (bn, op, a, t))
                if r is None:
                    missing.append(bn + "/" + op)
                elif t == 0:
                    empty.append(bn + "/" + op)
                elif not ok:
                    bad.append("%s/%s %s" % (bn, op, (r.get("fail_ids") or [])[:3]))
        v = "FAIL" if bad else "INCONCLUSIVE" if (missing or empty) else "PASS"
        add(name, os_, v, "failed: %s; missing: %s; no case run (not supported on ROCm0?): %s; cells: %s" % (bad, missing, empty, ", ".join(cells)))
    # 7. MUL_MAT_ID at pads 0 / 37 / 4096 (k15 failures must be k10's at the same pad), + the CPU-reference arm MULTI_K=0
    cells, bad, missing = [], [], []
    for pad in ("0", "37", "4096"):
        h, b = tbo(rows, "k15_p" + pad, "MUL_MAT_ID"), tbo(rows, "k10_p" + pad, "MUL_MAT_ID")
        if h is None or b is None:
            missing.append(pad)
            continue
        okh, ah, th = tbo_ok(h)
        okb, ab, tb_ = tbo_ok(b)
        new = sorted(set(h.get("fail_ids") or []) - set(b.get("fail_ids") or []))
        cells.append("pad %s k15 %s/%s k10 %s/%s" % (pad, ah, th, ab, tb_))
        if th == 0 or new or h.get("aborted") or h.get("rc") == "timeout":
            bad.append("pad %s new %s" % (pad, new[:3]))
    r = tbo(rows, "k15_mk0", "MUL_MAT_ID")
    okm, am, tm = tbo_ok(r)
    cells.append("k15 GGML_CPU_MMID_MULTI_K=0 %s/%s" % (am, tm))
    if r is not None and not okm:
        bad.append("MULTI_K=0 %s" % (r.get("fail_ids") or [])[:3])
    add("7 MUL_MAT_ID pads 0/37/4096", os_, "FAIL" if bad else "INCONCLUSIVE" if missing or r is None else "PASS",
        "failed: %s; missing pads: %s; %s" % (bad, missing, ", ".join(cells)))
    # 6. full suite with the 4096-byte pad: no case may fail only under the pad
    p = tbo(rows, "k15_p4096", "ALL")
    if os_ == "windows" and p is None:
        pass   # not scheduled on Windows (job budget): Linux only
    elif p is None or hf is None:
        add("6 test-backend-ops full, TBOPAD=4096", os_, "INCONCLUSIVE", "row missing")
    else:
        pp = p.get("passed") or [0, 0]
        new = sorted(set(p.get("fail_ids") or []) - set(hf.get("fail_ids") or []))
        gone = sorted(set(hf.get("fail_ids") or []) - set(p.get("fail_ids") or []))
        v = "VOID" if int(pp[1]) == 0 else "FAIL" if new or p.get("aborted") or p.get("rc") == "timeout" else "PASS"
        add("6 test-backend-ops full, TBOPAD=4096", os_, v, "pad %s/%s rc %s; failing only with the pad: %s; failing only without: %s" % (
            pp[0], pp[1], p.get("rc"), new[:20], gone[:20]))


# ---------------------------------------------------------------- 3 / 4 / 5 / 8. exactness
def eq_report(rows, arms, model):
    return {a: (text_key(one(rows, a, model)) or ("DIED" if one(rows, a, model) else "MISSING",)) for a in arms}


def item3(rows, os_):
    for model, p in (("q35", "e_"), ("q3m", "m_")):
        A = {k: one(rows, p + k, model) for k in ("g", "d", "d2", "f1", "f2", "fks")}
        if any(A[k] is None for k in A):
            add("3 eager fallback " + model, os_, "INCONCLUSIVE", "missing arms %s" % [k for k in A if A[k] is None])
            continue
        keys = {k: text_key(A[k]) for k in A}
        fb = {k: (A[k].get("stats") or {}).get("eager_fallback_n", 0) for k in A}
        why = "md5 %s; fallback lines %s" % ({k: (v and "/".join(x[:8] for x in v if x)) for k, v in keys.items()}, fb)
        if any(died(A[k]) for k in ("g", "d", "d2", "f1", "f2")):
            add("3 eager fallback " + model, os_, "FAIL", "an arm aborted or did not answer: %s; %s" % (
                [k for k in ("g", "d", "d2", "f1", "f2") if died(A[k])], why))
        elif keys["d"] != keys["d2"]:
            add("3 eager fallback " + model, os_, "VOID", "A/A default pair differs (text not reproducible at these pins); " + why)
        elif fb["f1"] == 0 or fb["f2"] == 0:
            add("3 eager fallback " + model, os_, "VOID", "a fault arm logged no 'runs without the graph' (fallback never ran); " + why)
        elif not died(A["fks"]):
            add("3 eager fallback " + model, os_, "VOID", "faults with GGML_CUDA_GRAPH_EAGER_FALLBACK=0 did not abort: the injected failure "
                "did not reach the old abort path, so the fault arms prove nothing; " + why)
        elif not (keys["f1"] == keys["g"] and keys["f2"] == keys["g"] and keys["d"] == keys["g"]):
            add("3 eager fallback " + model, os_, "FAIL", "text differs from graphs-off; " + why)
        else:
            add("3 eager fallback " + model, os_, "PASS", "graphs off = default x2 = faults = faults + destroy fault; fallback off aborts "
                "(%s); " % str((A["fks"].get("stats") or {}).get("err_lines") or A["fks"].get("err", ""))[:200] + why)


def same_set(rows, model, arms, item, os_, aa_pairs, extra=""):
    A = {a: one(rows, a, model) for a in arms}
    miss = [a for a in arms if A[a] is None]
    if miss:
        add(item, os_, "INCONCLUSIVE", "missing arms %s" % miss)
        return None
    keys = {a: text_key(A[a]) for a in arms}
    why = "md5 %s%s" % ({a: (k and "/".join(x[:8] for x in k if x)) for a, k in keys.items()}, extra)
    dead = [a for a in arms if died(A[a])]
    if dead:
        add(item, os_, "FAIL", "aborted / no answer: %s (%s); %s" % (dead, [(A[a].get("stats") or {}).get("err_lines") for a in dead][:2], why))
    elif any(keys[x] != keys[y] for x, y in aa_pairs):
        add(item, os_, "VOID", "an A/A pair differs %s; %s" % ([(x, y) for x, y in aa_pairs if keys[x] != keys[y]], why))
    elif len(set(keys.values())) != 1:
        add(item, os_, "FAIL", "text differs between arms; " + why)
    else:
        add(item, os_, "PASS", "all arms the same text; " + why)
    return A


def item45(rows, os_):
    A = same_set(rows, "q3m", ["m_d", "m_d2", "m_ip0", "m_dbg"], "4 GDN in-place, pinned cache 1024", os_, [("m_d", "m_d2")])
    kept = (A["m_dbg"].get("stats") or {}).get("gather_kept_n") if A else None
    B = same_set(rows, "q3m", ["n_d", "n_d2", "n_ip0", "n_dbg"], "4 GDN in-place, spilled --moe-cache-mib 0", os_, [("n_d", "n_d2")])
    keptn = (B["n_dbg"].get("stats") or {}).get("gather_kept_n") if B else None
    r = one(rows, "n_k10", "q3m")
    add("8 GDN kept-until guard, debug log", os_, "PASS" if A and B and not died(A["m_dbg"]) and not died(B["n_dbg"]) else "INCONCLUSIVE",
        "gather-kept lines: pinned 1024 %s, cache 0 %s (%s); k10 at cache 0 (reference): %s" % (
            kept, keptn, (B["n_dbg"].get("stats") or {}).get("gather_kept", [])[:1] if B else "", "aborted" if died(r) else "ran" if r else "missing"))
    same_set(rows, "q3m", ["m_d", "m_d2", "m_a0", "m_a02", "m_g"], "5 race fix: default vs GGML_CUDA_ADD_RMS_FUSE=0 vs graphs off", os_,
             [("m_d", "m_d2"), ("m_a0", "m_a02")])


# ---------------------------------------------------------------- r2 (c5429be4abd) additions
def item_r2(rows, os_):
    st = lambda r: (r or {}).get("stats") or {}
    # mailbox handshake fast path (c98b2f662f2) + one-region host experts (5f9e82fd578): kill switches vs default at the m_ pins
    A = same_set(rows, "q3m", ["m_d", "m_d2", "m_mbf0", "m_sf0", "m_s1"], "r2 mailbox fast / one-region host experts (switches)", os_,
                 [("m_d", "m_d2")])
    if A:
        on = {a: st(A[a]).get("single_on_n", 0) for a in A}
        nomb = st(A["m_s1"]).get("single_nomb_n", 0)
        add("r2 mailbox path reach on HIP", os_, "PASS" if not died(A["m_s1"]) else "FAIL",
            "single-graph host misses on: %s (0 = the path is off, as the code says for a backend without mapped mailboxes); forced "
            "LLAMA_MOE_SINGLE=1: 'no mapped mailbox' lines %d, %s" % (on, nomb, "ran on the split path" if not died(A["m_s1"]) else "ABORTED"))
    # replay-commit fallback (1811f0c7468), 4 GB-like budget + MTP
    R = {a: one(rows, a, "q3m") for a in ("r4_d", "r4_d2", "r4_ff", "r4_fb0")}
    if any(R[a] is None for a in ("r4_d", "r4_d2", "r4_ff")):
        add("r2 replay commit fallback, 4 GB budget", os_, "INCONCLUSIVE", "missing %s" % [a for a in R if R[a] is None])
    else:
        k = {a: text_key(R[a]) for a in R}
        why = "md5 %s; replay commits %s; fallback commits %s; reserve off + forced: %s" % (
            {a: (v and "/".join(x[:8] for x in v if x)) for a, v in k.items()}, {a: st(R[a]).get("rs_commits") for a in R if R[a]},
            {a: st(R[a]).get("rs_fallback_commits") for a in R if R[a]}, "aborted" if died(R["r4_fb0"]) else "ran" if R["r4_fb0"] else "missing")
        if any(died(R[a]) for a in ("r4_d", "r4_d2", "r4_ff")):
            add("r2 replay commit fallback, 4 GB budget", os_, "FAIL", "an arm aborted: %s; %s" % ([a for a in ("r4_d", "r4_d2", "r4_ff") if died(R[a])], why))
        elif k["r4_d"] != k["r4_d2"]:
            add("r2 replay commit fallback, 4 GB budget", os_, "VOID", "A/A pair differs; " + why)
        elif st(R["r4_ff"]).get("rs_fallback_commits", 0) == 0:
            add("r2 replay commit fallback, 4 GB budget", os_, "VOID", "LLAMA_RS_REPLAY_FALLBACK_FORCE=1 put no commit in the fallback buffer; " + why)
        elif k["r4_ff"] != k["r4_d"]:
            add("r2 replay commit fallback, 4 GB budget", os_, "FAIL", "the fallback buffer changes the text; " + why)
        else:
            add("r2 replay commit fallback, 4 GB budget", os_, "PASS", "no abort, fallback path = default text; " + why)
    # release-split (deb1079e81c + a4b47119d1a + c5429be4abd), 4 GB budget + MTP, chat / 2k / 8k: no abort, the release counted
    Q = {a: one(rows, a, "q3m") for a in ("rsp_on", "rsp_on2", "rsp_off")}
    if any(v is None for v in Q.values()):
        add("r4 release-split, 4 GB budget", os_, "INCONCLUSIVE", "missing %s" % [a for a in Q if Q[a] is None])
    else:
        key8 = lambda r: text_key(r) and text_key(r) + ((r.get("p8k") or {}).get("md5"),)
        k = {a: key8(Q[a]) for a in Q}
        cnt = {a: {c: st(Q[a]).get(c, 0) for c in ("relsplit_n", "bank_release_n", "pool_trim_n", "free_after_release_n")} for a in Q}
        why = "release counts (capped in the log) %s; md5 chat/2k/8k %s; lines %s" % (
            cnt, {a: (v and "/".join(str(x)[:8] for x in v if x)) for a, v in k.items()}, st(Q["rsp_on"]).get("release_lines", [])[:2])
        if any(died(Q[a]) for a in Q) or any(not (Q[a].get("p8k") or {}).get("md5") for a in Q):
            add("r4 release-split, 4 GB budget", os_, "FAIL", "an arm aborted or the 8k cell did not answer: %s; %s" % (
                [a for a in Q if died(Q[a]) or not (Q[a].get("p8k") or {}).get("md5")], why))
        elif cnt["rsp_on"]["relsplit_n"] == 0 and cnt["rsp_on2"]["relsplit_n"] == 0:
            add("r4 release-split, 4 GB budget", os_, "VOID", "no release-split decision in either default launch (the path did not run at "
                "this budget; banks released by the fixed share: %s); " % cnt["rsp_on"]["bank_release_n"] + why)
        elif cnt["rsp_off"]["relsplit_n"] > 0:
            add("r4 release-split, 4 GB budget", os_, "FAIL", "LLAMA_MOE_LM_RELEASE_SPLIT=0 still took the release-split path; " + why)
        else:
            add("r4 release-split, 4 GB budget", os_, "PASS", "no abort on / off; A/A text %s; text on vs off %s (a split graph may "
                "batch differently: reported, not gated); " % ("same" if k["rsp_on"] == k["rsp_on2"] else "DIFFERS",
                                                                 "same" if k["rsp_on"] == k["rsp_off"] else "differs") + why)
    # r5: lend free check (f3a105b68e3) + scheduler trim / arena fit (0363724a80a), 4 GB budget + MTP, chat / 2k / 8k: no abort,
    # every arm answers the 8k cell; text on vs kill switches reported (a different lend / arena can batch differently), not gated
    Q = {a: one(rows, a, "q3m") for a in ("b3_on", "b3_on2", "b3_lf0", "b3_tr0", "b3_dr0", "b3_rc0")}
    if any(v is None for v in Q.values()):
        add("r5 lend free check / sched trim, 4 GB budget", os_, "INCONCLUSIVE", "missing %s" % [a for a in Q if Q[a] is None])
    else:
        key8 = lambda r: text_key(r) and text_key(r) + ((r.get("p8k") or {}).get("md5"),)
        k = {a: key8(Q[a]) for a in Q}
        why = "md5 chat/2k/8k %s; counts %s" % ({a: (v and "/".join(str(x)[:8] for x in v if x)) for a, v in k.items()},
                                               {a: {c: st(Q[a]).get(c, 0) for c in ("lend_whole_n", "pf_notused_n", "oom_n")} for a in Q})
        if any(died(Q[a]) for a in Q) or any(not (Q[a].get("p8k") or {}).get("md5") for a in Q):
            add("r5 lend free check / sched trim, 4 GB budget", os_, "FAIL", "an arm aborted or the 8k cell did not answer: %s; %s" % (
                [a for a in Q if died(Q[a]) or not (Q[a].get("p8k") or {}).get("md5")], why))
        else:
            add("r5 lend free check / sched trim, 4 GB budget", os_, "PASS", "no abort on / off; A/A text %s; on vs lf0 %s, vs tr0 %s" % (
                "same" if k["b3_on"] == k["b3_on2"] else "DIFFERS", "same" if k["b3_on"] == k["b3_lf0"] else "differs",
                "same" if k["b3_on"] == k["b3_tr0"] else "differs") + ", vs dr0 %s; " % ("same" if k["b3_on"] == k["b3_dr0"] else "differs") + why)
    # PLE read-in: Linux makes the lazy table resident, Windows leaves it lazy (the read-in is Linux only); same text as LLAMA_PLE_RESIDENT=0
    P = {a: one(rows, a, "g4e2b") for a in ("ple_d", "ple_0", "ple_d2")}
    if any(v is None for v in P.values()):
        add("r2 PLE read-in", os_, "INCONCLUSIVE", "missing %s" % [a for a in P if P[a] is None])
        return
    k = {a: text_key(P[a]) for a in P}
    d = st(P["ple_d"])
    why = "default: made resident %d, left lazy %d (%s); md5 %s" % (d.get("ple_resident_n", 0), d.get("ple_lazy_n", 0), d.get("ple_res_lines", [])[:1],
                                                               {a: (v and "/".join(x[:8] for x in v if x)) for a, v in k.items()})
    if any(died(P[a]) for a in P):
        add("r2 PLE read-in", os_, "FAIL", "an arm aborted; " + why)
    elif k["ple_d"] != k["ple_d2"]:
        add("r2 PLE read-in", os_, "VOID", "A/A pair differs; " + why)
    elif os_ == "linux" and d.get("ple_resident_n", 0) == 0:
        add("r2 PLE read-in", os_, "VOID", "the table was not made resident on Linux (the read-in never ran); " + why)
    elif os_ == "windows" and (d.get("ple_resident_n", 0) > 0 or d.get("ple_lazy_n", 0) == 0):
        add("r2 PLE read-in", os_, "FAIL" if d.get("ple_resident_n", 0) else "VOID",
            ("made resident on Windows" if d.get("ple_resident_n", 0) else "no PLE decision line (the table was not lazy)") + "; " + why)
    elif k["ple_d"] != k["ple_0"]:
        add("r2 PLE read-in", os_, "FAIL", "text differs from LLAMA_PLE_RESIDENT=0; " + why)
    else:
        add("r2 PLE read-in", os_, "PASS", ("resident on Linux" if os_ == "linux" else "off on Windows") + ", same text as off; " + why)


# ---------------------------------------------------------------- 10. draft guard
def item10(rows, os_):
    dflt = [r for r in rows if r.get("kind") == "server" and r.get("model") == "q3m" and str(r.get("arm", "")).startswith(("k15", "m_", "n_", "r4_", "rsp_"))
            and not (r.get("arm_env") or {}).get("LLAMA_DRAFT_MIN_FREE_MIB") and r.get("ok")]
    f = one(rows, "dg_force", "q3m")
    if not dflt or f is None:
        add("10 draft guard (no 192 MiB floor on HIP)", os_, "INCONCLUSIVE", "rows: default %d, forced %s" % (len(dflt), f is not None))
        return
    floor = sum((r.get("stats") or {}).get("draft_floor_n", 0) for r in dflt)
    off = sum((r.get("stats") or {}).get("draft_off_n", 0) + (r.get("stats") or {}).get("mtp_ctx_fail_n", 0) for r in dflt)
    spec = sum(1 for r in dflt if (r.get("stats") or {}).get("spec"))
    fs = f.get("stats") or {}
    why = "default k15 MTP launches %d: floor lines %d, draft-off lines %d, launches with draft stats %d; forced floor: floor lines %d, draft-off %d" % (
        len(dflt), floor, off, spec, fs.get("draft_floor_n", 0), fs.get("draft_off_n", 0))
    if fs.get("draft_floor_n", 0) == 0:
        add("10 draft guard (no 192 MiB floor on HIP)", os_, "VOID", "LLAMA_DRAFT_MIN_FREE_MIB=1000000 did not trip the check on HIP; " + why)
    elif floor or off:
        add("10 draft guard (no 192 MiB floor on HIP)", os_, "FAIL", "the draft was dropped by default; " + why)
    else:
        add("10 draft guard (no 192 MiB floor on HIP)", os_, "PASS", why)


# ---------------------------------------------------------------- 11. bench
def item11(rows, os_):
    for model, mode in (("q3m", "spill8192"), ("q35", "full")):
        R = [r for r in rows if r.get("kind") == "server" and r.get("model") == model and r.get("mode") == mode
             and r.get("arm") in ("k10", "k10b", "k15", "k15b") and r.get("ok") and not r.get("err")]
        if len({r["arm"] for r in R}) < 4:
            add("11 bench %s %s" % (model, mode), os_, "INCONCLUSIVE", "arms present %s" % sorted({r["arm"] for r in R}))
            continue
        parts, worst = [], "NO_REGRESSION"
        for cell, met in (("dec", "tg_tps"), ("p2k", "pp_tps"), ("p2k", "tg_tps")):
            med = {a: statistics.median([(r.get(cell) or {}).get(met) or 0 for r in R if r["arm"] == a]) for a in ("k10", "k10b", "k15", "k15b")}
            base = statistics.median([(r.get(cell) or {}).get(met) or 0 for r in R if r["arm"] in ("k10", "k10b")])
            head = statistics.median([(r.get(cell) or {}).get(met) or 0 for r in R if r["arm"] in ("k15", "k15b")])
            if not base:
                worst = "INCONCLUSIVE"
                continue
            noise = max(abs(med["k10"] - med["k10b"]) / base, abs(med["k15"] - med["k15b"]) / max(head, 1e-9))
            d = head / base - 1
            tag = "REGRESSION" if d < -(noise + 0.03) else "inside A/A noise" if abs(d) <= noise else "faster" if d > 0 else "slower, inside noise + 3%"
            if tag == "REGRESSION":
                worst = "FAIL"
            parts.append("%s %s: k10 %.2f k15 %.2f (%+.1f%%, A/A %.1f%%) %s" % (cell, met, base, head, 100 * d, 100 * noise, tag))
        md5s = {a: sorted({str(text_key(r)) for r in R if r["arm"] == a}) for a in ("k10", "k15")}
        parts.append("text k10 %s k15 %s" % ({a: len(v) for a, v in md5s.items()}, "same" if md5s["k10"] == md5s["k15"] else "differs (code changed)"))
        add("11 bench %s %s" % (model, mode), os_, worst, "; ".join(parts))


# ---------------------------------------------------------------- 9. CPU
def item9(rows):
    if not rows:
        add("9 CPU multi-column MUL_MAT_ID (Zen 5)", "linux", "INCONCLUSIVE", "no rows in the k15-linux-cpu artifact")
        return
    bad, cells = [], []
    for a in ("c512_k1", "c512_k0", "c2_k1", "c2_k0"):
        e = [r for r in rows if r.get("kind") == "exe" and r.get("bin") == a]
        t = [r for r in rows if r.get("kind") == "tbo" and r.get("bin") == a]
        e, t = (e[-1] if e else None), (t[-1] if t else None)
        eok = e is not None and e.get("rc") == 0 and e.get("ok_line") == "OK" and e.get("n_bitwise", 0) > 0
        tok, ta, tt = tbo_ok(t)
        cells.append("%s mmid %s (%s cases) tbo %s/%s" % (a, e and e.get("ok_line"), e and e.get("n_bitwise"), ta, tt))
        if not eok or not tok:
            bad.append(a)
    f = glob.glob(os.path.join(D, "**", "cpu_diag.txt"), recursive=True)
    cpu = re.findall(r"model name\s*:\s*([^\n]+)", rd(f[0])) if f else []
    add("9 CPU multi-column MUL_MAT_ID (Zen 5)", "linux", "FAIL" if bad else "PASS", "host %s; failing arms %s; %s" % (cpu[:1], bad, ", ".join(cells)))


def untested(os_):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "amd_ci", "lib"))
    f = [p for p in glob.glob(os.path.join(D, "**", "host.json"), recursive=True) if ("windows" in p) == (os_ == "windows")]
    try:
        import capability
        if not f:
            return "### Not tested here (%s)\n\n- no host.json from this job: the host profile was never recorded" % os_
        with open(f[0], encoding="utf-8") as fh:
            prof = capability.HostProfile(**json.load(fh))
        return capability.untested_section(prof, NEEDS).replace("### Not tested here", "### Not tested here (%s job)" % os_, 1)
    except Exception as e:  # noqa: BLE001
        return "### Not tested here (%s)\n\n- capability.py failed: %r" % (os_, e)


item1()
for os_, fname in (("linux", "k15-linux-gpu"), ("windows", "k15-windows-gpu")):
    rows = load(fname)
    if not rows:
        add("2-11 GPU items", os_, "INCONCLUSIVE", "no %s (the GPU job did not run or produced nothing)" % fname)
        continue
    item2(rows, os_)
    item3(rows, os_)
    item45(rows, os_)
    item_r2(rows, os_)
    item10(rows, os_)
    item11(rows, os_)
item9(load("k15-linux-cpu"))

# k15c (correctness ticket): items it does not run are dropped when they have no rows (INCONCLUSIVE), and listed as skipped
SKIP = os.environ.get("K15_SKIP_ITEMS")
SKIPPED = [r for r in RES if SKIP and r["verdict"] == "INCONCLUSIVE" and re.search(SKIP, r["os"] + ":" + r["item"])]
RES = [r for r in RES if r not in SKIPPED]
order = {"FAIL": 0, "VOID": 1, "INCONCLUSIVE": 2, "PASS": 3, "NO_REGRESSION": 3}
MD.append("# k15 AMD gfx1151: a58cb888e2b (k15) vs a162dce854 (k10), upstream merge base ff5a2337\n")
MD.append("| item | OS | verdict | evidence |\n|---|---|---|---|")
for r in RES:
    MD.append("| %s | %s | **%s** | %s |" % (r["item"], r["os"], r["verdict"], r["why"].replace("|", "/")[:1800]))
counts = {}
for r in RES:
    counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
MD.append("\nTotals: %s\n" % counts)
if SKIPPED:
    MD.append("Skipped (not in this ticket, no rows): %s\n" % ", ".join("%s (%s)" % (r["item"], r["os"]) for r in SKIPPED))
MD.append(untested("linux"))
MD.append("")
MD.append(untested("windows"))
with open(os.path.join(OD, "VERDICT.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(MD) + "\n")
with open(os.path.join(OD, "verdict.json"), "w", encoding="utf-8") as fh:
    json.dump({"verdict": min((r["verdict"] for r in RES), key=lambda v: order[v]) if RES else "INCONCLUSIVE", "items": RES, "counts": counts}, fh, indent=1)
print("\n".join(MD))
for r in RES:
    if r["verdict"] not in ("PASS", "NO_REGRESSION"):
        print("::error title=k15 %s (%s)::%s" % (r["item"], r["os"], r["verdict"]))
sys.exit(0 if all(r["verdict"] in ("PASS", "NO_REGRESSION") for r in RES) else 1)
