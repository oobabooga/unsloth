#!/usr/bin/env python3
"""Round-8 gfx1151 Linux KERNEL items (observes only; judged by criteria/r8_judge_kern.py). Imported by r8_probe.py.

ITEMS (run order; each stores obs[name], ctx['save']() after every sub-result):
  kern_smoke  seconds: Studio RDNA W8A8 prepare + one GEMM vs reference_mm, INT8 attention table, ROCm tooling
  kern_build  comfy-kitchen HIP extension for gfx1151 at #338 head (k338) and its merge base (kmain), in scratch,
              concurrently with the token-free unsloth/Qwen-Image-2.1 prefetch (CPU only, before any GPU timing)
  kern_prof   C9-3 step 0 + carbon9 #4: Qwen-Image-2.1 W8A8 1024^2 render through Studio (main, defaults, 3 renders),
              kernel trace by rocprofv3 (fallback: torch.profiler on the last render); share of w8a8_mm (INT8 GEMM),
              rotq_rows, other GEMMs, attention per render window
  kern_a1     carbon6 #1: GROUP_M {1,2,4,8} x {contiguous out, out pitch N+128}, Qwen-2.1 shapes, M 4096/4352/2049/4097,
              3 fresh processes; then, only if a pitch arm clears 1.05x over the best GROUP_M, a Qwen-2.1 E2E
              M P P M (P = Studio with the padded output view, kern_e2e_pad.py)
  kern_bufops round-7 n6#7: AMDGCN_USE_BUFFER_OPS=0/1 processes O N N O N O, Studio tile, M 4608 / 9728
  kern_k338   C9-3 step 1: Studio w8a8_mm vs kitchen _C.int8_gemm, #338 shapes + Qwen-2.1 shapes, ragged M,
              processes K M M K K M (K = k338 build, M = kmain build); skipped when the INT8 GEMM share < 40%
  kern_attn   M10: kitchen int8_attention vs Studio Triton INT8 attention (f16_t), D128, S 4k / 16k / 37.7k,
              3 processes (arm order alternating)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r8_common import R8, child_env, e2e, prefetch, run_py  # noqa: E402

Q21 = "unsloth/Qwen-Image-2.1"
GEN = {"width": 1024, "height": 1024}
SHARE_KILL = 0.40


def _smoke_ok(obs):
    return bool((obs.get("kern_smoke") or {}).get("ok"))


def item_kern_smoke(ctx, obs):
    o = Path(ctx["work"]) / "kern_smoke.json"
    obs["kern_smoke"] = run_py(ctx, R8 / "kern_smoke.py", ["--backend", ctx["base_backend"], "--out", o], o,
                               timeout = 900, tag = "kern_smoke")
    ctx["save"]()


def item_kern_build(ctx, obs):
    dest = Path(ctx["scratch"]) / "kitchen"
    o = Path(ctx["work"]) / "kern_build.json"
    pre: dict = {}

    def fetch():
        pre["r"] = prefetch(ctx, Q21)
    th = threading.Thread(target = fetch, daemon = True)
    th.start()
    obs["kern_build"] = run_py(ctx, R8 / "kern_build.py", ["--dest", dest, "--out", o, "--timeout", "2700"], o,
                               timeout = 2 * 2700 + 1800, tag = "kern_build")
    ctx["save"]()
    th.join(timeout = 7200)
    obs["kern_build"]["q21_prefetch"] = pre.get("r", {"rc": "still running after the build"})
    ctx["save"]()


def _kitchen(obs, name):
    b = ((obs.get("kern_build") or {}).get("builds") or {}).get(name) or {}
    return b.get("tree") if b.get("ok") else None


# ---------------------------------------------------------------- profile (C9-3 step 0, carbon9 #4)
TINY = ("import torch\nx = torch.randn(2048, 2048, device = 'cuda', dtype = torch.bfloat16)\n"
        "for _ in range(5):\n    y = x @ x\ntorch.cuda.synchronize()\nprint('TINY_OK')\n")


def _rocprof_cands(obs):
    seen, out = set(), []
    for r in (obs.get("kern_smoke") or {}).get("rocm_roots") or []:
        if r.get("rocprofv3"):
            out.append(os.path.join(r["root"], "bin", "rocprofv3"))
    w = ((obs.get("kern_smoke") or {}).get("which") or {}).get("rocprofv3")
    if w:
        out.append(w)
    return [p for p in out if not (os.path.realpath(p) in seen or seen.add(os.path.realpath(p)))]


def _traces(d):
    return sorted(glob.glob(os.path.join(d, "**", "*kernel_trace.csv"), recursive = True))


def _rocprof(ctx, tool, d, argv, timeout, tag, extra_env = None):
    shutil.rmtree(d, ignore_errors = True)
    os.makedirs(d, exist_ok = True)
    cmd = [tool, "--kernel-trace", "--output-format", "csv", "-d", d, "-o", "trace", "--", *argv]
    t = time.time()
    with open(ctx["log"], "ab") as fh:
        fh.write(f"\n=== [{tag}] {' '.join(map(str, cmd))}\n".encode())
        fh.flush()
        try:
            rc = subprocess.run(cmd, stdout = fh, stderr = subprocess.STDOUT, env = child_env(ctx, extra_env, tag),
                                timeout = timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        except Exception as exc:  # noqa: BLE001
            rc = repr(exc)[:200]
    return {"rc": rc, "wall_s": round(time.time() - t, 1), "traces": _traces(d)}


def _cat(name):
    n = name.lower()
    if "w8a8_mm" in n:
        return "w8a8_mm"
    if "rotq_rows" in n:
        return "rotq_rows"
    if any(s in n for s in ("attn", "attention", "fmha", "flash")):
        return "attention"
    if any(s in n for s in ("cijk_", "gemm", "matmul", "_mm_", "mm_kernel", "hipblaslt", "rocblas", "wmma")):
        return "gemm_other"
    return "other"


def _shares_from_csv(paths, windows):
    """Device time per category inside each [t0, t1] ns window (plus the whole trace)."""
    acc = [{"n": 0, "dev_ns": 0, "cat": {}, "top": {}} for _ in windows]
    whole = {"n": 0, "dev_ns": 0, "cat": {}}
    for p in paths:
        with open(p, newline = "", encoding = "utf-8", errors = "replace") as fh:
            rd = csv.DictReader(fh)
            for row in rd:
                try:
                    name = row.get("Kernel_Name") or row.get("KernelName") or ""
                    s, e = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
                except Exception:  # noqa: BLE001
                    continue
                dur, c = max(0, e - s), _cat(name)
                whole["n"] += 1
                whole["dev_ns"] += dur
                whole["cat"][c] = whole["cat"].get(c, 0) + dur
                for i, (t0, t1) in enumerate(windows):
                    if t0 <= s and e <= t1:
                        a = acc[i]
                        a["n"] += 1
                        a["dev_ns"] += dur
                        a["cat"][c] = a["cat"].get(c, 0) + dur
                        a["top"][name[:120]] = a["top"].get(name[:120], 0) + dur
    for a in acc + [whole]:
        tot = a["dev_ns"] or 1
        a["share"] = {c: round(v / tot, 4) for c, v in a["cat"].items()}
        if "top" in a:
            a["top"] = sorted(([k, round(v / 1e6, 2)] for k, v in a["top"].items()), key = lambda x: -x[1])[:25]
    return acc, whole


def _shares_from_kernels_json(p):
    d = json.loads(Path(p).read_text(encoding = "utf-8"))
    cat, tot = {}, 0.0
    for k in d.get("kernels") or []:
        c = _cat(k["name"])
        cat[c] = cat.get(c, 0.0) + k["device_us"]
        tot += k["device_us"]
    return {"render": d.get("render"), "wall_s": d.get("wall_s"), "dev_ms": round(tot / 1e3, 1),
            "share": {c: round(v / (tot or 1), 4) for c, v in cat.items()},
            "top": [[k["name"][:120], round(k["device_us"] / 1e3, 2)] for k in (d.get("kernels") or [])[:25]]}


def item_kern_prof(ctx, obs):
    res: dict = {"repo": Q21}
    obs["kern_prof"] = res
    if not _smoke_ok(obs):
        res["skipped"] = "kern_smoke failed"
        return
    res["prefetch"] = prefetch(ctx, Q21)
    ctx["save"]()
    tool = None
    res["rocprofv3_tests"] = []
    for cand in _rocprof_cands(obs):
        d = str(Path(ctx["scratch"]) / "rp_tiny")
        r = _rocprof(ctx, cand, d, [ctx["py"], "-c", TINY], 600, "rp_tiny")
        rows = 0
        for p in r["traces"]:
            with open(p, encoding = "utf-8", errors = "replace") as fh:
                rows += max(0, sum(1 for _ in fh) - 1)
        r.update(tool = cand, kernel_rows = rows)
        r.pop("traces", None)
        res["rocprofv3_tests"].append(r)
        ctx["save"]()
        if r["rc"] == 0 and rows > 0:
            tool = cand
            break
    res["rocprofv3"] = tool
    od = Path(ctx["work"]) / "e2e"
    od.mkdir(parents = True, exist_ok = True)
    tag = "q21_prof"
    args = ["--root", ctx["base_backend"], "--model", Q21, "--tag", tag, "--out", str(od), "--renders", "3",
            "--mark-times", "--gen", json.dumps(GEN)]
    if tool:
        d = str(Path(ctx["scratch"]) / "rp_q21")
        r = _rocprof(ctx, tool, d, [ctx["py"], "-u", str(R8 / "e2e_arm.py"), *args], 3 * 3600, tag)
        res["run"] = {k: v for k, v in r.items() if k != "traces"}
        rp = od / f"{tag}.json"
        rec = json.loads(rp.read_text(encoding = "utf-8")) if rp.is_file() else {"verdict": "missing"}
        res["arm"] = _arm_summary(rec)
        wins = [(x.get("t0_ns_boot"), x.get("t1_ns_boot")) for x in rec.get("renders") or []]
        wins_m = [(x.get("t0_ns_mono"), x.get("t1_ns_mono")) for x in rec.get("renders") or []]
        if r["traces"] and wins and all(a and b for a, b in wins):
            per, whole = _shares_from_csv(r["traces"], wins)
            if per and all(p["n"] == 0 for p in per) and all(a and b for a, b in wins_m):
                per, whole = _shares_from_csv(r["traces"], wins_m)
                res["clock"] = "monotonic"
            else:
                res["clock"] = "boottime"
            res["per_render"], res["whole_trace"] = per, whole
            res["source"] = "rocprofv3"
        ctx["save"]()
    if not res.get("per_render") or not any(p.get("n") for p in res["per_render"]):
        tag2 = "q21_tprof"
        rec = e2e(ctx, ctx["base_backend"], Q21, tag2, ["--renders", "3", "--torch-profile", "2", "--gen",
                                                         json.dumps(GEN)], timeout = 3 * 3600)
        res["arm_torchprof"] = _arm_summary(rec)
        kp = od / f"{tag2}_kernels.json"
        if kp.is_file():
            res["torch_profile"] = _shares_from_kernels_json(kp)
            res["source"] = "torch.profiler (rocprofv3 unusable or empty: see rocprofv3_tests)"
        ctx["save"]()
    sh = _last_share(res)
    res["int8_gemm_share"], res["rotq_share"] = (sh or {}).get("w8a8_mm"), (sh or {}).get("rotq_rows")
    res["all_gemm_share"] = None if sh is None else round(sh.get("w8a8_mm", 0) + sh.get("gemm_other", 0), 4)
    ctx["save"]()


def _last_share(res):
    per = [p for p in res.get("per_render") or [] if p.get("n")]
    if per:
        return per[-1]["share"]
    tp = res.get("torch_profile")
    return tp["share"] if tp else None


def _arm_summary(rec):
    keep = ("verdict", "error", "load_s", "rc", "wall_s")
    out = {k: rec.get(k) for k in keep if k in rec}
    out["renders"] = [{k: r.get(k) for k in ("r", "wall_s", "s_per_step", "md5", "rdna_mm", "speed_mode",
                                             "speed_optims", "transformer_quant", "t0_ns_boot", "t1_ns_boot")}
                      for r in rec.get("renders") or []]
    st = rec.get("status") or {}
    out["status"] = {k: st.get(k) for k in ("transformer_quant", "transformer_quant_backend", "speed_mode",
                                            "attention_backend", "offload_policy")}
    return out


# ---------------------------------------------------------------- A1 out pitch + GROUP_M (carbon6 #1)
def _a1_best(rows):
    """Per row: best contiguous GROUP_M ms, best pitch ms, ratio; geomean over the M 4096 / 4352 rows."""
    out = []
    for r in rows or []:
        res = r.get("res") or {}
        c = {k: v["med"] for k, v in res.items() if k.startswith("gm") and isinstance(v, dict) and v.get("med")}
        p = {k: v["med"] for k, v in res.items() if k.startswith("pad_gm") and isinstance(v, dict) and v.get("med")}
        if c and p:
            out.append({"shape": r["shape"], "m": r["m"], "best_gm": min(c, key = c.get), "best_pad": min(p, key = p.get),
                        "ratio": min(c.values()) / min(p.values()),
                        "exact": all(v.get("exact_vs_studio") for k, v in res.items() if isinstance(v, dict) and "med" in v)})
    return out


def item_kern_a1(ctx, obs):
    res: dict = {"procs": []}
    obs["kern_a1"] = res
    if not _smoke_ok(obs):
        res["skipped"] = "kern_smoke failed"
        return
    for i in range(3):
        o = Path(ctx["work"]) / f"kern_a1_{i}.json"
        res["procs"].append(run_py(ctx, R8 / "kern_gemm.py", ["a1", "--backend", ctx["base_backend"], "--out", o], o,
                                   timeout = 2400, tag = f"kern_a1_{i}"))
        ctx["save"]()
    from r8_common import gmean
    best = [_a1_best(p.get("rows")) for p in res["procs"]]
    res["best"] = best
    big = [x["ratio"] for b in best for x in b if x["m"] in (4096, 4352)]
    res["pad_over_best_gm_geomean_m4096"] = gmean(big)
    if not res["pad_over_best_gm_geomean_m4096"] or res["pad_over_best_gm_geomean_m4096"] < 1.05:
        res["e2e_skipped"] = "kernel: pitch N+128 under 1.05x over the best GROUP_M (kill line), no E2E"
        ctx["save"]()
        return
    prefetch(ctx, Q21)
    rows = []
    for i, arm in enumerate("MPPM"):
        tag = f"q21_pad_{arm}{i}"
        if arm == "M":
            rec = e2e(ctx, ctx["base_backend"], Q21, tag, ["--renders", "2", "--gen", json.dumps(GEN)], timeout = 5400)
        else:
            od = Path(ctx["work"]) / "e2e"
            od.mkdir(parents = True, exist_ok = True)
            rec = run_py(ctx, R8 / "kern_e2e_pad.py", ["--root", ctx["base_backend"], "--model", Q21, "--tag", tag,
                                                      "--out", od, "--renders", "2", "--gen", json.dumps(GEN)],
                         od / f"{tag}.json", timeout = 5400, tag = tag)
        s = _arm_summary(rec)
        s.update(arm = arm, i = i)
        rows.append(s)
        res["e2e"] = rows
        ctx["save"]()


# ---------------------------------------------------------------- buffer ops (round-7 n6#7)
def item_kern_bufops(ctx, obs):
    res: dict = {"procs": []}
    obs["kern_bufops"] = res
    if not _smoke_ok(obs):
        res["skipped"] = "kern_smoke failed"
        return
    for i, arm in enumerate("ONNONO"):
        o = Path(ctx["work"]) / f"kern_bufops_{i}_{arm}.json"
        r = run_py(ctx, R8 / "kern_gemm.py", ["bufops", "--backend", ctx["base_backend"], "--out", o], o,
                   timeout = 2400, tag = f"kern_bufops_{i}",
                   extra_env = {"AMDGCN_USE_BUFFER_OPS": "1" if arm == "N" else "0",
                                "TRITON_CACHE_DIR": str(Path(ctx["scratch"]) / f"triton_bufops_{arm}")})
        r["arm"] = arm
        res["procs"].append(r)
        ctx["save"]()


# ---------------------------------------------------------------- C9-3 step 1: #338 race
def item_kern_k338(ctx, obs):
    res: dict = {"procs": []}
    obs["kern_k338"] = res
    if not _smoke_ok(obs):
        res["skipped"] = "kern_smoke failed"
        return
    share = (obs.get("kern_prof") or {}).get("int8_gemm_share")
    res["int8_gemm_share"] = share
    if share is not None and share < SHARE_KILL:
        res["skipped"] = f"INT8 GEMM share {share:.3f} < {SHARE_KILL} of the Qwen-2.1 W8A8 render (C9-3 kill line)"
        ctx["save"]()
        return
    k338, kmain = _kitchen(obs, "k338"), _kitchen(obs, "kmain")
    if not k338:
        res["not_run"] = "comfy-kitchen #338 build failed (kern_build)"
        ctx["save"]()
        return
    order = ["K", "M", "M", "K", "K", "M"] if kmain else ["K", "K", "K"]
    for i, arm in enumerate(order):
        tree = k338 if arm == "K" else kmain
        o = Path(ctx["work"]) / f"kern_k338_{i}_{arm}.json"
        r = run_py(ctx, R8 / "kern_gemm.py", ["k338", "--backend", ctx["base_backend"], "--out", o, "--kitchen", tree,
                                             "--kname", "k338" if arm == "K" else "kmain"], o, timeout = 2400,
                   tag = f"kern_k338_{i}")
        r["arm"] = arm
        res["procs"].append(r)
        ctx["save"]()


# ---------------------------------------------------------------- M10: kitchen INT8 attention
def item_kern_attn(ctx, obs):
    res: dict = {"procs": []}
    obs["kern_attn"] = res
    if not _smoke_ok(obs):
        res["skipped"] = "kern_smoke failed"
        return
    tree = _kitchen(obs, "kmain") or _kitchen(obs, "k338")
    res["kitchen"] = tree
    if not tree:
        res["not_run"] = "no comfy-kitchen build (kern_build)"
        ctx["save"]()
        return
    for i, order in enumerate(("sk", "ks", "sk")):
        o = Path(ctx["work"]) / f"kern_attn_{i}.json"
        r = run_py(ctx, R8 / "kern_attn.py", ["--backend", ctx["base_backend"], "--out", o, "--kitchen", tree,
                                             "--order", order], o, timeout = 2400, tag = f"kern_attn_{i}")
        res["procs"].append(r)
        ctx["save"]()


ITEMS = {"kern_smoke": item_kern_smoke, "kern_build": item_kern_build, "kern_prof": item_kern_prof,
         "kern_a1": item_kern_a1, "kern_bufops": item_kern_bufops, "kern_k338": item_kern_k338,
         "kern_attn": item_kern_attn}
