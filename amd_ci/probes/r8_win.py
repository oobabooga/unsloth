#!/usr/bin/env python3
"""Round-8 Windows gfx1151 items (observes only; judged by criteria/r8_judge_win.py). Every arm is a fresh process.

  wsmoke        seconds: device / arch / HIP, mem_get_info vs the WDDM counter, one GEMM, the HIP DLL for ctypes
  sdma          C6-W2: #12571 sequence x50 per cell, PAL_DISABLE_SDMA unset (U) vs 1 (D), order U D D U
  sdma_renders  C6-W2 cost, ONLY if any U cell went wrong: Qwen-Image-2.1 streamed (memory_mode low_vram), U D D U U D
  memset        C15-2: 256 MiB zero_() vs hipMemsetAsync / hipMemsetD32Async, ballast 0 and 2 GiB, 2 processes each
  knee3         C6-W1 + C15 #4: knee points x 4 env arms (base / GPU_MAX_SUBALLOC_SIZE=65536 / PAL_HIP_IPC_FLAG=0 /
                both) x 3 reps, rotated order; plus operands-before-ballast at 40 GiB / 1 GiB
  knee_render   C6-W1 render gate, ONLY if an env arm beats the base spread at 3+ knee points: Qwen-Image-2.1 at
                Studio defaults, base vs that arm, B A A B B A
  memprof       C15-2: warm Wan 2.2 TI2V-5B render (unsloth mirror) 832x480x33 / 8 steps, then one whole render
                under torch.profiler: memset events / device ms / bytes vs the unprofiled warm render
  c153          C15-3 step 1, ONLY when the head tree carries the C15-3 patch (R8_C153=1 in the job env): Krea-2-Turbo
                load + renders logging driver free / total-reserved / trusted / WDDM used / speed mode, main (M) vs
                head (H) order M H H M M H
"""
from __future__ import annotations

import json
import os
import shutil
import statistics as st
from pathlib import Path

from r8_common import R8, e2e, prefetch, run_code, run_py, spread  # noqa: F401

QWEN = "unsloth/Qwen-Image-2.1"
WAN = "unsloth/Wan2.2-TI2V-5B-FP8"
KREA = "unsloth/Krea-2-Turbo"

WSMOKE = r"""
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
out = {}
import torch
p = torch.cuda.get_device_properties(0)
out.update(torch = torch.__version__, hip = getattr(torch.version, "hip", None), cuda = torch.cuda.is_available(),
           arch = getattr(p, "gcnArchName", None), total_gib = round(p.total_memory / 2**30, 3),
           integrated = bool(getattr(p, "is_integrated", getattr(p, "integrated", False))))
f, t = torch.cuda.mem_get_info(); out["free0_gib"] = round(f / 2**30, 3)
held = [torch.ones(2**30, dtype = torch.uint8, device = "cuda") for _ in range(4)]
torch.cuda.synchronize()
f, t = torch.cuda.mem_get_info(); out["free_4x1g_gib"] = round(f / 2**30, 3)
try:
    from utils.hardware import hardware as HW
    t0 = time.perf_counter(); u = HW._rocm_windows_unified_used_bytes(); out["wddm_s"] = round(time.perf_counter() - t0, 2)
    out["wddm_used_gib"] = None if u is None else round(float(u) / 2**30, 3)
    from utils.hardware import trusted_mem_get_info
    out["trusted_free_gib"] = round(trusted_mem_get_info()[0] / 2**30, 3)
except Exception as e:
    out["hw_error"] = repr(e)[:300]
del held
x = torch.randn(4096, 4096, device = "cuda", dtype = torch.bfloat16)
for _ in range(3): x @ x
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): x @ x
torch.cuda.synchronize(); out["gemm_tflops"] = round(20 * 2 * 4096**3 / (time.perf_counter() - t) / 1e12, 2)
libs = sorted({os.path.basename(m) for m in os.listdir(os.path.join(os.path.dirname(torch.__file__), "lib"))
               if "hip" in m.lower()})[:30]
out["torch_lib_hip"] = libs
print("R8JSON " + json.dumps(out), flush = True)
"""


def _hf_rm(ctx, repo):
    d = Path(ctx["hf"]) / "hub" / ("models--" + repo.replace("/", "--"))
    shutil.rmtree(d, ignore_errors = True)


def _hub_repos(ctx):
    hub = Path(ctx["hf"]) / "hub"
    return sorted(p.name for p in hub.glob("models--*")) if hub.is_dir() else []


def item_wsmoke(ctx, obs):
    rows = []
    for tree in sorted({ctx["main_backend"], ctx["backend"]}):
        r = run_code(ctx, WSMOKE, [tree], timeout = 600, tag = "wsmoke")
        r["tree"] = ("main" if tree == ctx["main_backend"] else "") + ("head" if tree == ctx["backend"] else "base")
        rows.append(r)
        obs["wsmoke"] = rows
        ctx["save"]()
    obs["wsmoke_ok"] = all(r.get("cuda") and str(r.get("arch") or "").startswith("gfx1151") for r in rows)
    if not obs["wsmoke_ok"]:
        raise RuntimeError("wsmoke failed: no gfx1151 device")


def item_sdma(ctx, obs):
    rows = []
    for i, arm in enumerate("UDDU"):
        o = Path(ctx["work"]) / f"sdma_{i}_{arm}.json"
        env = {"PAL_DISABLE_SDMA": "1" if arm == "D" else None}
        r = run_py(ctx, R8 / "win_sdma.py", ["--out", o, "--iters", "50"], o, timeout = 1200, extra_env = env,
                   tag = f"sdma{i}")
        r.update(arm = arm, i = i)
        rows.append(r)
        obs["sdma"] = rows
        ctx["save"]()


def _sdma_wrong_unset(obs) -> int:
    n = 0
    for r in obs.get("sdma") or []:
        if r.get("arm") != "U":
            continue
        for cell, c in (r.get("cells") or {}).items():
            if not cell.endswith("-norec"):
                n += int(c.get("wrong") or 0)
    return n


def _arm(ctx, tree, repo, kind, tag, load, gen, warmup, prompts, extra_env = None, extra_args = None, timeout = 5400):
    od = Path(ctx["work"]) / "arms"
    od.mkdir(parents = True, exist_ok = True)
    args = ["--root", tree, "--kind", kind, "--model", repo, "--tag", tag, "--out", od, "--load", json.dumps(load),
            "--gen", json.dumps(gen), "--prompts", str(prompts), "--warmup", str(warmup), "--save-frames-every", "16",
            *(extra_args or [])]
    rec = run_py(ctx, R8 / "win_arm.py", args, od / tag / "record.json", timeout = timeout, extra_env = extra_env, tag = tag)
    for p in (od / tag).glob("r*.npz"):
        p.unlink(missing_ok = True)  # frames: md5 is in the record; keep the artifact small
    rec.pop("i8_log", None)
    return rec


def item_sdma_renders(ctx, obs):
    wrong = _sdma_wrong_unset(obs)
    obs["sdma_renders_gate"] = {"wrong_unset": wrong}
    if wrong == 0:
        obs["sdma_renders_skip"] = "0 wrong with PAL_DISABLE_SDMA unset: kill line hit (drop), no cost renders"
        return
    obs["sdma_renders_prefetch"] = prefetch(ctx, QWEN)
    rows = []
    for i, arm in enumerate("UDDUUD"):
        env = {"PAL_DISABLE_SDMA": "1" if arm == "D" else None}
        r = _arm(ctx, ctx["main_backend"], QWEN, "image", f"sdmar_{i}_{arm}", {"model_kind": "pipeline",
                 "memory_mode": "low_vram"}, {"width": 1024, "height": 1024}, 1, 2, extra_env = env)
        r.update(arm = arm, i = i)
        rows.append(r)
        obs["sdma_renders"] = rows
        ctx["save"]()


def item_memset(ctx, obs):
    rows = []
    for i, gib in enumerate((2, 0, 0, 2)):
        o = Path(ctx["work"]) / f"memset_{i}_{gib}.json"
        r = run_py(ctx, R8 / "win_memset.py", ["--out", o, "--ballast-gib", gib], o, timeout = 600, tag = f"memset{i}")
        r["i"] = i
        rows.append(r)
        obs["memset"] = rows
        ctx["save"]()


KNEE_ARMS = {"base": {}, "sub": {"GPU_MAX_SUBALLOC_SIZE": "65536"}, "ipc": {"PAL_HIP_IPC_FLAG": "0"},
             "both": {"GPU_MAX_SUBALLOC_SIZE": "65536", "PAL_HIP_IPC_FLAG": "0"}}
KNEE_POINTS = [(24, 24 * 2 ** 30, False), (24, 2 ** 30, False), (24, 64 * 2 ** 20, False), (24, 8 * 2 ** 20, False),
               (40, 40 * 2 ** 30, False), (40, 2 ** 30, False), (40, 64 * 2 ** 20, False), (40, 8 * 2 ** 20, False),
               (40, 2 ** 30, True)]


def item_knee3(ctx, obs):
    rows = []
    names = list(KNEE_ARMS)
    n = 0
    for rep in range(3):
        for pi, (gib, chunk, opfirst) in enumerate(KNEE_POINTS):
            order = names[(rep + pi) % 4:] + names[:(rep + pi) % 4]
            for arm in order:
                o = Path(ctx["work"]) / "knee" / f"k_{rep}_{pi}_{arm}.json"
                o.parent.mkdir(parents = True, exist_ok = True)
                args = ["--out", o, "--gib", gib, "--chunk", chunk] + (["--opfirst"] if opfirst else [])
                r = run_py(ctx, R8 / "win_knee.py", args, o, timeout = 900, extra_env = KNEE_ARMS[arm] or None,
                           tag = "knee")
                r.update(rep = rep, point = pi, arm = arm)
                r.pop("counters_gib", None) if arm != "base" else None
                rows.append(r)
                n += 1
                if n % 9 == 0:
                    obs["knee3"] = rows
                    ctx["save"]()
    obs["knee3"] = rows
    ctx["save"]()


def knee_summary(rows) -> dict:
    """point -> arm -> {med, spread_abs, vals}; and per arm the points it wins (median above the base max and 5% or
    more above the base median)."""
    by: dict = {}
    for r in rows or []:
        if r.get("gemm_tflops"):
            by.setdefault(r["point"], {}).setdefault(r["arm"], []).append(r["gemm_tflops"])
    summ, wins = {}, {a: [] for a in KNEE_ARMS if a != "base"}
    for p, arms in sorted(by.items()):
        summ[p] = {a: {"med": st.median(v), "min": min(v), "max": max(v), "n": len(v)} for a, v in arms.items()}
        b = summ[p].get("base")
        if not b:
            continue
        for a in wins:
            s = summ[p].get(a)
            if s and s["med"] > b["max"] and s["med"] >= 1.05 * b["med"]:
                wins[a].append(p)
    return {"points": summ, "wins": wins}


def item_knee_render(ctx, obs):
    ks = knee_summary(obs.get("knee3"))
    best = max(ks["wins"].items(), key = lambda kv: len(kv[1]), default = (None, []))
    obs["knee_render_gate"] = {"wins": ks["wins"], "pick": best[0] if len(best[1]) >= 3 else None}
    if len(best[1]) < 3:
        obs["knee_render_skip"] = "no env arm beats the base spread at 3+ knee points: kill line hit (inside base spread)"
        return
    obs["knee_render_prefetch"] = prefetch(ctx, QWEN)
    rows = []
    for i, arm in enumerate("BAABBA"):
        env = KNEE_ARMS[best[0]] if arm == "A" else None
        r = _arm(ctx, ctx["main_backend"], QWEN, "image", f"kneer_{i}_{arm}", {"model_kind": "pipeline"},
                 {"width": 1024, "height": 1024}, 1, 3, extra_env = env)
        r.update(arm = arm, i = i, env_arm = best[0])
        rows.append(r)
        obs["knee_render"] = rows
        ctx["save"]()


def item_memprof(ctx, obs):
    _hf_rm(ctx, QWEN)  # the Qwen arms are done: keep the box's disk for Wan
    obs["memprof_prefetch"] = prefetch(ctx, WAN)
    trace = Path(ctx["scratch"]) / "wan_prof_trace.json"
    r = _arm(ctx, ctx["main_backend"], WAN, "video", "memprof_wan", {}, {"width": 832, "height": 480, "num_frames": 33,
             "steps": 8}, 1, 1, extra_args = ["--prof-whole", str(trace)], timeout = 4800)
    obs["memprof"] = r
    obs["memprof_hub"] = _hub_repos(ctx)
    ctx["save"]()
    _hf_rm(ctx, WAN)


def item_c153(ctx, obs):
    if os.environ.get("R8_C153") != "1":
        obs["c153_skip"] = "no C15-3 patch in this job (step 0 decided)"
        return
    obs["c153_prefetch"] = prefetch(ctx, KREA)
    rows = []
    env = {"R8_MEMLOG_RENDERS": "1"}
    for i, arm in enumerate("MHHMMH"):
        tree = ctx["base_backend"] if arm == "M" else ctx["backend"]
        r = _arm(ctx, tree, KREA, "image", f"c153_{i}_{arm}", {"model_kind": "pipeline"},
                 {"width": 1024, "height": 1024, "steps": 8}, 0, 3, extra_env = env, timeout = 3 * 3600)
        r.update(arm = arm, i = i)
        rows.append(r)
        obs["c153"] = rows
        ctx["save"]()
    obs["c153_hub"] = _hub_repos(ctx)


ITEMS = {"wsmoke": item_wsmoke, "sdma": item_sdma, "memset": item_memset, "knee3": item_knee3,
         "sdma_renders": item_sdma_renders, "knee_render": item_knee_render, "memprof": item_memprof,
         "c153": item_c153}
