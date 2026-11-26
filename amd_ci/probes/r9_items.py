#!/usr/bin/env python3
"""Round-9 gfx1151 items, Linux and Windows (observes only; judged by criteria/r9_judge.py). Every arm is a fresh
process. Small items first so a crash in a later one cannot lose them.

  smoke      seconds: device / arch / HIP, one bf16 GEMM. Fails the job's later items if no gfx1151.
  c63        C6-3: fp32 Conv3d patch-embed (N,3,2,16,16), N in {112,113,4032,5600,16384}, one process per N, x2 sweeps
             (native crash = child rc); then the patch-embed share of a Qwen3-VL-4B vision encode (config only,
             random weights) at N 4032 / 5600 / 16384.
  m4         M4 (Linux): FLUX.3 klein VAE reference encode x5, arms U (flags as Studio) / D (scoped
             deterministic=True, benchmark=False), processes U D D U U D, bf16 then fp32 (U D only).
  c61_knee   C6-1 (Windows): 42 GiB ballast knee, arms base / opfirst / reserve 2 / 4 / 8 GiB, 3 reps, rotated order.
  c61_krea   C6-1 render gate (Windows), ONLY if some reserve N has median GEMM >= 29 TFLOP/s after the ballast:
             Krea-2-Turbo 1024^2 at Studio defaults, A (no reserve) vs B (R9_RESERVE_GIB = smallest passing N),
             order A B B A, 1 warm-up + 3 renders each.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from r9_common import R9, prefetch, run_code, run_py

KREA = "unsloth/Krea-2-Turbo"
KLEIN = "unsloth/flux-3-image-klein"
QWEN3VL = "unsloth/Qwen3-VL-4B-Instruct"
C63_N = (112, 113, 4032, 5600, 16384)
KNEE_TFLOPS_KILL = 29.0

SMOKE = r"""
import json, time, torch
p = torch.cuda.get_device_properties(0)
out = dict(torch = torch.__version__, hip = getattr(torch.version, "hip", None), cuda = torch.cuda.is_available(),
           arch = getattr(p, "gcnArchName", None), total_gib = round(p.total_memory / 2**30, 3))
x = torch.randn(4096, 4096, device = "cuda", dtype = torch.bfloat16)
for _ in range(3): x @ x
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): x @ x
torch.cuda.synchronize(); out["gemm_tflops"] = round(20 * 2 * 4096**3 / (time.perf_counter() - t) / 1e12, 2)
print("R9JSON " + json.dumps(out), flush = True)
"""


def item_smoke(ctx, obs):
    r = run_code(ctx, SMOKE, [], timeout = 600, tag = "smoke")
    obs["smoke"] = r
    obs["smoke_ok"] = bool(r.get("cuda")) and str(r.get("arch") or "").startswith("gfx1151")
    ctx["save"]()
    if not obs["smoke_ok"]:
        raise RuntimeError("smoke failed: no gfx1151 device")


def item_c63(ctx, obs):
    res: dict = {"crash": [], "share": []}
    obs["c63"] = res
    od = Path(ctx["work"]) / "c63"
    od.mkdir(parents = True, exist_ok = True)
    for sweep in range(2):
        for n in (C63_N if sweep == 0 else C63_N[::-1]):
            o = od / f"crash_{sweep}_{n}.json"
            r = run_py(ctx, R9 / "c63_conv3d.py", ["crash", "--n", n, "--out", o], o, timeout = 900, tag = f"c63_{n}")
            r.update(sweep = sweep, n = n)
            res["crash"].append(r)
            ctx["save"]()
    pf = prefetch(ctx, QWEN3VL, ["config.json"], timeout = 600)
    res["config_prefetch"] = {k: pf.get(k) for k in ("rc", "path", "missing", "tail")}
    if not pf.get("path"):
        res["share_not_run"] = "Qwen3-VL config prefetch failed"
        ctx["save"]()
        return
    for n in (4032, 5600, 16384):
        o = od / f"share_{n}.json"
        r = run_py(ctx, R9 / "c63_conv3d.py", ["share", "--n", n, "--config", pf["path"], "--out", o], o,
                   timeout = 1200, tag = f"c63s_{n}")
        res["share"].append(r)
        ctx["save"]()


def item_m4(ctx, obs):
    res: dict = {"procs": []}
    obs["m4"] = res
    pf = prefetch(ctx, KLEIN, ["vae/*"], timeout = 3600)
    res["prefetch"] = {k: pf.get(k) for k in ("rc", "path", "missing", "tail", "wall_s")}
    if not pf.get("path"):
        res["not_run"] = "klein VAE prefetch failed"
        ctx["save"]()
        return
    od = Path(ctx["work"]) / "m4"
    od.mkdir(parents = True, exist_ok = True)
    plan = [("bf16", x) for x in "UDDUUD"] + [("fp32", x) for x in "UD"]
    for i, (dt, arm) in enumerate(plan):
        o = od / f"m4_{i}_{dt}_{arm}.json"
        r = run_py(ctx, R9 / "m4_vae_encode.py", ["--repo-dir", pf["path"], "--arm", arm, "--dtype", dt, "--reps", 5,
                                                  "--out", o], o, timeout = 1800, tag = f"m4_{i}")
        r.update(i = i, arm = arm, dtype = dt)
        res["procs"].append(r)
        ctx["save"]()


KNEE_ARMS = {"base": [], "opfirst": ["--opfirst"], "r2": ["--reserve-gib", "2"], "r4": ["--reserve-gib", "4"],
             "r8": ["--reserve-gib", "8"]}


def item_c61_knee(ctx, obs):
    rows: list = []
    obs["c61_knee"] = rows
    names = list(KNEE_ARMS)
    for rep in range(3):
        k = rep % len(names)
        for arm in names[k:] + names[:k]:
            o = Path(ctx["work"]) / "knee" / f"k_{rep}_{arm}.json"
            o.parent.mkdir(parents = True, exist_ok = True)
            r = run_py(ctx, R9 / "win_knee_reserve.py", ["--out", o, "--gib", 42, *KNEE_ARMS[arm]], o, timeout = 900,
                       tag = "knee")
            r.update(rep = rep, arm = arm)
            rows.append(r)
            ctx["save"]()


def knee_summary(rows) -> dict:
    by: dict = {}
    for r in rows or []:
        if r.get("gemm_tflops"):
            by.setdefault(r["arm"], []).append(r["gemm_tflops"])
    return {a: {"med": st.median(v), "min": min(v), "max": max(v), "n": len(v), "vals": v} for a, v in by.items()}


def knee_pick(rows):
    """The smallest reserve whose median GEMM after the ballast clears the kill line, with every rep engaged
    (the GEMM operand inside the freed reserve)."""
    s = knee_summary(rows)
    for arm, gib in (("r2", 2), ("r4", 4), ("r8", 8)):
        eng = [((r.get("in_reserve") or {}).get("gemm_x")) for r in rows if r.get("arm") == arm]
        if s.get(arm) and s[arm]["med"] >= KNEE_TFLOPS_KILL and eng and all(eng):
            return arm, gib
    return None, None


def _arm(ctx, tree, repo, tag, load, gen, warmup, prompts, extra_env = None, timeout = 5400):
    od = Path(ctx["work"]) / "arms"
    od.mkdir(parents = True, exist_ok = True)
    args = ["--root", tree, "--kind", "image", "--model", repo, "--tag", tag, "--out", od, "--load", json.dumps(load),
            "--gen", json.dumps(gen), "--prompts", str(prompts), "--warmup", str(warmup), "--save-frames-every", "16"]
    rec = run_py(ctx, R9 / "win_arm.py", args, od / tag / "record.json", timeout = timeout, extra_env = extra_env,
                 tag = tag)
    for p in (od / tag).glob("r*.npz"):
        p.unlink(missing_ok = True)
    rec.pop("i8_log", None)
    return rec


def item_c61_krea(ctx, obs):
    arm, gib = knee_pick(obs.get("c61_knee"))
    obs["c61_krea_gate"] = {"pick": arm, "gib": gib, "summary": knee_summary(obs.get("c61_knee"))}
    if arm is None:
        obs["c61_krea_skip"] = f"no reserve N reaches {KNEE_TFLOPS_KILL} TFLOP/s after the 42 GiB ballast: kill line hit"
        ctx["save"]()
        return
    obs["c61_krea_prefetch"] = {k: v for k, v in prefetch(ctx, KREA).items() if k in ("rc", "path", "wall_s", "tail")}
    rows = []
    for i, a in enumerate("ABBA"):
        env = {"R9_RESERVE_GIB": str(gib)} if a == "B" else {"R9_RESERVE_GIB": None}
        r = _arm(ctx, ctx["main_backend"], KREA, f"krea_{i}_{a}", {"model_kind": "pipeline"},
                 {"width": 1024, "height": 1024}, 1, 3, extra_env = env, timeout = 2 * 3600)
        r.update(arm = a, i = i)
        rows.append(r)
        obs["c61_krea"] = rows
        ctx["save"]()


ITEMS = {"smoke": item_smoke, "c63": item_c63, "m4": item_m4, "c61_knee": item_c61_knee, "c61_krea": item_c61_krea}
