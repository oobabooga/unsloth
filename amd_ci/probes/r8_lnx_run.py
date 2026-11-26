#!/usr/bin/env python3
"""Round-8 Linux gfx1151 render / training / host items (observes only; judged by criteria/r8_judge_run.py).

ITEMS (run order; cheap / correctness first, the hang-risk decode last):
  c156_spin      C15-6  idle process CPU after a sync, HSA_TOOLS_DISABLE_REGISTER unset vs 1, A B B A A B; only if
                        it spins (median unset >= 5% of a core): Wan 2.2 TI2V-5B renders A B B A (5 timed each) and
                        a rocprofv3 --kernel-trace check with the variable set (and unset, the control)
  m8b_mmap       M8 b   native-cache mapping (ACCESS_COPY, main) vs ACCESS_READ: 64 MiB / 1 GiB micro uploads,
                        smaps_rollup Anonymous / Private_Dirty delta; only if not VOID: Qwen-Image-2.1 cold Studio
                        load C R R C C R (load s, RssAnon peak, map calls)
  m8a_wankitchen M8 a   Wan 2.2 TI2V-5B VAE decode 1280x704x49f untiled: stock vs comfy-kitchen channel-first ops,
                        S K K S S K + one fp32 stock reference; PSNR vs stock bf16 and vs fp32
  m8c_h3enc      M8 c   MiniMax-H3 VAE 1024 px single-image encode: stock vs Conv3d -> Conv2d last-tap lowering,
                        S L L S S L + fp32 stock reference; PSNR of the posterior mean
  c186a_train    C18-6A Qwen-Image-2.1 LoRA 30 steps 512 px: bf16 / nf4 on main, int8 W8A8 on the head tree with
                        UNSLOTH_DIFFUSION_ROCM_W8A8_TRAIN=1; B42 I42 N42 N43 I43 B43
  m8d_qvae       M8 d   Qwen-Image-2.1-Turbo VAE untiled decode 1024^2 x5 then 1280^2 x5, one process each, 600 s
                        timeout = hang; a GPU health check after any timeout (last: a hang may wedge the device)
Weights: public unsloth/* repos only (token-free). Big files stay in ctx['scratch'].
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from r8_common import R8, child_env, prefetch, run_code, run_py

SPIN_VAR = "HSA_TOOLS_DISABLE_REGISTER"
SPIN_KILL = 0.05  # core fraction
WAN_REPO = "unsloth/Wan2.2-TI2V-5B-FP8"
WAN_DIFFUSERS = ["model_index.json", "scheduler/*", "transformer/*", "text_encoder/*", "tokenizer/*", "vae/*"]
Q21_REPO = "unsloth/Qwen-Image-2.1"
QT_REPO = "unsloth/Qwen-Image-2.1-Turbo-FP8"
H3_REPO = "unsloth/MiniMax-H3"


def _disk_free_gib(p) -> float:
    try:
        return round(shutil.disk_usage(str(p)).free / 2 ** 30, 1)
    except Exception:  # noqa: BLE001
        return -1.0


def _snap(ctx, obs, key, repo, patterns):
    r = prefetch(ctx, repo, patterns)
    obs.setdefault("r8run_prefetch", []).append({k: r.get(k) for k in ("repo", "rc", "wall_s", "path", "tail")})
    ctx["save"]()
    return r.get("path") if r.get("rc") == 0 else None


def _psnr(ctx, a, b, peak = "range"):
    """PSNR of two saved tensors (fp16 / fp32 .pt) in a child; peak 'range' = ref max - min, else a float."""
    code = ("import json, math, sys, torch\n"
            "a = torch.load(sys.argv[1]).float(); b = torch.load(sys.argv[2]).float()\n"
            "if a.shape != b.shape: print('R8JSON ' + json.dumps({'shape_mismatch': [list(a.shape), list(b.shape)]}))"
            "; raise SystemExit\n"
            "mse = float(((a - b) ** 2).mean())\n"
            "pk = float(a.max() - a.min()) if sys.argv[3] == 'range' else float(sys.argv[3])\n"
            "print('R8JSON ' + json.dumps({'psnr': (float('inf') if mse == 0 else 10 * math.log10(pk * pk / mse)),"
            " 'equal': bool(torch.equal(a, b)), 'maxabs': float((a - b).abs().max())}))\n")
    r = run_code(ctx, code, [a, b, peak], timeout = 900, tag = "psnr")
    return {k: r.get(k) for k in ("psnr", "equal", "maxabs", "shape_mismatch", "rc", "missing", "tail")}


# ---------------------------------------------------------------------------------------------------------- C15-6
def item_c156_spin(ctx, obs):
    rows = []
    obs["c156_spin"] = {"rows": rows}
    for i, arm in enumerate("ABBAAB"):
        o = Path(ctx["work"]) / f"spin_{i}_{arm}.json"
        env = {SPIN_VAR: "1"} if arm == "B" else {SPIN_VAR: None}
        r = run_py(ctx, R8 / "run_idle_spin.py", ["--out", o, "--idle", "10"], o, timeout = 300, extra_env = env,
                   tag = f"spin{i}")
        r.update(arm = arm, i = i)
        rows.append(r)
        ctx["save"]()
    a = sorted(r["core_frac"] for r in rows if r["arm"] == "A" and r.get("core_frac") is not None)
    med = a[len(a) // 2] if a else None
    spins = med is not None and med >= SPIN_KILL
    obs["c156_spin"].update(median_unset = med, spins = spins)
    ctx["save"]()
    if not spins:
        obs["c156_spin"]["renders_skipped"] = "no spin: median unset under 5% of a core (kill line), no renders"
        ctx["save"]()
        return
    path = _snap(ctx, obs, "wan", WAN_REPO, WAN_DIFFUSERS)
    arm_py = Path(__file__).resolve().parent / "wintiers" / "arm.py"
    renders = []
    obs["c156_spin"]["renders"] = renders
    if path and arm_py.is_file():
        od = Path(ctx["scratch"]) / "spin_wan"
        for i, arm in enumerate("ABBA"):
            env = {SPIN_VAR: "1"} if arm == "B" else {SPIN_VAR: None}
            tag = f"wan_{arm}_{i}"
            r = run_py(ctx, arm_py, ["--root", ctx["base_backend"], "--kind", "video", "--model", WAN_REPO, "--tag", tag,
                                     "--out", od, "--gen", json.dumps({"width": 832, "height": 480, "num_frames": 33,
                                                                       "steps": 8}),
                                     "--prompts", "5", "--warmup", "1"], od / tag / "record.json", timeout = 5400,
                       extra_env = env, tag = tag)
            r = {k: r.get(k) for k in ("rc", "wall_s", "verdict", "load_s", "renders", "error")}
            r.update(arm = arm, i = i)
            renders.append(r)
            ctx["save"]()
    else:
        obs["c156_spin"]["renders_skipped"] = f"wan snapshot {bool(path)}, arm.py {arm_py.is_file()}"
    prof = shutil.which("rocprofv3") or ("/opt/rocm/bin/rocprofv3" if Path("/opt/rocm/bin/rocprofv3").is_file() else None)
    checks = []
    obs["c156_spin"]["rocprof"] = {"bin": prof, "checks": checks}
    if prof:
        for arm in ("A", "B"):
            d = Path(ctx["scratch"]) / f"rocprof_{arm}"
            shutil.rmtree(d, ignore_errors = True)
            env = child_env(ctx, {SPIN_VAR: "1"} if arm == "B" else {SPIN_VAR: None}, tag = "rocprof")
            o = Path(ctx["work"]) / f"rocprof_{arm}.json"
            try:
                p = subprocess.run([prof, "--kernel-trace", "--output-format", "csv", "-d", str(d), "-o", "r8", "--",
                                    ctx["py"], str(R8 / "run_rocprof_check.py"), "--out", str(o)], env = env,
                                   capture_output = True, text = True, timeout = 600)
                rc, tail = p.returncode, (p.stdout + p.stderr)[-1200:]
            except subprocess.TimeoutExpired:
                rc, tail = "timeout", ""
            rows_n = 0
            for f in d.rglob("*kernel_trace*.csv") if d.is_dir() else []:
                rows_n += max(0, sum(1 for _ in open(f, encoding = "utf-8", errors = "replace")) - 1)
            checks.append({"arm": arm, "rc": rc, "kernel_rows": rows_n, "tail": tail})
            ctx["save"]()


# ---------------------------------------------------------------------------------------------------------- M8 (b)
def item_m8b_mmap(ctx, obs):
    d = Path(ctx["scratch"]) / "cow"
    d.mkdir(parents = True, exist_ok = True)
    rows = []
    obs["m8b_mmap"] = {"micro": rows}
    plan = [("copy", 64, 0), ("read", 64, 0), ("read", 64, 0), ("copy", 64, 0), ("copy", 64, 1), ("read", 64, 1),
            ("copy", 1024, 0), ("read", 1024, 0)]
    for i, (arm, mib, nb) in enumerate(plan):
        o = Path(ctx["work"]) / f"cow_{i}_{arm}_{mib}_{nb}.json"
        r = run_py(ctx, R8 / "run_mmap_cow.py", ["--backend", ctx["base_backend"], "--dir", d, "--arm", arm, "--out", o,
                                                  "--mib", mib, "--tensor-mib", 8, "--nb", nb], o, timeout = 900,
                   tag = f"cow{i}")
        r.update(i = i)
        rows.append(r)
        ctx["save"]()
    shutil.rmtree(d, ignore_errors = True)
    base = [r.get("frac_Anonymous") for r in rows if r.get("arm") == "copy" and r.get("ok")]
    frac = max(base) if base else None
    # 1 GiB-equivalent on #17751's 19.43 GiB H3 mapping: a dirty fraction under 1 / 19.43 is VOID (UMA does not repro)
    void = frac is not None and frac * 19.43 < 1.0
    obs["m8b_mmap"].update(copy_anon_frac_max = frac, void = void)
    ctx["save"]()
    if frac is None:
        obs["m8b_mmap"]["load_skipped"] = "no ACCESS_COPY micro arm finished"
        ctx["save"]()
        return
    if void:
        obs["m8b_mmap"]["load_skipped"] = "VOID: ACCESS_COPY upload dirtied under the 1 GiB-equivalent, no load A/B"
        ctx["save"]()
        return
    if not _snap(ctx, obs, "q21", Q21_REPO, None):
        obs["m8b_mmap"]["load_skipped"] = "Qwen-Image-2.1 snapshot failed"
        return
    loads = []
    obs["m8b_mmap"]["loads"] = loads
    od = Path(ctx["work"]) / "e2e"
    od.mkdir(parents = True, exist_ok = True)
    for i, arm in enumerate("CRRCCR"):
        access = "copy" if arm == "C" else "read"
        tag = f"q21map_{access}_{i}"
        side = Path(ctx["work"]) / f"{tag}_side.json"
        r = run_py(ctx, R8 / "run_mmap_load.py", ["--root", ctx["base_backend"], "--access", access, "--side-out", side,
                                                   "--", "--model", Q21_REPO, "--tag", tag, "--out", od, "--renders", "0",
                                                   "--cold"], od / f"{tag}.json", timeout = 3600, tag = tag)
        keep = {k: r.get(k) for k in ("rc", "wall_s", "verdict", "load_s", "load_vmhwm_gib", "load_rss_gib", "error")}
        try:
            keep["side"] = json.loads(side.read_text(encoding = "utf-8"))
        except Exception as exc:  # noqa: BLE001
            keep["side_error"] = repr(exc)[:200]
        keep.update(arm = access, i = i)
        loads.append(keep)
        ctx["save"]()


# ---------------------------------------------------------------------------------------------------------- M8 (a)
def item_m8a_wankitchen(ctx, obs):
    path = _snap(ctx, obs, "wanvae", WAN_REPO, ["vae/*", "model_index.json"])
    res = {"rows": []}
    obs["m8a_wankitchen"] = res
    if not path:
        res["skipped"] = "vae snapshot failed"
        return
    pts = {}
    plan = [("stock", 0), ("kitchen", 1), ("kitchen", 2), ("stock", 3), ("stock", 4), ("kitchen", 5)]
    for arm, i in plan:
        o = Path(ctx["work"]) / f"wank_{i}_{arm}.json"
        save = Path(ctx["scratch"]) / f"wank_{i}_{arm}.pt"
        args = ["--backend", ctx["base_backend"], "--dir", path, "--arm", arm, "--out", o]
        if i in (0, 1, 3):
            args += ["--save", save]
            pts[i] = str(save)
        r = run_py(ctx, R8 / "run_wan_vae_kitchen.py", args, o, timeout = 2400, tag = f"wank{i}")
        r.pop("tb", None) if not r.get("error") else None
        r.update(i = i)
        res["rows"].append(r)
        ctx["save"]()
    o = Path(ctx["work"]) / "wank_fp32.json"
    ref = Path(ctx["scratch"]) / "wank_fp32.pt"
    r = run_py(ctx, R8 / "run_wan_vae_kitchen.py", ["--backend", ctx["base_backend"], "--dir", path, "--arm", "stock",
                                                     "--dtype", "fp32", "--out", o, "--save", ref], o, timeout = 3600,
               tag = "wankf")
    res["fp32_ref"] = r
    ctx["save"]()
    q = {}
    if 0 in pts and 1 in pts:
        q["kitchen_vs_stock"] = _psnr(ctx, pts[0], pts[1], "2.0")
    if 0 in pts and 3 in pts:
        q["stock_aa"] = _psnr(ctx, pts[0], pts[3], "2.0")
    if ref.is_file():
        for name, i in (("stock_vs_fp32", 0), ("kitchen_vs_fp32", 1)):
            if i in pts:
                q[name] = _psnr(ctx, str(ref), pts[i], "2.0")
    res["psnr"] = q
    ctx["save"]()
    for p in list(pts.values()) + [str(ref)]:
        Path(p).unlink(missing_ok = True)


# ---------------------------------------------------------------------------------------------------------- M8 (c)
def item_m8c_h3enc(ctx, obs):
    path = _snap(ctx, obs, "h3vae", H3_REPO, ["vae/*", "model_index.json"])
    res = {"rows": []}
    obs["m8c_h3enc"] = res
    if not path:
        res["skipped"] = "unsloth/MiniMax-H3 vae snapshot failed"
        return
    pts = {}
    for arm, i in (("stock", 0), ("lowered", 1), ("lowered", 2), ("stock", 3), ("stock", 4), ("lowered", 5)):
        o = Path(ctx["work"]) / f"h3e_{i}_{arm}.json"
        save = Path(ctx["scratch"]) / f"h3e_{i}_{arm}.pt"
        args = ["--backend", ctx["base_backend"], "--dir", path, "--arm", arm, "--out", o]
        if i in (0, 1, 3):
            args += ["--save", save]
            pts[i] = str(save)
        r = run_py(ctx, R8 / "run_h3_ref_encode.py", args, o, timeout = 2400, tag = f"h3e{i}")
        r.update(i = i)
        res["rows"].append(r)
        ctx["save"]()
    o = Path(ctx["work"]) / "h3e_fp32.json"
    ref = Path(ctx["scratch"]) / "h3e_fp32.pt"
    res["fp32_ref"] = run_py(ctx, R8 / "run_h3_ref_encode.py", ["--backend", ctx["base_backend"], "--dir", path, "--arm",
                                                                 "stock", "--dtype", "fp32", "--out", o, "--save", ref],
                             o, timeout = 3600, tag = "h3ef")
    q = {}
    if 0 in pts and 1 in pts:
        q["lowered_vs_stock"] = _psnr(ctx, pts[0], pts[1])
    if 0 in pts and 3 in pts:
        q["stock_aa"] = _psnr(ctx, pts[0], pts[3])
    if ref.is_file():
        for name, i in (("stock_vs_fp32", 0), ("lowered_vs_fp32", 1)):
            if i in pts:
                q[name] = _psnr(ctx, str(ref), pts[i])
    res["psnr"] = q
    ctx["save"]()


# ---------------------------------------------------------------------------------------------------------- C18-6A
def item_c186a_train(ctx, obs):
    res = {"rows": []}
    obs["c186a_train"] = res
    res["disk_free_gib"] = _disk_free_gib(ctx["hf"])
    if not _snap(ctx, obs, "q21", Q21_REPO, None):
        res["skipped"] = "Qwen-Image-2.1 snapshot failed"
        return
    data = Path(ctx["scratch"]) / "train_data"
    run_py(ctx, R8 / "run_train.py", ["--make-data", data, "--n", 8, "--resolution", 512], None, timeout = 300,
           tag = "mkdata")
    work = Path(ctx["scratch"]) / "train_runs"
    opt = {"UNSLOTH_DIFFUSION_ROCM_W8A8_TRAIN": "1", "UNSLOTH_DIFFUSION_INT8_TRAIN": "w8a8"}
    for prec, seed in (("bf16", 42), ("int8", 42), ("nf4", 42), ("nf4", 43), ("int8", 43), ("bf16", 43)):
        tree = ctx["backend"] if prec == "int8" else ctx["base_backend"]
        env = opt if prec == "int8" else {"UNSLOTH_DIFFUSION_ROCM_W8A8_TRAIN": None}
        o = Path(ctx["work"]) / f"train_{prec}_{seed}.json"
        r = run_py(ctx, R8 / "run_train.py", ["--backend", tree, "--precision", prec, "--seed", seed, "--data", data,
                                               "--work", work, "--out", o, "--steps", 30, "--resolution", 512], o,
                   timeout = 5400, extra_env = env, tag = f"train_{prec}_{seed}")
        r.update(tree = "head" if prec == "int8" else "base")
        res["rows"].append(r)
        ctx["save"]()
    shutil.rmtree(work, ignore_errors = True)


# ---------------------------------------------------------------------------------------------------------- M8 (d)
HEALTH = ("import json, torch\nx = torch.randn(1024, 1024, device = 'cuda'); y = (x @ x).sum().item()\n"
          "torch.cuda.synchronize(); print('R8JSON ' + json.dumps({'ok': True}))\n")


def item_m8d_qvae(ctx, obs):
    path = _snap(ctx, obs, "qtvae", QT_REPO, ["vae/*", "model_index.json"])
    res = {"rows": []}
    obs["m8d_qvae"] = res
    if not path:
        res["skipped"] = "vae snapshot failed"
        return
    for side in (1024, 1280):
        for i in range(5):
            o = Path(ctx["work"]) / f"qvae_{side}_{i}.json"
            r = run_py(ctx, R8 / "run_qvae_untiled.py", ["--dir", path, "--side", side, "--out", o], o, timeout = 600,
                       tag = f"qvae{side}_{i}")
            r.pop("tb", None) if not r.get("error") else None
            r.update(i = i)
            res["rows"].append(r)
            ctx["save"]()
            if r.get("rc") == "timeout":
                h = run_code(ctx, HEALTH, [], timeout = 180, tag = "health")
                r["health_after"] = {k: h.get(k) for k in ("ok", "rc", "tail")}
                ctx["save"]()
                if not h.get("ok"):
                    res["stopped"] = f"device unhealthy after a {side}px timeout"
                    return


ITEMS = {
    "c156_spin": item_c156_spin,
    "m8b_mmap": item_m8b_mmap,
    "m8a_wankitchen": item_m8a_wankitchen,
    "m8c_h3enc": item_m8c_h3enc,
    "c186a_train": item_c186a_train,
    "m8d_qvae": item_m8d_qvae,
}
