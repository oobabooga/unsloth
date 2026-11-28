#!/usr/bin/env python3
"""Probe: nov28 revisit of four gfx1151 levers killed on a speed line or a bitwise / distance-from-main gate (observes
only; criteria/rv28_measure.py judges). Every arm is a fresh process; base-tree arms run from the sibling base
checkout (origin/main), head-tree arms from the head checkout (origin/main + patches/rv28.diff). No token is read or
passed (fetches are token-free, child envs drop every *TOKEN* / *SECRET* variable).

Items (``smoke`` gates the rest):
  smoke   pytest of the changed test files on the head tree
  h3vae   MiniMax-H3 video VAE decode, 121 frames 832x480 latent: S (base, stock bf16) / T (head, Triton fast path,
          decoder levers) ABBA x2, plus R = float32 VAE stock decode of the same latent (reference). PSNR vs R.
  q21     Qwen-Image-2.1 W8A8 at Studio defaults: B (Studio w8a8_mm) vs K (comfy-kitchen #338 int8_gemm raced per key,
          bit-identity required per key), B1 K1 K2 B2 B3 K3, 3 renders each; output sha per render across arms.
  klein   FLUX.3 klein at Studio defaults: B (base) vs Q (head, UNSLOTH_DIFFUSION_ROCM_FUSED_LIFT=1: fused QK-norm +
          RoPE on ROCm) B1 Q1 Q2 B2; when Q's median wall is >= 3% under B's: F (TunableOp off, kernel-swap floor) and
          R (fp32 denoiser, speed_mode off, native attention), LPIPS vs R.
  wan     Wan 2.2 TI2V-5B 832x480x49 8 steps, 8 clips per arm, base tree: H (main defaults), P (weight pitch 64 on every
          eligible weight), F (TunableOp off), K (rocBLAS instead of hipBLASLt), R (fp32 denoiser reference), P2, H2.

  python rv28_probe.py --state head --checkout states/head --out obs_head.json [--items ...] [--small]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RV = HERE / "rv28"
sys.path.insert(0, str(HERE))
from i8auto_probe import arm_env, fetch  # noqa: E402

TOOLS = HERE / "wintiers"
PITCH = "UNSLOTH_DIFFUSION_ROCM_WEIGHT_PITCH"
TUNE = "UNSLOTH_DIFFUSION_ROCM_TUNABLEOP"
LIFT = "UNSLOTH_DIFFUSION_ROCM_FUSED_LIFT"
REF_LOAD = {"speed_mode": "off", "attention_backend": "native"}
REF_ENV = {"ARM_FP32_DENOISER": "1", PITCH: "0", TUNE: "0"}
TESTS = ("tests/test_video_minimax_h3_vae.py", "tests/test_video_minimax_h3_vae_rocm_single_frame.py",
         "tests/test_diffusion_flux3_qk_rope.py")
FAMILIES = {
    "wan": {"repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "kind": "video",
            "ignore": ["*.png", "*.jpg", "*.gif", "*.mp4", "assets/*", "examples/*"], "load": {},
            "gen": {"width": 832, "height": 480, "num_frames": 49, "steps": 8}, "warmup": 1, "prompts": 8},
    "klein": {"repo": "unsloth/flux-3-image-klein", "kind": "image", "ignore": ["*.png", "*.jpg", "variants/*"],
              "load": {"model_kind": "pipeline"}, "gen": {"width": 1024, "height": 1024}, "warmup": 2, "prompts": 8},
    "q21": {"repo": "Qwen/Qwen-Image-2.1", "kind": "image", "ignore": ["*.png", "*.jpg", "*.webp", "assets/*"],
            "load": {"model_kind": "pipeline"}, "gen": {"width": 1024, "height": 1024}, "warmup": 2, "prompts": 3},
}
SMALL = {"wan": {"gen": {"width": 832, "height": 480, "num_frames": 17, "steps": 2}, "warmup": 0, "prompts": 1},
         "klein": {"warmup": 1, "prompts": 1, "gen": {"width": 1024, "height": 1024, "steps": 2}},
         "q21": {"warmup": 1, "prompts": 1, "gen": {"width": 1024, "height": 1024, "steps": 2}}}
ENGAGE_KEYS = ("diffusion.speed", "video.speed", "transformer_quant", "W8A8", "rocm_compile", "regional compile",
               "rocm_weight_pitch", "tunableop", "TunableOp", "diffusion.loaded", "video.loaded", "int8", "flux3_qk",
               "qk_rope", "precision", "dtype", "h3_vae")
SECRET = ("TOKEN", "SECRET", "PASSWORD", "WANDB_API_KEY")


def child_env(extra = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in SECRET)}
    env.update(extra or {})
    return env


def run(args, out: Path, log: Path, env = None, timeout = 3600, cwd = None) -> dict:
    t = time.time()
    with open(log, "ab") as fh:
        fh.write(f"\n=== {' '.join(map(str, args))} env={env}\n".encode())
        fh.flush()
        try:
            rc = subprocess.run([sys.executable, *map(str, args)], stdout = fh, stderr = subprocess.STDOUT,
                                env = child_env(env), timeout = timeout, cwd = cwd).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res = {"rc": rc, "wall_s": round(time.time() - t, 1)}
    if out.is_file():
        try:
            res.update(json.loads(out.read_text(encoding = "utf-8")))
        except Exception as exc:  # noqa: BLE001
            res["parse_error"] = repr(exc)
    else:
        res["missing"] = True
    return res


def run_arm(tree: Path, fam: str, tag: str, spec: dict, load_extra: dict, env_extra: dict, outdir: Path,
            hf_home: Path, timeout = 7200) -> dict:
    backend = tree / "studio" / "backend"
    load = dict(spec["load"], **(load_extra or {}))
    cmd = [sys.executable, "-u", str(RV / "arm.py"), "--root", str(backend), "--kind", spec["kind"],
           "--model", spec["repo"], "--tag", f"{fam}_{tag}", "--out", str(outdir), "--load", json.dumps(load),
           "--gen", json.dumps(spec["gen"]), "--prompts", str(spec["prompts"]), "--warmup", str(spec["warmup"]),
           "--tf", "none", "--prof-step", "-1"]
    log = outdir / f"{fam}_{tag}.log"
    t0 = time.time()
    try:
        with open(log, "w", encoding = "utf-8") as fh:
            rc = subprocess.run(cmd, cwd = backend, stdout = fh, stderr = subprocess.STDOUT,
                                env = arm_env(hf_home, env_extra), timeout = timeout).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    try:
        rec = json.loads((outdir / f"{fam}_{tag}" / "record.json").read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        rec = {"verdict": "error", "error": f"no record: {exc}"[:300]}
    rec.update({"rc": rc, "arm_seconds": round(time.time() - t0, 1), "tree": tree.name, "env": env_extra,
                "load_extra": load_extra})
    text = log.read_text(encoding = "utf-8", errors = "replace") if log.is_file() else ""
    rec["log_tail"] = text[-3000:]
    rec["engage_lines"] = [x[:700] for x in text.splitlines() if any(k in x for k in ENGAGE_KEYS)][:300]
    for k in ("proc_load", "proc_end", "proc_dropped", "shapes", "i8_log", "flags_load", "flags_end"):
        rec.pop(k, None)
    return rec


def score(ctx, obs, fam, ref, pairs, kind):
    arms_dir = ctx["work"] / f"{fam}_arms"
    sc = subprocess.run([sys.executable, "-u", str(TOOLS / "score.py"), "--out", str(arms_dir), "--family", fam,
                         "--ref", ref, "--pairs", ",".join(pairs), "--kind", kind],
                        capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                        env = arm_env(ctx["hf_home"], {}), timeout = 3600)
    try:
        obs[f"{fam}_scores"] = json.loads((arms_dir / f"scores_{fam}.json").read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        obs[f"{fam}_scores"] = {"error": repr(exc)[:300], "tail": (sc.stdout + sc.stderr)[-1500:]}


def family(ctx, obs, fam, plan, after = None):
    """plan: [(tag, tree, load_extra, env)]; after(obs, res) may return more arms to run (conditional arms)."""
    trees = {"base": ctx["base_root"], "head": ctx["head_root"]}
    spec = dict(FAMILIES[fam])
    if ctx["small"]:
        spec.update(SMALL[fam])
    hf_home = ctx["hf_home"]
    arms_dir = ctx["work"] / f"{fam}_arms"
    arms_dir.mkdir(parents = True, exist_ok = True)
    got = fetch(spec["repo"], spec["ignore"], hf_home, obs.setdefault(f"{fam}_fetch", []))
    if got["rc"] != 0:
        obs[f"{fam}_error"] = "fetch failed"
        return
    res = obs.setdefault(fam, {})
    # untimed, unscored base-tree arm first: the first process on a fresh box pays cold MIOpen / compile costs
    warm_spec = dict(spec, prompts = 1, warmup = 1 if not ctx["small"] else 0)
    if not ctx["small"]:
        w = run_arm(trees["base"], fam, "W0", warm_spec, {}, {}, arms_dir, hf_home)
        w.pop("renders", None)
        obs[f"{fam}_warm"] = w
    todo = list(plan)
    extra_done = False
    while todo:
        tag, tree, load_extra, env = todo.pop(0)
        res[tag] = run_arm(trees[tree], fam, tag, spec, load_extra, env, arms_dir, hf_home)
        res[tag]["arm"] = tag.rstrip("0123456789")
        obs[fam] = res
        ctx["save"]()
        if not todo and after is not None and not extra_done:
            extra_done = True
            more = after(obs, res) or []
            obs[f"{fam}_conditional"] = [t for t, *_ in more] or "not run"
            todo.extend(more)
    shas = {tag: [r.get("sha") for r in (rec.get("renders") or [])] for tag, rec in res.items()}
    obs[f"{fam}_shas"] = shas
    return arms_dir


def cleanup(ctx, fam):
    arms_dir = ctx["work"] / f"{fam}_arms"
    for p in list(arms_dir.glob(f"{fam}_*/r*.npz")):
        p.unlink()
    shutil.rmtree(ctx["hf_home"] / "hub", ignore_errors = True)


def _med_wall(rec):
    xs = sorted(r["wall_s"] for r in (rec.get("renders") or []) if isinstance(r.get("wall_s"), (int, float)))
    return xs[len(xs) // 2] if xs else None


def item_smoke(ctx, obs):
    head = ctx["head_root"] / "studio" / "backend"
    t = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", *TESTS],
                       cwd = head, capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                       env = child_env(), timeout = 1800)
    obs["smoke_pytest"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-3000:], "s": round(time.time() - t, 1)}
    obs["smoke_ok"] = r.returncode == 0


def _psnr(a_path, b_path):
    import torch
    a, b = torch.load(a_path).float(), torch.load(b_path).float()
    if a.shape != b.shape:
        return None
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(4.0 / mse)


def item_h3vae(ctx, obs):
    d = ctx["hf_home"].parent / "rv_h3vae"
    code = ("import sys\nfrom huggingface_hub import snapshot_download\nsnapshot_download(sys.argv[1], "
            "allow_patterns = ['vae/*', 'model_index.json'], local_dir = sys.argv[2], token = False)\n")
    t = time.time()
    r = subprocess.run([sys.executable, "-c", code, "MiniMaxAI/MiniMax-H3", str(d)], capture_output = True, text = True,
                       encoding = "utf-8", errors = "replace", timeout = 3600, env = child_env())
    obs["h3vae_fetch"] = {"rc": r.returncode, "s": round(time.time() - t, 1), "tail": (r.stdout + r.stderr)[-600:]}
    base = ctx["base_root"] / "studio" / "backend"
    head = ctx["head_root"] / "studio" / "backend"
    small = ctx["small"]
    lat = ["--lat-t", "5"] if small else []
    plan = [("S1", "stock", base, 2), ("T1", "fast", head, 2), ("T2", "fast", head, 2), ("S2", "stock", base, 2),
            ("R", "fp32", base, 1)]
    if small:
        plan = [("S1", "stock", base, 1), ("T1", "fast", head, 1), ("R", "fp32", base, 1)]
    res = []
    for tag, arm, tree, reps in plan:
        o = ctx["work"] / f"h3vae_{tag}.json"
        sv = ctx["work"] / f"h3vae_{tag}.pt"
        args = [RV / "h3vae.py", "--backend", tree, "--dir", d, "--arm", arm, "--out", o, "--save", sv,
                "--reps", str(reps), *lat]
        rr = run(args, o, ctx["log"], timeout = 3600)
        rr["tag"] = tag
        res.append(rr)
        obs["h3vae"] = res
        ctx["save"]()
    ref = ctx["work"] / "h3vae_R.pt"
    obs["h3vae_psnr_vs_R"] = {tag: _psnr(ctx["work"] / f"h3vae_{tag}.pt", ref) for tag, *_ in plan
                              if tag != "R" and ref.is_file() and (ctx["work"] / f"h3vae_{tag}.pt").is_file()}
    s1 = ctx["work"] / "h3vae_S1.pt"
    obs["h3vae_psnr_vs_S1"] = {tag: _psnr(ctx["work"] / f"h3vae_{tag}.pt", s1) for tag, *_ in plan
                               if tag not in ("S1", "R") and s1.is_file() and (ctx["work"] / f"h3vae_{tag}.pt").is_file()}
    for p in ctx["work"].glob("h3vae_*.pt"):
        p.unlink()
    shutil.rmtree(d, ignore_errors = True)


def item_q21(ctx, obs):
    kdir = ctx["hf_home"].parent / "rv_kitchen"
    o = ctx["work"] / "kern_build.json"
    obs["kern_build"] = run([RV / "kern_build.py", "--dest", kdir, "--out", o, "--only", "k338"], o, ctx["log"],
                            timeout = 3600)
    b = ((obs["kern_build"].get("builds") or {}).get("k338") or {})
    if not b.get("ok"):
        obs["q21_error"] = "kitchen #338 build failed"
        return
    kenv = {"ARM_KITCHEN_RACE": b["tree"]}
    if ctx["small"]:
        plan = [("B1", "base", {}, {}), ("K1", "base", {}, kenv)]
    else:
        plan = [("B1", "base", {}, {}), ("K1", "base", {}, kenv), ("K2", "base", {}, kenv), ("B2", "base", {}, {}),
                ("B3", "base", {}, {}), ("K3", "base", {}, kenv)]
    family(ctx, obs, "q21", plan)
    cleanup(ctx, "q21")


def item_klein(ctx, obs):
    lift = {LIFT: "1"}
    if ctx["small"]:
        plan = [("B1", "base", {}, {}), ("Q1", "head", {}, lift), ("R1", "base", REF_LOAD, REF_ENV)]
        after = None
    else:
        plan = [("B1", "base", {}, {}), ("Q1", "head", {}, lift), ("Q2", "head", {}, lift), ("B2", "base", {}, {})]

        def after(obs, res):
            b = [_med_wall(res[t]) for t in ("B1", "B2") if t in res]
            q = [_med_wall(res[t]) for t in ("Q1", "Q2") if t in res]
            b, q = [x for x in b if x], [x for x in q if x]
            if not b or not q:
                return []
            gain = 1 - (sum(q) / len(q)) / (sum(b) / len(b))
            obs["klein_gain_for_condition"] = round(gain, 4)
            if gain < 0.03:
                return []
            return [("F1", "base", {}, {TUNE: "0"}), ("R1", "base", REF_LOAD, REF_ENV)]
    arms_dir = family(ctx, obs, "klein", plan, after)
    if arms_dir is not None and "R1" in (obs.get("klein") or {}):
        score(ctx, obs, "klein", "R1", [], "image")
    cleanup(ctx, "klein")


def item_wan(ctx, obs):
    if ctx["small"]:
        plan = [("H1", "base", {}, {}), ("P1", "base", {}, {PITCH: "64"}), ("R1", "base", REF_LOAD, REF_ENV)]
    else:
        plan = [("H1", "base", {}, {}), ("P1", "base", {}, {PITCH: "64"}), ("F1", "base", {}, {TUNE: "0"}),
                ("K1", "base", {}, {"TORCH_BLAS_PREFER_HIPBLASLT": "0"}), ("R1", "base", REF_LOAD, REF_ENV),
                ("P2", "base", {}, {PITCH: "64"}), ("H2", "base", {}, {})]
    arms_dir = family(ctx, obs, "wan", plan)
    if arms_dir is not None:
        score(ctx, obs, "wan", "R1", ["P1:H1", "H2:H1", "P2:P1"], "video")
    cleanup(ctx, "wan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--items", default = "smoke,h3vae,q21,klein,wan")
    ap.add_argument("--small", action = "store_true", help = "smoke sizes: 1 render per arm, short clips / steps")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    obs: dict = {"state": a.state, "platform": sys.platform, "args": vars(a)}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")
    try:
        import torch
        obs["env"] = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
                      "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
                      "python": sys.version.split()[0]}
        try:
            import triton
            obs["env"]["triton"] = triton.__version__
        except Exception as exc:  # noqa: BLE001
            obs["env"]["triton"] = repr(exc)[:200]
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)[:500]
    if a.state != "head":
        obs["skipped"] = "measured from the head step only (base tree arms run from there)"
        save()
        return 0
    head = Path(a.checkout).resolve()
    work = out.parent / "rvwork"
    work.mkdir(parents = True, exist_ok = True)
    ctx = {"head_root": head, "base_root": head.parent / "base", "work": work, "log": out.parent / "rv_children.log",
           "hf_home": Path(os.environ.get("AMD_CI_WORK") or work) / "hf_rv", "save": save, "small": a.small}
    obs["trees_exist"] = {"head": head.is_dir(), "base": ctx["base_root"].is_dir()}
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "lpips"], capture_output = True)
    save()
    items = [x for x in a.items.split(",") if x]
    fns = {"smoke": item_smoke, "h3vae": item_h3vae, "q21": item_q21, "klein": item_klein, "wan": item_wan}
    for item in items:
        if item != "smoke" and "smoke" in items and not obs.get("smoke_ok"):
            obs[f"{item}_skipped"] = "smoke failed"
            continue
        t = time.time()
        try:
            fns[item](ctx, obs)
        except Exception:  # noqa: BLE001
            import traceback
            obs[f"{item}_error"] = traceback.format_exc()[-2000:]
        obs.setdefault("item_s", {})[item] = round(time.time() - t, 1)
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
