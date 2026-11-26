#!/usr/bin/env python3
"""Probe: two ROCm gfx1151 Studio levers before shipping (observes only; criteria/ab_measure.py gates that each item ran).

  A = vendored Triton conv3d for video VAEs (UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D=1 on the head tree)
  P = Linear weight row pitch K + 64 with the TunableOp table rule (UNSLOTH_DIFFUSION_ROCM_WEIGHT_PITCH=64)

Head state only: base arms run from the sibling base checkout (origin/main), head arms from the head checkout
(origin/main + both branches, every lever forced by its env var per arm; the other lever pinned off). Every arm is a
fresh process; arms interleave ABBA over 3 rounds, so same-arm repeats give the A/A spread and LPIPS. No token is read
or passed (fetches are token-free, the child env drops every *TOKEN* variable).

Items (in order; ``smoke`` gates the rest):
  smoke    pytest of the two new test files on the head tree + smoke_gpu.py (pitch torch.equal per GEMM, a few
           conv3d shapes vs fp32, engagement)
  wane2e   Wan 2.2 TI2V-5B through Studio at its defaults, 832x480x49, 8 steps: B (main), P, A, AP (both)
  klein    FLUX.3 klein at Studio defaults (bf16, shipped TunableOp table active): B vs P (table-on control)
  q21      Qwen-Image-2.1 at Studio defaults (INT8 + table): B vs P, B1 P1 P2 B2
  krea     Krea-2 Turbo at Studio defaults 1024^2: B vs P
  vae      conv3d_vendor_check.py: micro (Wan decoder convs), full Wan decode, LTX-2.5 VAE decode; stock vs Triton

  python ab_probe.py --state head --checkout states/head --out obs_head.json [--items smoke,wane2e,...]
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
AB = HERE / "ab"
sys.path.insert(0, str(HERE))
from i8auto_probe import arm_env, fetch  # noqa: E402

TOOLS = HERE / "wintiers"
ROUNDS = 3
PITCH_ENV = "UNSLOTH_DIFFUSION_ROCM_WEIGHT_PITCH"
CONV_ENV = "UNSLOTH_DIFFUSION_ROCM_TRITON_CONV3D"
ARM_ENV = {
    "B": {},
    "P": {PITCH_ENV: "64", CONV_ENV: "0"},
    "A": {CONV_ENV: "1", PITCH_ENV: "0"},
    "AP": {CONV_ENV: "1", PITCH_ENV: "64"},
    "H": {},  # head tree at its shipped defaults
}
TESTS = ("tests/test_diffusion_rocm_weight_pitch.py", "tests/test_diffusion_rocm_triton_conv3d.py")
FAMILIES = {
    "wan": {"repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "kind": "video",
            "ignore": ["*.png", "*.jpg", "*.gif", "*.mp4", "assets/*", "examples/*"], "load": {},
            "gen": {"width": 832, "height": 480, "num_frames": 49, "steps": 8}, "warmup": 1, "prompts": 3},
    "klein": {"repo": "unsloth/flux-3-image-klein", "kind": "image", "ignore": ["*.png", "*.jpg", "variants/*"],
              "load": {"model_kind": "pipeline"}, "gen": {"width": 1024, "height": 1024, "steps": 16}, "warmup": 2,
              "prompts": 3},
    "q21": {"repo": "Qwen/Qwen-Image-2.1", "kind": "image", "ignore": ["*.png", "*.jpg", "*.webp", "assets/*"],
            "load": {"model_kind": "pipeline"}, "gen": {"width": 1024, "height": 1024, "steps": 10}, "warmup": 2,
            "prompts": 3},
    "krea": {"repo": "krea/Krea-2-Turbo", "kind": "image", "ignore": ["*.png", "*.jpg", "images/*", "turbo.safetensors",
                                                                       "LICENSE.pdf"],
             "load": {"model_kind": "pipeline"}, "gen": {"width": 1024, "height": 1024}, "warmup": 2, "prompts": 3},
}
ENGAGE_KEYS = ("diffusion.speed", "video.speed", "transformer_quant", "W8A8", "rocm_compile", "regional compile",
               "rocm_weight_pitch", "conv3d", "tunableop", "TunableOp", "diffusion.loaded", "video.loaded", "int8",
               "precision", "dtype")
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


def abba(arms):
    """3 rounds, every other round reversed: B P P B B P for two arms."""
    seq, n = [], {}
    for rnd in range(ROUNDS):
        for arm in (arms if rnd % 2 == 0 else arms[::-1]):
            n[arm] = n.get(arm, 0) + 1
            seq.append((f"{arm}{n[arm]}", arm))
    return seq


def run_arm(tree: Path, fam: str, tag: str, spec: dict, env_extra: dict, outdir: Path, hf_home: Path,
            timeout = 5400) -> dict:
    backend = tree / "studio" / "backend"
    cmd = [sys.executable, "-u", str(AB / "arm.py"), "--root", str(backend), "--kind", spec["kind"],
           "--model", spec["repo"], "--tag", f"{fam}_{tag}", "--out", str(outdir), "--load", json.dumps(spec["load"]),
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
    rec.update({"rc": rc, "arm_seconds": round(time.time() - t0, 1), "tree": tree.name, "env": env_extra})
    text = log.read_text(encoding = "utf-8", errors = "replace") if log.is_file() else ""
    rec["log_tail"] = text[-3000:]
    rec["engage_lines"] = [x[:700] for x in text.splitlines() if any(k in x for k in ENGAGE_KEYS)][:300]
    for k in ("proc_load", "proc_end", "proc_dropped", "shapes", "i8_log", "flags_load", "flags_end"):
        rec.pop(k, None)
    return rec


def e2e(ctx, obs, fam, arms, pairs_extra = ()):
    trees = {"base": ctx["base_root"], "head": ctx["head_root"]}
    spec = FAMILIES[fam]
    hf_home = ctx["hf_home"]
    arms_dir = ctx["work"] / f"{fam}_arms"
    arms_dir.mkdir(parents = True, exist_ok = True)
    got = fetch(spec["repo"], spec["ignore"], hf_home, obs.setdefault(f"{fam}_fetch", []))
    if got["rc"] != 0:
        obs[f"{fam}_error"] = "fetch failed"
        return
    res = obs.setdefault(fam, {})
    plan = arms if isinstance(arms[0], tuple) else abba(arms)
    # One untimed, unscored base-tree arm first: the first process on a fresh box pays MIOpen / compile-cache cold
    # costs (Windows run 45766869391: B1 152 s vs B2 86 s wall), which would otherwise land on whichever arm is first.
    warm_spec = dict(spec, prompts = 1, warmup = 1)
    obs.setdefault(f"{fam}_warm", run_arm(trees["base"], fam, "W0", warm_spec, {}, arms_dir, hf_home))
    obs[f"{fam}_warm"].pop("renders", None)
    for tag, arm in plan:
        tree = "base" if arm == "B" else "head"
        res[tag] = run_arm(trees[tree], fam, tag, spec, ARM_ENV[arm], arms_dir, hf_home)
        res[tag]["arm"] = arm
        obs[fam] = res
        ctx["save"]()
    ref = plan[0][0]
    firsts = {}
    for tag, arm in plan:
        firsts.setdefault(arm, tag)
    pairs = [f"{firsts[a]}:{ref}" for a in firsts if firsts[a] != ref]
    pairs += [f"{tag}:{firsts[arm]}" for tag, arm in plan if tag != firsts[arm]]
    pairs += list(pairs_extra)
    sc = subprocess.run([sys.executable, "-u", str(TOOLS / "score.py"), "--out", str(arms_dir), "--family", fam,
                         "--ref", ref, "--pairs", ",".join(pairs), "--kind", spec["kind"]],
                        capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                        env = arm_env(hf_home, {}), timeout = 3600)
    try:
        obs[f"{fam}_scores"] = json.loads((arms_dir / f"scores_{fam}.json").read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        obs[f"{fam}_scores"] = {"error": repr(exc)[:300], "tail": (sc.stdout + sc.stderr)[-1500:]}
    # byte identity per render index between arms (sha of the full uint8 output)
    shas = {tag: [r.get("sha") for r in (res[tag].get("renders") or [])] for tag, _ in plan}
    obs[f"{fam}_sha_equal"] = {p: (shas.get(p.split(":")[0]) == shas.get(p.split(":")[1])
                                   and bool(shas.get(p.split(":")[0]))) for p in pairs}
    for p in arms_dir.glob(f"{fam}_*/r*.npz"):
        p.unlink()
    shutil.rmtree(hf_home / "hub", ignore_errors = True)


def item_smoke(ctx, obs):
    head = ctx["head_root"] / "studio" / "backend"
    junit = ctx["work"] / "smoke_junit.xml"
    t = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider",
                        f"--junitxml={junit}", *TESTS], cwd = head, capture_output = True, text = True,
                       encoding = "utf-8", errors = "replace", env = child_env(), timeout = 1800)
    obs["smoke_pytest"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-3000:], "s": round(time.time() - t, 1)}
    o = ctx["work"] / "smoke_gpu.json"
    obs["smoke_gpu"] = run([AB / "smoke_gpu.py", "--backend", head, "--out", o], o, ctx["log"], timeout = 2400)
    g = obs["smoke_gpu"]
    obs["smoke_ok"] = bool(r.returncode == 0 and g.get("rc") == 0 and g.get("pitch_all_equal")
                           and g.get("conv_ok"))


def _vae_fetch(ctx, obs, repo, sub, name):
    d = ctx["hf_home"].parent / "ab_vae" / name  # outside the uploaded out/ tree
    code = ("import sys\nfrom huggingface_hub import snapshot_download\nsnapshot_download(sys.argv[1], "
            "allow_patterns = [sys.argv[3] + '/*', 'model_index.json'], local_dir = sys.argv[2], token = False)\n")
    r = subprocess.run([sys.executable, "-c", code, repo, str(d), sub], capture_output = True, text = True,
                       encoding = "utf-8", errors = "replace", timeout = 3600, env = child_env())
    obs.setdefault("vae_fetch", []).append({"repo": repo, "rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]})
    return d


def _psnr(a_path, b_path):
    import torch
    a, b = torch.load(a_path).float(), torch.load(b_path).float()
    if a.shape != b.shape:
        return None
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(4.0 / mse)


def item_vae(ctx, obs, modes = ("micro", "decode", "ltxdecode")):
    head = ctx["head_root"] / "studio" / "backend"
    res = obs.setdefault("vae", [])
    dirs = {}
    if "decode" in modes or "micro" in modes:
        dirs["wan"] = _vae_fetch(ctx, obs, "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "vae", "wan")
    if "ltxdecode" in modes:
        dirs["ltx"] = _vae_fetch(ctx, obs, ctx["ltx_repo"], "vae", "ltx")
    for mode in modes:
        vdir = dirs["ltx"] if mode == "ltxdecode" else dirs["wan"]
        saved = []
        for tag, arm in abba(["stock", "triton"]):
            o = ctx["work"] / f"vae_{mode}_{tag}.json"
            sv = ctx["work"] / f"vae_{mode}_{tag}.pt"
            args = [AB / "conv3d_vendor_check.py", "--backend", head, "--mode", mode, "--vae-dir", vdir, "--out", o,
                    "--arm", arm]
            if mode != "micro":
                args += ["--save", sv] + ctx.get("decode_args", [])
                saved.append(tag)
            r = run(args, o, ctx["log"], timeout = 4800)
            r.update({"mode": mode, "tag": tag, "arm": arm})
            res.append(r)
            obs["vae"] = res
            ctx["save"]()
        if saved:
            ref = ctx["work"] / f"vae_{mode}_{saved[0]}.pt"
            obs.setdefault("vae_psnr", {})[mode] = {
                t: _psnr(ref, ctx["work"] / f"vae_{mode}_{t}.pt") for t in saved[1:]
                if ref.is_file() and (ctx["work"] / f"vae_{mode}_{t}.pt").is_file()}
            for t in saved:
                p = ctx["work"] / f"vae_{mode}_{t}.pt"
                if p.is_file():
                    p.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--items", default = "smoke,wane2e,klein,krea,q21,vae")
    ap.add_argument("--vae-modes", default = "micro,decode,ltxdecode")
    ap.add_argument("--wan-arms", default = "B,P,A,AP")
    ap.add_argument("--wan-plan", default = "", help = "explicit order instead of ABBA x3, e.g. B1:B,P1:P,P2:P,B2:B")
    ap.add_argument("--ltx-repo", default = "Lightricks/LTX-2")
    ap.add_argument("--decode-args", default = "")
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
    work = out.parent / "abwork"
    work.mkdir(parents = True, exist_ok = True)
    ctx = {"head_root": head, "base_root": head.parent / "base", "work": work, "log": out.parent / "ab_children.log",
           "hf_home": Path(os.environ.get("AMD_CI_WORK") or work) / "hf_ab", "save": save, "ltx_repo": a.ltx_repo,
           "decode_args": a.decode_args.split() if a.decode_args else []}
    obs["trees_exist"] = {"head": head.is_dir(), "base": ctx["base_root"].is_dir()}
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "lpips"], capture_output = True)
    save()
    items = [x for x in a.items.split(",") if x]
    for item in items:
        if item != "smoke" and "smoke" in items and not obs.get("smoke_ok"):
            obs[f"{item}_skipped"] = "smoke failed"
            continue
        t = time.time()
        try:
            if item == "smoke":
                item_smoke(ctx, obs)
            elif item == "wane2e":
                plan = [tuple(x.split(":")) for x in a.wan_plan.split(",") if x] if a.wan_plan else None
                e2e(ctx, obs, "wan", plan or a.wan_arms.split(","))
            elif item in ("klein", "krea"):
                e2e(ctx, obs, item, ["B", "P"])
            elif item == "q21":
                e2e(ctx, obs, "q21", [("B1", "B"), ("P1", "P"), ("P2", "P"), ("B2", "B")])
            elif item == "vae":
                item_vae(ctx, obs, tuple(a.vae_modes.split(",")))
            else:
                obs[f"{item}_error"] = "unknown item"
        except Exception:  # noqa: BLE001
            import traceback
            obs[f"{item}_error"] = traceback.format_exc()[-2000:]
        obs.setdefault("item_s", {})[item] = round(time.time() - t, 1)
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
