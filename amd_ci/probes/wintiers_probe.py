#!/usr/bin/env python3
"""Probe: which RDNA compile / int8 tiers engage on this gfx1151 host (Windows or Linux), base tree vs head tree,
and what each costs per step. Observes only; judged by criteria/wintiers_win.py (Windows, differential) or
criteria/wintiers_lnx.py (Linux, regression).

Both states:
  * pytest: the changed test files of the checkout (junit ids);
  * decisions.py on the real GPU: torch_compile_runtime_available, rocm_rdna_w8a8_ready, the RDNA tier reasons
    (rdna_auto_w8a8 / rdna_explicit_int8 / rdna_explicit_w8a8) per family, then the ROCm regional compile probe.
Head state only (its sibling ``base`` checkout ran first): per family, fetch the public weights once (token-free)
into the job's HF cache, then fresh-process arms (wintiers/arm.py), each --warmup untimed renders and --prompts timed
renders with fixed seeds, order B1 H1 H2 B2 (B = base tree, H = head tree; B1/B2 and H1/H2 are the A/A pairs):
   klein  FLUX.3 Klein, explicit int8 (transformer_quant int8, bf16 text encoder), 1024^2, 16 steps
   q21    Qwen-Image-2.1, Studio defaults (auto precision / speed), 1024^2, 10 steps
   wan    Wan2.2-TI2V-5B, Studio defaults, 832x480x33, 8 steps
With --ia (Windows): one more arm IA1 for klein, base tree with UNSLOTH_NATIVE_INT8_ACT=1 (eager W8A8 on the RDNA
kernels), to check the eager-W8A8 claim. Engagement per arm = the load / speed / quant log lines plus the engine
status before and after the renders. Then wintiers/score.py: LPIPS / PSNR of every arm vs B1 (same prompt and seed).
No torch profiler (it hung the previous Windows harness). The family's weights are deleted before the next family.

--mode log: decisions plus one B and one H arm per family (no warm-up, 3 renders, short steps): which tiers engage.

  python wintiers_probe.py --state head --checkout states/head --out obs_head.json [--families klein,q21,wan]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from i8auto_probe import arm_env, fetch, pytest as run_pytest  # noqa: E402

TOOLS = HERE / "wintiers"
TESTS = ("tests/test_diffusion_native_quant.py", "tests/test_diffusion_flux3_rdna_shared_quant.py",
         "tests/test_sd_cpp_conv_direct_off.py")
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")
FAMILIES = {
    "klein": {"repo": "unsloth/flux-3-image-klein", "kind": "image", "ignore": ["*.png", "*.jpg", "variants/*"],
              "load": {"model_kind": "pipeline", "transformer_quant": "int8", "text_encoder_quant": "none"},
              "gen": {"width": 1024, "height": 1024, "steps": 16}, "log_steps": 8, "warmup": 2},
    "q21": {"repo": "Qwen/Qwen-Image-2.1", "kind": "image", "ignore": ["*.png", "*.jpg", "*.webp", "assets/*"],
            "load": {"model_kind": "pipeline"}, "gen": {"width": 1024, "height": 1024, "steps": 10}, "log_steps": 6,
            "warmup": 3},
    "wan": {"repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "kind": "video",
            "ignore": ["*.png", "*.jpg", "*.gif", "*.mp4", "assets/*", "examples/*"],
            "load": {}, "gen": {"width": 832, "height": 480, "num_frames": 33, "steps": 8}, "log_steps": 4,
            "warmup": 1},
}
ENGAGE_KEYS = ("diffusion.speed", "video.speed", "transformer_quant", "W8A8", "rocm_compile", "regional compile",
               "deferred profile", "rocm_fused", "eager-patch", "compile_cache", "native_quant", "flux_convrot_fused",
               "int8_attention", "speed_mode", "diffusion.loaded", "video.loaded")


def run_arm(tree: Path, fam: str, tag: str, spec: dict, env_extra: dict, outdir: Path, hf_home: Path, prompts: int,
            warmup: int, gen: dict, timeout: int = 5400) -> dict:
    backend = tree / "studio" / "backend"
    cmd = [sys.executable, "-u", str(TOOLS / "arm.py"), "--root", str(backend), "--kind", spec["kind"],
           "--model", spec["repo"], "--tag", f"{fam}_{tag}", "--out", str(outdir), "--load", json.dumps(spec["load"]),
           "--gen", json.dumps(gen), "--prompts", str(prompts), "--warmup", str(warmup), "--tf", "none",
           "--prof-step", "-1"]
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
    rec["rc"] = rc
    rec["arm_seconds"] = round(time.time() - t0, 1)
    rec["tree"] = tree.name
    text = log.read_text(encoding = "utf-8", errors = "replace") if log.is_file() else ""
    rec["log_tail"] = text[-3000:]
    rec["engage_lines"] = [x[:700] for x in text.splitlines() if any(k in x for k in ENGAGE_KEYS)][:300]
    for k in ("proc_load", "proc_end", "proc_dropped", "shapes", "i8_log"):
        rec.pop(k, None)
    return rec


def decisions(checkout: Path, out: Path) -> dict:
    backend = checkout / "studio" / "backend"
    dest = out.with_name(out.stem + ".decisions.json")
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-u", str(TOOLS / "decisions.py"), "--root", str(backend), "--out",
                            str(dest)], cwd = backend, capture_output = True, text = True, encoding = "utf-8",
                           errors = "replace", env = arm_env(out.parent / "hf_unused", {}), timeout = 1800)
        rc, tail = r.returncode, (r.stdout + r.stderr)[-3000:]
    except subprocess.TimeoutExpired:
        rc, tail = "timeout", ""
    try:
        res = json.loads(dest.read_text(encoding = "utf-8"))
    except Exception as exc:  # noqa: BLE001
        res = {"error": f"no decisions: {exc}"[:300]}
    res["rc"], res["tail"], res["seconds"] = rc, tail, round(time.time() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--mode", default = "full", choices = ("full", "log"))
    ap.add_argument("--families", default = "klein,q21,wan")
    ap.add_argument("--prompts", type = int, default = 4)
    ap.add_argument("--ia", action = "store_true")
    a = ap.parse_args()
    for k in TOKEN_VARS:
        os.environ.pop(k, None)
    out = Path(a.out).resolve()
    checkout = Path(a.checkout).resolve()
    obs: dict = {"state": a.state, "mode": a.mode, "python": sys.executable, "checkout": str(checkout),
                 "platform": sys.platform}

    def save():
        out.write_text(json.dumps(obs, indent = 1, default = str), encoding = "utf-8")

    try:
        import torch
        obs["torch"] = torch.__version__
        obs["hip"] = getattr(torch.version, "hip", None)
        obs["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        obs["arch"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", None) if obs["gpu"] else None
        import triton
        obs["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        obs["env_error"] = repr(exc)[:500]
    try:
        obs["rev"] = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output = True,
                                    text = True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    obs["decisions"] = decisions(checkout, out)
    save()
    import i8auto_probe
    i8auto_probe.TESTS = TESTS
    obs["pytest"] = run_pytest(checkout, sys.executable, out)
    save()
    if a.state != "head":
        return 0
    trees = {"base": checkout.parent / "base", "head": checkout}
    if not (trees["base"] / "studio").is_dir():
        obs["driver_error"] = f"no base checkout at {trees['base']}"
        save()
        return 0
    work = Path(os.environ.get("AMD_CI_WORK") or out.parent) / "wintiers_work"
    work.mkdir(parents = True, exist_ok = True)
    hf_home = work / "hf"
    arms_dir = out.parent / "wintiers_arms"
    arms_dir.mkdir(parents = True, exist_ok = True)
    obs["lpips_install"] = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps", "lpips"],
        capture_output = True, text = True).returncode
    obs["fetch"] = []
    obs["arms"] = {}
    obs["scores"] = {}
    for fam in [f for f in a.families.split(",") if f]:
        spec = FAMILIES[fam]
        got = fetch(spec["repo"], spec["ignore"], hf_home, obs["fetch"])
        save()
        if got["rc"] != 0:
            obs.setdefault("driver_errors", []).append(f"{fam}: fetch failed")
            save()
            continue
        if a.mode == "log":
            gen = dict(spec["gen"], steps = spec["log_steps"])
            plan = [("B1", "base", {}), ("H1", "head", {})]
            prompts, warmup = 3, 0
        else:
            gen = spec["gen"]
            plan = [("B1", "base", {}), ("H1", "head", {}), ("H2", "head", {}), ("B2", "base", {})]
            if a.ia and fam == "klein":
                plan.append(("IA1", "base", {"UNSLOTH_NATIVE_INT8_ACT": "1"}))
            prompts, warmup = a.prompts, spec["warmup"]
        for tag, tree, env in plan:
            rec = run_arm(trees[tree], fam, tag, spec, env, arms_dir, hf_home, prompts, warmup, gen)
            rec.update(side = tag.rstrip("0123456789"), family = fam)
            obs["arms"][f"{fam}_{tag}"] = rec
            save()
        t0 = time.time()
        try:
            pairs = "H1:B1,B2:B1,H2:H1,H2:B2" + (",IA1:B1" if a.ia and fam == "klein" else "")
            sc = subprocess.run([sys.executable, "-u", str(TOOLS / "score.py"), "--out", str(arms_dir), "--family",
                                 fam, "--ref", "B1", "--pairs", pairs, "--kind", spec["kind"]],
                                capture_output = True, text = True, encoding = "utf-8", errors = "replace",
                                env = arm_env(hf_home, {}), timeout = 3600)
            obs.setdefault("score_rc", {})[fam] = sc.returncode
            obs.setdefault("score_tail", {})[fam] = (sc.stdout + sc.stderr)[-2000:]
        except subprocess.TimeoutExpired:
            obs.setdefault("score_rc", {})[fam] = "timeout"
        try:
            obs["scores"][fam] = json.loads((arms_dir / f"scores_{fam}.json").read_text(encoding = "utf-8"))
        except Exception as exc:  # noqa: BLE001
            obs["scores"][fam] = {"error": repr(exc)[:300], "seconds": round(time.time() - t0, 1)}
        save()
        shutil.rmtree(hf_home / "hub", ignore_errors = True)
        save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
