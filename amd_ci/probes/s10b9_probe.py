#!/usr/bin/env python3
"""unsloth #19508 + unsloth-zoo #2778 on gfx1151: saved attention in AUTOMATIC mode (UNSLOTH_GC_SAVE_ATTENTION unset),
Qwen3-0.6B LoRA at 8192 tokens. Observes only; JSON via --out.

Environment and the zoo companion checkout come from train5_probe.py (same venv recipe as the train5 units).
  base  : --reps repeats of (unsloth base, zoo base).
  head  : interleaved H, B, H, B, ... : --reps repeats of (unsloth head, zoo head) and --reps A/A repeats of the base
          pair (sibling states/base checkout + zoo base), so drift between the blocks is visible; then one
          --short-seq repeat on head (below the 8192 threshold: nothing should be routed).
  merge : one repeat of (unsloth merge, zoo head): engagement on the PR merged into main.
Each repeat is its own process with its own compile cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train5_probe as t5  # noqa: E402


def rep(py, tag, paths, out_dir: Path, seq, a, hf_home):
    out, log = out_dir / f"cell_{tag}.json", out_dir / f"cell_{tag}.log"
    cdir = out_dir / f"run_{tag}"
    cdir.mkdir(parents = True, exist_ok = True)
    env = t5.state_env(paths, {"UNSLOTH_COMPILE_LOCATION": str(cdir / "unsloth_compiled_cache"), "HF_HOME": hf_home})
    for k in ("UNSLOTH_GC_SAVE_ATTENTION", "UNSLOTH_GC_SAVE_ATTENTION_MIN_SEQ"):
        env.pop(k, None)
    rc = t5.sh([py, "-u", str(HERE / "s10b9_cell.py"), "--out", str(out), "--seq", str(seq), "--warmup", str(a.warmup),
                "--steps", str(a.steps), "--model", a.model], log, timeout = a.cell_timeout, env = env, cwd = str(cdir))
    r: dict = {"rc": rc, "paths": paths}
    if out.is_file():
        try:
            r.update(json.loads(out.read_text(encoding = "utf-8")))
        except Exception as e:  # noqa: BLE001
            r["parse_error"] = repr(e)
    else:
        r["missing_output"] = True
    r["log_tail"] = t5.tail(log, 2500)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--venv", type = Path, default = None)
    ap.add_argument("--use-current", action = "store_true")
    ap.add_argument("--torch-spec", default = "torch==2.11.0+rocm7.13.0 torchvision==0.26.0+rocm7.13.0")
    ap.add_argument("--torch-index", default = "https://repo.amd.com/rocm/whl/gfx1151/")
    ap.add_argument("--transformers", default = "5.23.0")
    ap.add_argument("--trl", default = "1.18.0")
    ap.add_argument("--peft", default = "0.21.3")
    ap.add_argument("--companion-repo", required = True)
    ap.add_argument("--companion-base", required = True)
    ap.add_argument("--companion-head", required = True)
    ap.add_argument("--model", default = "unsloth/Qwen3-0.6B")
    ap.add_argument("--seq", type = int, default = 8192)
    ap.add_argument("--short-seq", type = int, default = 2048)
    ap.add_argument("--reps", type = int, default = 4)
    ap.add_argument("--warmup", type = int, default = 3)
    ap.add_argument("--steps", type = int, default = 10)
    ap.add_argument("--cell-timeout", type = int, default = 1500)
    a = ap.parse_args()
    a.extra_pip = []
    if bool(a.venv) == bool(a.use_current):
        raise SystemExit("exactly one of --venv / --use-current")
    a.out = a.out.resolve()
    work = a.out.parent
    out_dir = work / f"{a.state}_artifacts"
    out_dir.mkdir(parents = True, exist_ok = True)
    hf_home = str(Path(os.environ.get("AMD_CI_WORK") or work.parent.parent) / "hf_home")  # outside the uploaded out/
    obs: dict = {"state": a.state, "checkout": a.checkout, "platform": sys.platform}

    def dump():
        a.out.write_text(json.dumps(obs, indent = 2, default = str), encoding = "utf-8")

    import subprocess
    obs["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd = a.checkout, capture_output = True, text = True,
                                     encoding = "utf-8").stdout.strip()
    obs["env"] = t5.build_env(a, work)
    dump()
    if obs["env"].get("error"):
        return 0
    py = obs["env"]["python"]
    zoo_base = t5.companion(a, work, "base")
    zoo_head = t5.companion(a, work, "head")
    obs["companion"] = {"base": [a.companion_base, zoo_base], "head": [a.companion_head, zoo_head]}
    obs["reps"] = {}
    if a.state == "base":
        pairs = [(f"base{i}", [a.checkout, zoo_base], a.seq) for i in range(a.reps)]
    elif a.state == "head":
        base_ck = str(Path(a.checkout).parent / "base")
        pairs = []
        for i in range(a.reps):
            pairs.append((f"head{i}", [a.checkout, zoo_head], a.seq))
            pairs.append((f"aa{i}", [base_ck, zoo_base], a.seq))
        if a.short_seq:
            pairs.append(("head_short", [a.checkout, zoo_head], a.short_seq))
    else:
        pairs = [(f"{a.state}0", [a.checkout, zoo_head], a.seq)]
    obs["expected_trees"] = {tag: paths for tag, paths, _ in pairs}
    for tag, paths, seq in pairs:
        obs["reps"][tag] = rep(py, f"{a.state}_{tag}", paths, out_dir, seq, a, hf_home)
        dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
