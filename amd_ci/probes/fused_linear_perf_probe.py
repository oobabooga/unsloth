#!/usr/bin/env python3
"""Probe: unsloth-zoo fused_linear (branch fused-linear-triton) vs zoo main, step time / memory / fp64 error on this GPU.

Drives fused_linear_bench.py (copied next to this file: `cp scripts/fused_linear_bench.py amd_ci/probes/`) once per
state, every arm in its own process, ABBA rounds:
  base -> unsloth-zoo main: old fused CE (ce), #1782 fused_linear_token_logps (dpo), the default GRPO chunked
          log-softmax and the opt-in recompute (grpo, grpo_ent)
  head -> the new tree: unsloth_fused_ce_loss as routed (ce, what users get) + fused_linear_cross_entropy direct,
          fused_linear_logprobs (dpo, grpo, grpo_ent)
The head tree is zoo main at the overlay's base SHA plus `--head-overlay` (a fused_linear_bench.py --make-overlay
tarball committed under amd_ci/data/), or `--head-ref` on `--head-repo`. Judging is criteria/fused_linear_perf.py's job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_ARMS = "old_zoo,main1782,main_old,main_recompute,dense"
HEAD_ARMS = "new,new_routed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state")
    ap.add_argument("--checkout", type = Path)
    ap.add_argument("--out", type = Path)
    ap.add_argument("--head-overlay", default = "amd_ci/data/new_overlay.tgz")
    ap.add_argument("--head-ref")
    ap.add_argument("--head-env", default = "half:UNSLOTH_LOGPROB_FP32_LOGITS=0,exact:UNSLOTH_LOGPROB_FP32_LOGITS=2",
                    help = "extra new-arm variants, fused_linear_bench --new-env syntax ('' = none)")
    ap.add_argument("--head-repo", default = "https://github.com/unslothai/unsloth-zoo")
    ap.add_argument("--tokens", default = "1024,4096")
    ap.add_argument("--shapes", default = "2048x151936")
    ap.add_argument("--workloads", default = "ce,dpo,grpo,grpo_ent")
    ap.add_argument("--rounds", default = "2")
    ap.add_argument("--reps", default = "6")
    args = ap.parse_args()
    state = "head" if "head" in args.state else "base"
    obs: dict = {"state": args.state, "role": state}
    try:
        bench = HERE / "fused_linear_bench.py"
        if not bench.is_file():
            raise FileNotFoundError(f"{bench} missing: cp scripts/fused_linear_bench.py amd_ci/probes/ before scaffolding")
        sys.path.insert(0, str(HERE))
        import fused_linear_bench as fb

        work = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / f"fl_probe_{state}"
        work.mkdir(parents = True, exist_ok = True)
        if state == "base":
            ref = "main"
            overlay = Path(args.head_overlay)
            if not overlay.is_absolute() and not overlay.exists():
                overlay = HERE.parents[1] / args.head_overlay
            if overlay.is_file():  # pin base to the overlay's own base SHA, so base and head share everything else
                import tarfile

                with tarfile.open(overlay) as tf:
                    ref = json.loads(tf.extractfile("OVERLAY_META.json").read())["base"]
            tree = fb.clone_tree(work, "main", ref)
            flag, arms = "--zoo-main", BASE_ARMS
            variants = []
            obs["zoo_ref"] = ref
        else:
            if args.head_ref:
                tree = fb.clone_tree(work, "new", args.head_ref, args.head_repo)
                obs["zoo_ref"] = args.head_ref
            else:
                overlay = Path(args.head_overlay)
                if not overlay.is_absolute() and not overlay.exists():
                    overlay = HERE.parents[1] / args.head_overlay
                tree, meta = fb.apply_overlay(work, overlay)
                obs["overlay"] = meta
            flag, arms = "--zoo-new", HEAD_ARMS
            # HIP logits mode is decided here: half (0), default (1 = exact target on HIP), exact target (2).
            variants = ["--new-env", args.head_env] if args.head_env else []
        obs["zoo_head"] = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output = True,
                                         text = True).stdout.strip()
        out = work / "bench"
        cmd = [sys.executable, str(bench), flag, str(tree), "--arms", arms, "--tokens", args.tokens,
               "--shapes", args.shapes, "--no-extra", "--workloads", args.workloads, "--rounds", args.rounds,
               "--reps", args.reps, "--accuracy", "--out", str(out)] + variants
        p = subprocess.run(cmd, capture_output = True, text = True, timeout = 3 * 3600)
        (work / "bench.log").write_text(p.stdout[-200000:] + "\n" + p.stderr[-50000:], encoding = "utf-8")
        res = json.loads((out / "results.json").read_text(encoding = "utf-8")) if (out / "results.json").exists() else None
        if res is None:
            obs["error"] = f"bench rc={p.returncode}: {p.stderr[-1500:]}"
        else:
            obs["env"] = res["meta"]["env"]
            obs["rows"] = [{k: r.get(k) for k in ("workload", "head", "tokens", "H", "V", "arm", "arm_label", "stats",
                                                   "skip", "error")} for r in res["rows"]]
    except Exception as e:  # noqa: BLE001
        obs["error"] = f"{type(e).__name__}: {e}"
    args.out.write_text(json.dumps(obs, indent = 1), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
