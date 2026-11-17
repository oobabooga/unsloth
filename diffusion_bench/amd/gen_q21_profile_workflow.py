#!/usr/bin/env python3
"""gfx1151 profile job: one Qwen-Image-2.1 bf16 1024^2 denoiser step under torch.profiler, Studio vs ComfyUI v0.44.0.

  python diffusion_bench/amd/gen_q21_profile_workflow.py --out DIR --main-sha SHA [--arms a,b,c] [--patch P]

Reuses gen_nov5_workflow's job template (install, flash-attn --target, ComfyUI on Studio's torch, files from the
official repo) and replaces the differential step with run_cell + q21_prof_hook per arm, in one job on one box.
Every arm: warmup 1 + 4 prompts + one long/short pair (calls 0..154); calls 37 and 137 are profiled.
With --patch, a second Studio tree (main + patch) is measured too: arms named ``<tag>@head``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_nov5_workflow as G  # noqa: E402

BRANCH = "amd-ci-q21-profile"
DEFAULT_ARMS = "q21_c_rock,q21_s_def,q21_s_flashx"

PROFILE_STEP = """      - name: "Profile Qwen-Image-2.1 bf16 decode steps per arm"
        run: |
          set +e
          export TORCH_HOME="$AMD_CI_WORK/torch_home" HF_HOME="$AMD_CI_WORK/hf_home"
          cp "$GITHUB_WORKSPACE/diffusion_bench/amd/q21_prof_comfy_node.py" "$AMD_CI_WORK/ComfyUI/custom_nodes/q21_prof_node.py"
          export Q21_PROF_HOOK_DIR="$GITHUB_WORKSPACE/diffusion_bench/amd"
          export Q21_PROF_CALLS="{calls}"
          mkdir -p "$AMD_CI_WORK/out/prof" "$AMD_CI_WORK/dbench/states/prof/runs"
          for st in base head; do
            PYTHONPATH="$AMD_CI_WORK/states/$st/studio/backend" "$AMD_CI_PY" "$GITHUB_WORKSPACE/diffusion_bench/amd/q21_eager_kernel_check.py" \
              --out "$AMD_CI_WORK/out/prof/kernel_check_$st.json" > "$AMD_CI_WORK/out/prof/kernel_check_$st.log" 2>&1
            echo "kernel check $st:"; cat "$AMD_CI_WORK/out/prof/kernel_check_$st.json"
          done
          for spec_arm in {arms}; do
            rep=""; envset=""
            case "$spec_arm" in *"%"*) kv="${{spec_arm#*%}}"; spec_arm="${{spec_arm%%%*}}"
              envset="--set env.${{kv%%=*}}=${{kv#*=}}"; rep="_${{kv%%=*}}${{kv#*=}}" ;; esac
            case "$spec_arm" in *"#"*) rep="${{rep}}_r${{spec_arm##*#}}"; spec_arm="${{spec_arm%#*}}" ;; esac
            arm="${{spec_arm%@*}}"; state=base
            case "$spec_arm" in *@head) state=head ;; esac
            name="${{arm}}_${{state}}${{rep}}"
            export DIFFUSION_BENCH_STUDIO_SRC="$AMD_CI_WORK/states/$state"
            export Q21_PROF_OUT="$AMD_CI_WORK/out/prof/$name" Q21_PROF_TAG="$name"
            echo "=== $name ($(date -u +%H:%M:%S))"
            cd "$AMD_CI_WORK"
            timeout 3600 "$AMD_CI_PY" -u "$GITHUB_WORKSPACE/diffusion_bench/amd/q21_prof_cell.py" \\
              --spec "$GITHUB_WORKSPACE/diffusion_bench/specs/amd_nov5.json" --tag "$arm" \\
              --set n=4 --set short_n=1 $envset {extra_set}\\
              --out "$AMD_CI_WORK/dbench/states/prof/runs/$state$rep" > "$AMD_CI_WORK/out/prof/$name.log" 2>&1
            echo "$name rc=$?"
            grep -E "q21_prof|loaded in|ok:|BROKEN" "$AMD_CI_WORK/out/prof/$name.log" | tail -8
            cp "$AMD_CI_WORK/dbench/states/prof/runs/$state$rep/$arm/record.json" "$AMD_CI_WORK/out/prof/$name.record.json" 2>/dev/null
            gzip -f "$AMD_CI_WORK/out/prof/$name"/*.trace.json 2>/dev/null
          done
          "$AMD_CI_PY" "$GITHUB_WORKSPACE/diffusion_bench/amd/q21_prof_report.py" "$AMD_CI_WORK/out/prof" | tee "$AMD_CI_WORK/out/prof/REPORT.md" >> "$GITHUB_STEP_SUMMARY"
          exit 0

"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--main-sha", required = True)
    ap.add_argument("--arms", default = DEFAULT_ARMS)
    ap.add_argument("--patch", default = "nov5_marker.patch", help = "head state = main + this amd_ci/patches file")
    ap.add_argument("--calls", default = "37,137")
    ap.add_argument("--extra-set", default = "", help = "extra --set KEY=JSON for every arm, space separated")
    ap.add_argument("--branch", default = BRANCH)
    args = ap.parse_args()
    text = G.HEADER.format(branch = args.branch, main_sha = args.main_sha, comfy_sha = G.COMFY_SHA, fa_sha = G.FA_SHA)
    text = text.replace("Studio vs ComfyUI v0.44.0 on gfx1151 (Linux, nov5)", "Qwen-Image-2.1 step profile on gfx1151")
    prep = G.PREP_STEP.format(groups = "q21", wan_cmd = "", snap_only = "")
    job = G.JOB.format(jid = "q21_profile", title = "Qwen-Image-2.1 bf16 step profile", needs = "",
                       comfy_tag = G.COMFY_TAG, prep_step = prep, patch = args.patch, criteria = "no_regression",
                       preflight_gib = 50, probe_args = "")
    # only the ComfyUI venv on Studio's torch (the arm profiled); no torch 2.14.1 Studio venv
    job = job.replace("for arm in default rock; do", "for arm in rock; do")
    a = job.index("      - name: Studio venv on download.pytorch.org torch 2.14.1")
    b = job.index("      - name: ComfyUI files from the official repos")
    job = job[:a] + job[b:]
    a = job.index('      - name: "Differential (')
    b = job.index("      - name: Stop anything this job started")
    extra = "".join(f"--set {s} " for s in args.extra_set.split() if s)
    job = job[:a] + PROFILE_STEP.format(calls = args.calls, arms = " ".join(args.arms.split(",")),
                                        extra_set = extra) + job[b:]
    job = job.replace("name: nov5-q21_profile", "name: q21-profile")
    text += job
    wf = args.out / ".github" / "workflows" / f"{args.branch}.yml"
    wf.parent.mkdir(parents = True, exist_ok = True)
    for old in wf.parent.glob("*.yml"):
        if old != wf:
            old.unlink()
    wf.write_text(text, encoding = "utf-8")
    print(f"wrote {wf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
