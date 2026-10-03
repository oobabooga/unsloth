#!/usr/bin/env python3
"""Build a ready-to-push AMD CI branch that runs diffusion_bench on gfx1151, through amd_ci/scaffold.py.

  python diffusion_bench/amd/scaffold_diffusion.py --pr 11766 --out $WORKSPACE/temp/dbench_amd/ci_dbench_pr11766
  python diffusion_bench/amd/scaffold_diffusion.py --pr 11766 --windows --out .../ci_dbench_pr11766w
  python diffusion_bench/amd/scaffold_diffusion.py --pr 11713 --mode differential \\
      --defect slow:heavy_q21_fbcache:heavy_q21_bf16:0.9 --only 'heavy_q21_*' --out .../ci_dbench_pr11713

What it adds on top of amd_ci/scaffold.py (which it calls, so the toolkit, the templates and the lint stay the
single source of truth):
  - probe diffusion_bench/amd/diffusion_probe.py and criteria amd/criteria_no_regression.py (default) or
    amd/criteria_differential.py, with the probe args after "--" so differential.py passes them through;
  - Linux: --no-suites (the GPU job is the whole point), and the GPU job's timeout raised to 330 min (installs
    plus base and head cells; the probe's --budget-min keeps each state inside it);
  - Windows: the template's commented-out concurrency group enabled as amd-ci-gfx1151-gpu-windows (every past
    Windows GPU measurement used it) and the job timeout raised the same way; the probe installs ROCm torch
    from repo.amd.com/rocm/whl/gfx1151 into the job's venv itself (--install auto);
  - a copy of diffusion_bench/ next to amd_ci/, because the runner checks out this branch and nothing else;
  - the lint again on the edited workflow, and the push command printed, never run.

It never pushes, never writes a secret into the workflow, and refuses an --out that already holds workflows.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLKIT = HERE.parent
AMD_CI = TOOLKIT.parent / "amd_ci"
HOST_REPO = "https://github.com/oobabooga/unsloth.git"

WIN_CONCURRENCY_OFF = ("    # concurrency:\n"
                       "    #   group: amd-ci-strix-halo-windows-gpu\n"
                       "    #   cancel-in-progress: false\n")
WIN_CONCURRENCY_ON = ("    concurrency:\n"
                      "      group: amd-ci-gfx1151-gpu-windows\n"
                      "      cancel-in-progress: false\n")


def quote_for(windows: bool, arg: str) -> str:
    # Single quotes are literal in both bash and PowerShell 5.1, which keeps fnmatch patterns from globbing.
    if any(ch in arg for ch in "*?[] '\"$;&|<>(){}"):
        return "'" + arg.replace("'", "''" if windows else "'\"'\"'") + "'"
    return arg


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pr", type = int, required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--branch", default = None, help = "default amd-ci-dbench-pr<N>[-win]")
    ap.add_argument("--merged", action = "store_true", help = "PR already merged: merge commit vs its parent")
    ap.add_argument("--windows", action = "store_true")
    ap.add_argument("--mode", choices = ["regression", "differential"], default = "regression")
    ap.add_argument("--defect", default = None, help = "differential only, e.g. broken:zimg_fp8")
    ap.add_argument("--spec", default = "diffusion_bench/specs/amd_strix_small.json",
                    help = "path inside the branch")
    ap.add_argument("--only", action = "append", default = [], help = "cell patterns (default: the probe's)")
    ap.add_argument("--budget-min", type = float, default = 100.0, help = "per state")
    ap.add_argument("--edge", default = None, help = "edge-suite command for the probe to record (TODO(verify))")
    ap.add_argument("--probe-extra", default = "", help = "more probe args, appended verbatim")
    ap.add_argument("--job-timeout-min", type = int, default = 330)
    args = ap.parse_args()

    if args.mode == "differential" and not args.defect:
        ap.error("--mode differential needs --defect (the VOID rule needs to know what the base must show)")
    if args.mode == "regression" and args.defect:
        ap.error("--defect only means something with --mode differential")
    branch = args.branch or f"amd-ci-dbench-pr{args.pr}" + ("-win" if args.windows else "")
    criteria = f"diffusion_bench/amd/criteria_{'differential' if args.mode == 'differential' else 'no_regression'}.py"
    probe_args = ["--spec", args.spec, "--budget-min", str(args.budget_min)]
    for pat in args.only:
        probe_args += ["--only", pat]
    if args.defect:
        probe_args += ["--defect", args.defect]
    if args.edge:
        probe_args += ["--edge", args.edge]
    probe_str = "-- " + " ".join(quote_for(args.windows, a) for a in probe_args)
    if args.probe_extra:
        probe_str += " " + args.probe_extra

    cmd = [sys.executable, str(AMD_CI / "scaffold.py"), "--pr", str(args.pr), "--out", str(args.out),
           "--branch", branch, "--probe", "diffusion_bench/amd/diffusion_probe.py", "--criteria", criteria,
           "--probe-args", probe_str]
    if args.merged:
        cmd.append("--merged")
    cmd.append("--windows" if args.windows else "--no-suites")
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush = True)
    rc = subprocess.run(cmd, capture_output = True, text = True)
    if rc.returncode != 0:
        print(rc.stdout + rc.stderr)
        print("amd_ci/scaffold.py failed; nothing else done")
        return rc.returncode

    wf = args.out / ".github" / "workflows" / f"{branch}.yml"
    text = wf.read_text(encoding = "utf-8")
    if args.windows:
        if WIN_CONCURRENCY_OFF not in text:
            raise SystemExit("the Windows template's commented concurrency block moved; update scaffold_diffusion.py")
        text = text.replace(WIN_CONCURRENCY_OFF, WIN_CONCURRENCY_ON, 1)
    if "    timeout-minutes: 120\n" not in text:
        raise SystemExit("the GPU job's 'timeout-minutes: 120' moved; update scaffold_diffusion.py")
    text = text.replace("    timeout-minutes: 120\n", f"    timeout-minutes: {args.job_timeout_min}\n")
    text = text.replace(f'name: "AMD CI{" (Windows)" if args.windows else ""}: validate PR {args.pr}"',
                        f'name: "AMD CI{" (Windows)" if args.windows else ""}: diffusion_bench PR {args.pr} '
                        f'({args.mode})"', 1)
    wf.write_text(text, encoding = "utf-8")

    dest = args.out / "diffusion_bench"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(TOOLKIT, dest, ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.log", "outputs",
                                                                 "*.enc.tmp"))
    lint = subprocess.run([sys.executable, str(AMD_CI / "lib" / "lint_workflow.py"), str(wf)],
                          capture_output = True, text = True)
    print(lint.stdout + lint.stderr)
    if lint.returncode != 0:
        print("lint found error-level problems after the diffusion edits; fix them before pushing")
        return lint.returncode
    print(f"""wrote {wf}
branch contents: .github/workflows/{branch}.yml, amd_ci/, diffusion_bench/ (no other workflow can fire)
mode {args.mode}{' defect ' + args.defect if args.defect else ''}; probe args: {probe_str}

next (NOT run by this script; the push TRIGGERS the run):
  cd {args.out} && git init -q && git checkout -q -b {branch}
  git add -A && git commit -q -m "diffusion_bench PR {args.pr} on gfx1151 ({'Windows' if args.windows else 'Linux'})"
  git remote add ooba {HOST_REPO}
  git push -q ooba HEAD:refs/heads/{branch}

monitor and read the verdict (not the tick):
  RUN=$(gh run list --repo oobabooga/unsloth --branch {branch} --limit 1 --json databaseId --jq '.[0].databaseId')
  gh run download $RUN --repo oobabooga/unsloth -D $WORKSPACE/temp/dbench_amd/run_$RUN
  find $WORKSPACE/temp/dbench_amd/run_$RUN -name VERDICT.md -exec cat {{}} +
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
