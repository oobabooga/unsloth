#!/usr/bin/env python3
"""Create a ready-to-push AMD CI branch for a PR.

Produces a directory containing the toolkit plus one generated workflow, lints
it, and prints the push command. The lint is not optional: it is the step that
turns "a run I will lose in ten minutes" into "a message I read in one second".

  python amd_ci/scaffold.py --pr 9487 --out ci_pr9487 \\
      --tests tests/test_sd_cpp_install.py

  python amd_ci/scaffold.py --pr 9315 --merged --out ci_pr9315 --no-gpu \\
      --tests tests/test_whatever.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Must match pytest_probe.py's --subdir default; selftest.py asserts they agree.
PYTEST_SUBDIR = "studio/backend"

# Probe / criteria names that measure time; lint_workflow.py E108 uses the same pattern.
TIMING_MODULE = re.compile(r"perf|bench|timing", re.I)


def shared_gpu_job(text: str) -> str:
    """The gpu job without the shared concurrency group, marked so capability.py and lint see it.

    The group holds one runner plus one waiter and a third arrival evicts the waiter, so
    under contention a queued correctness probe can be cancelled indefinitely. Dropping it
    trades exclusivity (a co-tenant may share the card) for starting at all.
    """
    head, sep, gpu_job = text.partition("  gpu:")
    block = re.search(r"    # A shared constant, deliberately:.*?\n    concurrency:\n"
                      r"      group: amd-ci-gfx1151-gpu\n      cancel-in-progress: false\n",
                      gpu_job, re.S)
    if not sep or not block:
        raise SystemExit("template has no concurrency group in the gpu job to drop")
    gpu_job = (gpu_job[:block.start()]
               + "    # --shared-gpu: no concurrency group, so a co-tenant may share the card.\n"
                 "    # Correctness only; capability.py reports `exclusive_gpu` as a gap.\n"
                 "    env:\n      AMD_CI_SHARED_GPU: \"1\"\n"
               + gpu_job[block.end():])
    gpu_job = gpu_job.replace("GPU measurement (exclusive)", "GPU check (shared GPU)", 1)
    return head + sep + gpu_job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", type = int, required = True)
    ap.add_argument("--repo", default = "https://github.com/unslothai/unsloth",
                    help = "repository the PR belongs to (states.py clones it)")
    ap.add_argument("--no-studio", action = "store_true",
                    help = "Linux GPU job: skip `install.sh --local` and run the probe on the "
                           "runner's python3. For PRs outside unslothai/unsloth, whose probe "
                           "needs no Studio venv; torch-derived capabilities then read "
                           "UNDETERMINED, so the criteria gates must prove the GPU was used.")
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--branch", default = None)
    ap.add_argument("--merged", action = "store_true")
    ap.add_argument("--patch", type = Path, default = None,
                    help = "Linux: head = --patch-base plus this git diff (a fix that is not a PR yet); base = "
                           "--patch-base. The diff rides in the branch under amd_ci/patches/.")
    ap.add_argument("--patch-base", default = None, help = "full commit SHA the --patch applies to")
    ap.add_argument("--tests", nargs = "*", default = ["tests/"])
    ap.add_argument("--probe", default = "amd_ci/probes/pytest_probe.py")
    ap.add_argument("--criteria", default = "amd_ci/criteria/pytest_no_regression.py")
    ap.add_argument("--probe-args", default = "")
    ap.add_argument("--windows", action = "store_true",
                    help = "target the WINDOWS half of the pool: emits "
                           "`runs-on: [self-hosted, Windows, strix-halo, devlab-dispatch]` "
                           "with `shell: powershell`, the PowerShell preamble, and a "
                           "linux-control job that differs only in the OS label so a stall "
                           "can be read as 'no matching runner' rather than 'busy pool'. "
                           "Non-Docker workloads only: docker is not installed on any of "
                           "the four Windows boxes.")
    ap.add_argument("--no-gpu", action = "store_true",
                    help = "drop the GPU job; use when nothing needs measuring")
    ap.add_argument("--no-suites", action = "store_true")
    ap.add_argument("--shared-gpu", action = "store_true",
                    help = "Linux GPU job: drop the shared `amd-ci-gfx1151-gpu` concurrency group "
                           "so the job starts as soon as a runner is free instead of waiting "
                           "behind (and being evicted by) every other branch's measurement. For "
                           "correctness probes only (decisions, dtypes, losses, crashes): the card "
                           "may have a co-tenant, so capability.py reports `exclusive_gpu` as a gap "
                           "and timing / throughput / peak-memory numbers are not evidence. "
                           "Refused for timing probes and criteria.")
    ap.add_argument("--spoof-devices", type = int, default = 0, metavar = "N",
                    help = "present N EXTRA HIP devices to torch via LD_PRELOAD, so "
                           "multi-GPU code paths become reachable on this one-GPU "
                           "runner. The extra devices are the real GPU wearing other "
                           "numbers: good for selection and index logic, useless for "
                           "sharding, throughput or collectives. capability.py is told "
                           "and forces multi_gpu to stay UNMET, so the verdict still "
                           "declares the gap. See lib/device_multiplier.py.")
    ap.add_argument("--spoof-nvidia", action = "store_true",
                    help = "present the AMD GPU as an NVIDIA one (torch.version, device "
                           "name / capability, a fake nvidia-smi) so NVIDIA-only branches, "
                           "Windows + NVIDIA included, become reachable. Linux and "
                           "--windows. Wiring only: capability.py keeps `nvidia` UNMET "
                           "and names the spoof. See lib/nvidia_identity.py.")
    args = ap.parse_args()
    # states.py clones --repo as given: `owner/name` would be read as a local path on the runner.
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        args.repo = f"https://github.com/{args.repo}"

    branch = args.branch or f"amd-ci-pr{args.pr}"
    out = args.out
    if bool(args.patch) != bool(args.patch_base):
        raise SystemExit("--patch and --patch-base go together")
    if args.patch and (args.windows or args.merged):
        raise SystemExit("--patch is Linux-only and compares a commit with itself plus a diff (no --merged)")
    if args.patch and not args.patch.is_file():
        raise SystemExit(f"--patch {args.patch}: no such file")
    # Both land unescaped in the workflow's YAML env block and in an unquoted shell word.
    if args.patch and not re.fullmatch(r"[0-9a-f]{40}", args.patch_base):
        raise SystemExit(f"--patch-base {args.patch_base!r}: expected a full 40-hex commit SHA")
    if args.patch and not re.fullmatch(r"[A-Za-z0-9._+-]+", args.patch.name):
        raise SystemExit(f"--patch {args.patch.name!r}: file name may only use letters, digits and ._+-")

    # The Linux-template flags name jobs and mechanisms the Windows template does
    # not have. Refusing is deliberate: accepting and ignoring them is how
    # --spoof-devices once silently did nothing while the workflow linted clean.
    if args.windows:
        for flag, value in (("--no-gpu", args.no_gpu), ("--no-suites", args.no_suites),
                            ("--spoof-devices", args.spoof_devices)):
            if value:
                raise SystemExit(
                    f"{flag} applies to the Linux template's jobs, which the Windows "
                    f"template does not have. "
                    + ("The device multiplier is an LD_PRELOAD shim over the HIP runtime "
                       "and has no Windows equivalent, so multi-GPU wiring stays "
                       "unreachable there."
                       if flag == "--spoof-devices" else
                       "The Windows template emits one `windows` job plus a `linux-control` "
                       "job; drop the flag."))

    if args.shared_gpu:
        if args.windows:
            raise SystemExit("--shared-gpu applies to the Linux GPU job; the Windows template "
                             "declares no concurrency group, so it is already shared")
        if args.no_gpu:
            raise SystemExit("--shared-gpu needs the GPU job; drop --no-gpu")
        timing = [m for m in (args.probe, args.criteria) if TIMING_MODULE.search(Path(m).name)]
        if timing:
            raise SystemExit(
                f"--shared-gpu refuses timing modules ({', '.join(timing)}): a co-tenant on the "
                f"card moves the number being measured. Keep the exclusive group for timing.")

    # `--tests` is relative to the probe's --subdir (studio/backend by default),
    # so passing a repo-root path silently doubles the prefix, selects nothing,
    # and burns a run to reach INCONCLUSIVE. Caught here rather than explained.
    if args.probe.endswith("pytest_probe.py"):
        bad = [t for t in args.tests if t.startswith(PYTEST_SUBDIR + "/")]
        if bad:
            raise SystemExit(
                f"--tests is relative to the probe's --subdir ({PYTEST_SUBDIR}), so "
                f"{bad[0]!r} resolves to {PYTEST_SUBDIR}/{bad[0]} and matches nothing. "
                f"Drop the prefix: {bad[0][len(PYTEST_SUBDIR) + 1:]!r}")

    # Scaffolding writes a workflow, so it must be impossible to aim it at an
    # existing repository's CI. Adding a workflow to a checkout that already has
    # them is the self-propagation shape the repo's semgrep rules exist to catch,
    # and this tool has no reason to do it: it builds a fresh throwaway branch
    # directory. Refuse anything else.
    existing = out / ".github" / "workflows"
    if existing.is_dir() and any(existing.iterdir()):
        raise SystemExit(
            f"refusing to scaffold into {existing}: it already contains workflows. "
            f"Point --out at a new directory; this tool creates a throwaway CI branch, "
            f"it does not add workflows to an existing repo.")
    if (out / ".git").exists() and (out / ".github" / "workflows").is_dir():
        raise SystemExit(f"refusing to scaffold into the existing git repo at {out}")

    out.mkdir(parents = True, exist_ok = True)

    # The toolkit has to travel with the branch: the runner checks out this
    # branch and nothing else.
    dest = out / "amd_ci"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ROOT, dest, ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "templates"))
    (dest / "templates").mkdir(exist_ok = True)
    for name in ("workflow.yml", "workflow_windows.yml"):
        shutil.copy(ROOT / "templates" / name, dest / "templates" / name)
    if args.patch:
        (dest / "patches").mkdir(exist_ok = True)
        shutil.copy(args.patch, dest / "patches" / args.patch.name)

    template = ROOT / "templates" / ("workflow_windows.yml" if args.windows else "workflow.yml")

    # The Windows template has ONE Differential step serving whatever probe was
    # chosen, so the pytest probe's `--tests` and a custom probe's args cannot both
    # be spliced in blindly: passing `--tests` to a probe that has no such flag
    # fails at argparse, inside the job, for nothing.
    if args.probe.endswith("pytest_probe.py"):
        diff_args = "-- --tests " + " ".join(args.tests)
        if args.probe_args:
            diff_args += " " + args.probe_args
    else:
        # differential.py collects probe args as a REMAINDER: without `--` it rejects them.
        diff_args = ("-- " + args.probe_args) if args.probe_args.strip() else ""

    text = template.read_text(encoding = "utf-8")
    text = (text
            .replace("__DIFF_ARGS__", diff_args)
            .replace("__PR__", str(args.pr))
            .replace("__BRANCH__", branch)
            .replace("__MERGED__", "1" if args.merged else "0")
            .replace("__PATCH_BASE__", args.patch_base or "")
            .replace("__PATCH_FILE__", f"amd_ci/patches/{args.patch.name}" if args.patch else "")
            .replace("__TESTS__", " ".join(args.tests))
            .replace("__PROBE__", args.probe)
            .replace("__CRITERIA__", args.criteria)
            .replace("__PROBE_ARGS__", args.probe_args)
            .replace('UPSTREAM: "https://github.com/unslothai/unsloth"', f'UPSTREAM: "{args.repo}"'))
    if f'UPSTREAM: "{args.repo}"' not in text:
        raise SystemExit("template has no UPSTREAM line to point at --repo")

    if args.no_studio:
        if args.windows:
            raise SystemExit("--no-studio applies to the Linux GPU job; the Windows template installs no Studio")
        start = text.find("      - name: Environment and real gate\n")
        end = text.find("      - name: Differential\n", start)
        if start < 0 or end < 0:
            raise SystemExit("template has no `Environment and real gate` step to replace")
        step = (
            "      - name: Environment and host gate (no Studio)\n"
            "        run: |\n"
            "          set -euo pipefail\n"
            "          echo \"AMD_CI_PY=$(command -v python3)\" >> \"$GITHUB_ENV\"\n"
            "          # No torch here, so rocm/gpu read UNDETERMINED: the criteria gates prove the GPU ran.\n"
            "          python3 \"$GITHUB_WORKSPACE/amd_ci/lib/gate.py\" --no-torch --out \"$AMD_CI_WORK/out/host.json\"\n"
            "\n")
        text = text[:start] + step + text[end:]

    if args.spoof_devices:
        if args.no_gpu:
            raise SystemExit("--spoof-devices needs the GPU job; drop --no-gpu")
        # Inserted AFTER the gate step, so the gate still measures the real host, and
        # before Differential, so the probe and everything it spawns inherit it.
        #
        # Into the GPU JOB specifically. The template has a Differential step in both
        # the suites and gpu jobs, and a plain replace(..., 1) lands on the suites one
        # -- which --no-suites then deletes, so the flag silently did nothing and lint
        # reported the workflow clean. Split on the job header first, and assert.
        anchor = "      - name: Differential\n"
        head, sep, gpu_job = text.partition("  gpu:")
        if not sep or anchor not in gpu_job:
            raise SystemExit("template has no Differential step in the gpu job")
        step = (
            "      - name: Present extra HIP devices\n"
            "        run: |\n"
            "          set -euo pipefail\n"
            "          \"$AMD_CI_PY\" \"$GITHUB_WORKSPACE/amd_ci/lib/device_multiplier.py\" \\\n"
            f"            --build-into \"$AMD_CI_WORK/shim\" --extra {args.spoof_devices} --github-env\n"
            "\n")
        gpu_job = gpu_job.replace(anchor, step + anchor, 1)
        text = head + sep + gpu_job
        if "device_multiplier.py" not in text:
            raise SystemExit("failed to insert the device multiplier step")

    if args.spoof_nvidia:
        if args.no_gpu and not args.windows:
            raise SystemExit("--spoof-nvidia needs the GPU job; drop --no-gpu")
        # After the gate (it measures the real host) and immediately before the
        # Differential step of the job that measures: the gpu job on Linux, the one
        # windows job on Windows. Same partition-and-assert as --spoof-devices.
        anchor = "      - name: Differential\n"
        job = "  windows:" if args.windows else "  gpu:"
        head, sep, tail = text.partition(job)
        if not sep or anchor not in tail:
            raise SystemExit(f"template has no Differential step in the{job.rstrip(':')} job")
        if args.windows:
            step = (
                "      - name: Present the GPU as NVIDIA\n"
                "        run: |\n"
                "          $ErrorActionPreference = \"Continue\"\n"
                "          & $env:AMD_CI_PY \"$env:GITHUB_WORKSPACE\\amd_ci\\lib\\nvidia_identity.py\" `\n"
                "            --build-into \"$env:AMD_CI_WORK\\nvidia\" --github-env\n"
                "          if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
                "\n")
        else:
            step = (
                "      - name: Present the GPU as NVIDIA\n"
                "        run: |\n"
                "          set -euo pipefail\n"
                "          \"$AMD_CI_PY\" \"$GITHUB_WORKSPACE/amd_ci/lib/nvidia_identity.py\" \\\n"
                "            --build-into \"$AMD_CI_WORK/nvidia\" --github-env\n"
                "\n")
        text = head + sep + tail.replace(anchor, step + anchor, 1)
        if "nvidia_identity.py" not in text:
            raise SystemExit("failed to insert the NVIDIA identity step")

    if args.shared_gpu:
        text = shared_gpu_job(text)

    if args.no_gpu:
        text = text.split("  gpu:")[0].rstrip() + "\n"
    if args.no_suites:
        head, _, tail = text.partition("  suites:")
        text = head + "  gpu:" + tail.partition("  gpu:")[2]

    wf_dir = out / ".github" / "workflows"
    wf_dir.mkdir(parents = True, exist_ok = True)
    wf = wf_dir / f"{branch}.yml"
    wf.write_text(text, encoding = "utf-8")
    (out / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding = "utf-8")

    print(f"wrote {wf}")
    rc = subprocess.run([sys.executable, str(ROOT / "lib" / "lint_workflow.py"), str(wf)]).returncode
    if rc != 0:
        print("\nlint found error-level problems; fix them before pushing")
        return rc

    print(f"""
next (queued: waits its turn instead of evicting another branch's GPU job):
  python {ROOT / "push_queue.py"} submit --dir {out} --branch {branch}

or push directly, skipping the queue:
  cd {out} && git init -q && git checkout -q -b {branch}
  git add -A && git commit -q -m "Validate PR {args.pr} on gfx1151"
  git remote add ooba https://github.com/oobabooga/unsloth.git
  git push -q ooba HEAD:refs/heads/{branch}
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
