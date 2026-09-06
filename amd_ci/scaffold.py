#!/usr/bin/env python3
"""Create a ready-to-push AMD CI branch for a PR.

Produces a directory containing the toolkit plus one generated workflow, lints
it, and prints the push command. The lint is not optional: it is the step that
turns "a run I will lose in ten minutes" into "a message I read in one second".

  python amd_ci/scaffold.py --pr 9487 --out ci_pr9487 \\
      --tests tests/test_sd_cpp_install.py

  python amd_ci/scaffold.py --pr 9315 --merged --out ci_pr9315 --no-gpu \\
      --tests tests/test_whatever.py

A differential over two PREBUILT llama.cpp releases (no PR, no source):

  python amd_ci/scaffold.py --prebuilt b10715-mix-86bd2d3,b10798-mix-659e406 \\
      --asset rocm-gfx1151 --model unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ1_S \\
      --env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 --control-env absent --control-env GGML_CUDA_ENABLE_UNIFIED_MEMORY=0 \\
      --control-asset vulkan --control-tag b10687-mix-67dfc8b --out ci_uma [--windows]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "lib"))
import prebuilt  # noqa: E402

# Must match pytest_probe.py's --subdir default; selftest.py asserts they agree.
PYTEST_SUBDIR = "studio/backend"


def refuse_existing_workflows(out: Path) -> None:
    """Scaffolding writes a workflow, so it must be impossible to aim it at an
    existing repository's CI. Adding a workflow to a checkout that already has
    them is the self-propagation shape the repo's semgrep rules exist to catch,
    and this tool has no reason to do it: it builds a fresh throwaway branch
    directory. Refuse anything else."""
    existing = out / ".github" / "workflows"
    if existing.is_dir() and any(existing.iterdir()):
        raise SystemExit(
            f"refusing to scaffold into {existing}: it already contains workflows. "
            f"Point --out at a new directory; this tool creates a throwaway CI branch, "
            f"it does not add workflows to an existing repo.")
    if (out / ".git").exists() and (out / ".github" / "workflows").is_dir():
        raise SystemExit(f"refusing to scaffold into the existing git repo at {out}")


def copy_toolkit(out: Path) -> None:
    """The toolkit has to travel with the branch: the runner checks out this
    branch and nothing else."""
    dest = out / "amd_ci"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ROOT, dest, ignore = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "templates"))
    (dest / "templates").mkdir(exist_ok = True)
    for tpl in sorted((ROOT / "templates").glob("*.yml")):
        shutil.copy(tpl, dest / "templates" / tpl.name)


def write_and_lint(out: Path, branch: str, text: str) -> tuple[Path, int]:
    wf_dir = out / ".github" / "workflows"
    wf_dir.mkdir(parents = True, exist_ok = True)
    wf = wf_dir / f"{branch}.yml"
    wf.write_text(text, encoding = "utf-8")
    (out / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding = "utf-8")
    print(f"wrote {wf}")
    rc = subprocess.run([sys.executable, str(ROOT / "lib" / "lint_workflow.py"), str(wf)]).returncode
    if rc != 0:
        print("\nlint found error-level problems; fix them before pushing")
    return wf, rc


def push_recipe(out: Path, branch: str, subject: str) -> str:
    return f"""
next:
  cd {out} && git init -q && git checkout -q -b {branch}
  git add -A && git commit -q -m "{subject}"
  git remote add ooba https://github.com/oobabooga/unsloth.git
  git push -q ooba HEAD:refs/heads/{branch}
"""


def scaffold_prebuilt(args) -> int:
    """A release-tag differential: two prebuilts, a model, a sentinel, the
    llama-server probe at both states, then the control arms."""
    # PR-mode flags name mechanisms this mode does not have; refuse, do not ignore.
    for flag, value in (("--pr", args.pr), ("--merged", args.merged), ("--probe-args", args.probe_args),
                        ("--no-gpu", args.no_gpu), ("--no-suites", args.no_suites),
                        ("--spoof-devices", args.spoof_devices)):
        if value:
            raise SystemExit(f"{flag} belongs to PR mode; a prebuilt differential has no PR, no source "
                             f"tree, no test suites and no torch. Drop it.")
    if args.tests != ["tests/"]:
        raise SystemExit("--tests belongs to the pytest probe; the prebuilt probe takes --cells")
    if not args.model:
        raise SystemExit("--prebuilt needs --model REPO[@REV]:FILE[,FILE...] or REPO[@REV]:FOLDER")
    tags = [t.strip() for t in args.prebuilt.split(",") if t.strip()]
    if len(tags) != 2 or tags[0] == tags[1]:
        raise SystemExit(f"--prebuilt wants two different tags, base then head; got {tags}")

    model = prebuilt.parse_spec(args.model)
    if prebuilt.needs_resolution(model):
        model = prebuilt.hf_resolve(model)
        print(f"resolved {args.model} -> {model.repo}@{model.revision[:12]}: "
              f"{len(model.paths)} file(s), {model.total_bytes / 1e9:.1f} GB")
    sentinel = prebuilt.parse_spec(args.sentinel)
    if prebuilt.needs_resolution(sentinel):
        sentinel = prebuilt.hf_resolve(sentinel)
    if len(sentinel.paths) != 1:
        raise SystemExit(f"--sentinel must name exactly one file, got {sentinel.paths}")

    probe = args.probe if args.probe != "amd_ci/probes/pytest_probe.py" else prebuilt.DEFAULT_PROBE
    criteria = (args.criteria if args.criteria != "amd_ci/criteria/pytest_no_regression.py"
                else prebuilt.DEFAULT_CRITERIA)
    gb = model.total_bytes / 1e9
    min_free = args.min_free_gb if args.min_free_gb is not None else (int(gb * 1.3) + 25 if gb else 150)
    plan = prebuilt.Plan(
        base_tag = tags[0], head_tag = tags[1], asset = args.asset, windows = args.windows,
        release_repo = args.release_repo, model = model, sentinel = sentinel,
        env = args.env, unset_env = args.unset_env, control_env = args.control_env,
        control_assets = args.control_asset, control_tags = args.control_tag,
        probe = probe, criteria = criteria, cells = args.cells, n_predict = args.n_predict,
        load_timeout = args.load_timeout, timeout_minutes = args.timeout_minutes,
        min_free_gb = min_free, title = args.title or "", branch = args.branch or "")
    branch = plan.branch or plan.default_branch()

    out = args.out
    refuse_existing_workflows(out)
    out.mkdir(parents = True, exist_ok = True)
    copy_toolkit(out)
    template = ROOT / "templates" / ("workflow_prebuilt_windows.yml" if args.windows
                                     else "workflow_prebuilt.yml")
    text = prebuilt.render(template.read_text(encoding = "utf-8"), plan)
    wf, rc = write_and_lint(out, branch, text)
    if rc != 0:
        return rc
    arms = ", ".join(n for n, _, _ in plan.controls()) or "none"
    print(f"\nstates: base {plan.base_tag} / head {plan.head_tag} on {plan.os_word} {plan.asset}\n"
          f"model: {model.repo}@{model.revision[:12]} entry {model.entry}\n"
          f"controls: {arms}")
    print(push_recipe(out, branch, f"Prebuilt differential {plan.base_tag} vs {plan.head_tag}"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", type = int, default = None, help = "the PR to validate (PR mode)")
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--branch", default = None)
    ap.add_argument("--merged", action = "store_true")
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
    ap.add_argument("--spoof-devices", type = int, default = 0, metavar = "N",
                    help = "present N EXTRA HIP devices to torch via LD_PRELOAD, so "
                           "multi-GPU code paths become reachable on this one-GPU "
                           "runner. The extra devices are the real GPU wearing other "
                           "numbers: good for selection and index logic, useless for "
                           "sharding, throughput or collectives. capability.py is told "
                           "and forces multi_gpu to stay UNMET, so the verdict still "
                           "declares the gap. See lib/device_multiplier.py.")
    pb = ap.add_argument_group("prebuilt mode", "a differential over two release tags; no PR, no source")
    pb.add_argument("--prebuilt", default = None, metavar = "BASE_TAG,HEAD_TAG",
                    help = "two release tags of --release-repo; the head is the build under test")
    pb.add_argument("--release-repo", default = prebuilt.DEFAULT_RELEASE_REPO)
    pb.add_argument("--asset", default = "rocm-gfx1151",
                    help = "asset suffix for base and head; the OS prefix and extension are added "
                           "(rocm-gfx1151 -> app-TAG-linux-x64-rocm-gfx1151.tar.gz, or the Windows zip)")
    pb.add_argument("--model", default = None, metavar = "REPO[@REV]:FILE[,FILE...]|REPO[@REV]:FOLDER",
                    help = "the GGUF to serve. A folder is expanded into its .gguf files through the "
                           "Hub API and an unpinned revision is pinned to the current commit; name "
                           "the files and the revision to scaffold offline")
    pb.add_argument("--sentinel", default = prebuilt.DEFAULT_SENTINEL, metavar = "REPO@REV:FILE",
                    help = "small known-good GGUF run before and after every state's cells")
    pb.add_argument("--env", action = "append", default = [], metavar = "K=V",
                    help = "environment for the differential (both states)")
    pb.add_argument("--unset-env", action = "append", default = [], metavar = "K")
    pb.add_argument("--control-env", action = "append", default = [], metavar = "absent|K=V",
                    help = "a control arm: the head build with the --env variables absent, or one of them set to V")
    pb.add_argument("--control-asset", action = "append", default = [], metavar = "ASSET",
                    help = "a control arm: the head tag on another asset (say vulkan), same environment")
    pb.add_argument("--control-tag", action = "append", default = [], metavar = "TAG",
                    help = "a control arm: another tag on the same asset, same environment")
    pb.add_argument("--cells", default = "single,multiseg,unified4")
    pb.add_argument("--n-predict", type = int, default = 128)
    pb.add_argument("--load-timeout", type = int, default = 2400)
    pb.add_argument("--timeout-minutes", type = int, default = 480)
    pb.add_argument("--min-free-gb", type = int, default = None,
                    help = "free disk the gate demands; derived from the model size when the Hub was asked")
    pb.add_argument("--title", default = None)
    args = ap.parse_args()

    if args.prebuilt:
        return scaffold_prebuilt(args)
    if args.pr is None:
        raise SystemExit("--pr is required (or use --prebuilt for a release differential)")

    branch = args.branch or f"amd-ci-pr{args.pr}"
    out = args.out

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

    refuse_existing_workflows(out)
    out.mkdir(parents = True, exist_ok = True)

    copy_toolkit(out)

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
        diff_args = args.probe_args

    text = template.read_text(encoding = "utf-8")
    text = (text
            .replace("__DIFF_ARGS__", diff_args)
            .replace("__PR__", str(args.pr))
            .replace("__BRANCH__", branch)
            .replace("__MERGED__", "1" if args.merged else "0")
            .replace("__TESTS__", " ".join(args.tests))
            .replace("__PROBE__", args.probe)
            .replace("__CRITERIA__", args.criteria)
            .replace("__PROBE_ARGS__", args.probe_args))

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

    if args.no_gpu:
        text = text.split("  gpu:")[0].rstrip() + "\n"
    if args.no_suites:
        head, _, tail = text.partition("  suites:")
        text = head + "  gpu:" + tail.partition("  gpu:")[2]

    wf, rc = write_and_lint(out, branch, text)
    if rc != 0:
        return rc
    print(push_recipe(out, branch, f"Validate PR {args.pr} on gfx1151"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
