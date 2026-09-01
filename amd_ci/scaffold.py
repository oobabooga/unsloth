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
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Must match pytest_probe.py's --subdir default; selftest.py asserts they agree.
PYTEST_SUBDIR = "studio/backend"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", type = int, required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--branch", default = None)
    ap.add_argument("--merged", action = "store_true")
    ap.add_argument("--tests", nargs = "*", default = ["tests/"])
    ap.add_argument("--probe", default = "amd_ci/probes/pytest_probe.py")
    ap.add_argument("--criteria", default = "amd_ci/criteria/pytest_no_regression.py")
    ap.add_argument("--probe-args", default = "")
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
    args = ap.parse_args()

    branch = args.branch or f"amd-ci-pr{args.pr}"
    out = args.out

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
    shutil.copy(ROOT / "templates" / "workflow.yml", dest / "templates" / "workflow.yml")

    text = (ROOT / "templates" / "workflow.yml").read_text()
    text = (text
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

    wf_dir = out / ".github" / "workflows"
    wf_dir.mkdir(parents = True, exist_ok = True)
    wf = wf_dir / f"{branch}.yml"
    wf.write_text(text)
    (out / ".gitignore").write_text("__pycache__/\n*.pyc\n")

    print(f"wrote {wf}")
    rc = subprocess.run([sys.executable, str(ROOT / "lib" / "lint_workflow.py"), str(wf)]).returncode
    if rc != 0:
        print("\nlint found error-level problems; fix them before pushing")
        return rc

    print(f"""
next:
  cd {out} && git init -q && git checkout -q -b {branch}
  git add -A && git commit -q -m "Validate PR {args.pr} on gfx1151"
  git remote add ooba https://github.com/oobabooga/unsloth.git
  git push -q ooba HEAD:refs/heads/{branch}
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
