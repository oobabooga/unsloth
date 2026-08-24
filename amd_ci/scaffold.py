#!/usr/bin/env python3
"""Create a ready-to-push AMD CI branch for a PR.

Produces a directory containing the toolkit plus one generated workflow, lints
it, and prints the push command. The lint is not optional: it is the step that
turns "a run I will lose in ten minutes" into "a message I read in one second".

  python amd_ci/scaffold.py --pr 9487 --out ci_pr9487 \\
      --tests studio/backend/tests/test_sd_cpp_install.py

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
    args = ap.parse_args()

    branch = args.branch or f"amd-ci-pr{args.pr}"
    out = args.out

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
  env -u GH_TOKEN git push -q ooba HEAD:refs/heads/{branch}
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
