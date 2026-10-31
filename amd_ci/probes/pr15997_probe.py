#!/usr/bin/env python3
"""Probe for unsloth #15997 (state checkout) x unsloth-zoo #2252 (cloned here): thread-global torch config.

#15997's test module skips without unsloth_zoo.torch_config, which only #2252 adds, so the two PRs are
probed together. Arms (the state's unsloth checkout + one zoo worktree, both first on PYTHONPATH):
  base:  unsloth merge base + zoo PR head, running the HEAD's test files (copied in). Zoo's merge base
         has no unsloth_zoo.torch_config, so the new tests cannot even import there; with #2252 present
         the base isolates #15997's own change
  head:  unsloth PR head + zoo PR head
  merge: unsloth merge state + zoo PR head
Writes the pytest_probe.py schema (ids prefixed `unsloth/` or `zoo/`), judged by pytest_no_regression.
Observes only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ZOO_URL = "https://github.com/unslothai/unsloth-zoo"
CLEAR = ("PYTEST_ADDOPTS", "UNSLOTH_COMPILE_DISABLE", "TORCHDYNAMO_DISABLE", "UNSLOTH_FORCE_FLOAT32")


def git(args, cwd):
    p = subprocess.run(["git"] + args, cwd = cwd, capture_output = True, text = True)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr[-400:]}")
    return p.stdout


def zoo_worktree(work: Path, zoo_pr: int, which: str) -> Path:
    src = work / "zoo_src"
    if not src.is_dir():
        git(["clone", "-q", "--filter=blob:none", ZOO_URL, str(src)], cwd = work)
        git(["fetch", "-q", "origin", f"refs/pull/{zoo_pr}/head:zpr-head"], cwd = src)
    dst = work / f"zoo_{which}"
    if not dst.is_dir():
        head = git(["rev-parse", "zpr-head"], cwd = src).strip()
        sha = head if which == "head" else git(["merge-base", "zpr-head", "origin/main"], cwd = src).strip()
        git(["worktree", "add", "-q", "--detach", str(dst), sha], cwd = src)
    return dst


def copy_head_tests(checkout: Path, unsloth_pr: int, tests: list[str]) -> list[str]:
    missing = [t for t in tests if not (checkout / t).exists()]
    if missing:
        git(["fetch", "-q", "origin", f"refs/pull/{unsloth_pr}/head:upr-head"], cwd = checkout)
        for t in missing:
            (checkout / t).parent.mkdir(parents = True, exist_ok = True)
            (checkout / t).write_text(git(["show", f"upr-head:{t}"], cwd = checkout), encoding = "utf-8")
    return missing


def pytest(python, cwd, tests, env, timeout):
    have_timeout = subprocess.run([python, "-c", "import pytest_timeout"], capture_output = True).returncode == 0
    cmd = [python, "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider", "-o", "addopts="]
    if have_timeout:
        cmd += ["--timeout", str(timeout)]
    p = subprocess.run(cmd + tests, cwd = cwd, env = env, capture_output = True, text = True, timeout = timeout + 120)
    return p.returncode, p.stdout[-20000:], p.stderr[-2000:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required = True)
    ap.add_argument("--checkout", required = True)
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--python", default = sys.executable)
    ap.add_argument("--timeout", type = int, default = 1200)
    ap.add_argument("--unsloth-pr", type = int, default = 15997)
    ap.add_argument("--zoo-pr", type = int, default = 2252)
    ap.add_argument("--unsloth-tests", nargs = "+", default = ["tests/test_compile_config_threads.py"])
    ap.add_argument("--zoo-tests", nargs = "+", default = ["tests/test_torch_config_threads.py"])
    args = ap.parse_args()

    checkout = Path(args.checkout).resolve()
    work = args.out.resolve().parent / "pr15997_work"
    work.mkdir(parents = True, exist_ok = True)
    obs: dict = {"state": args.state, "workdir": str(checkout), "absent_at_this_state": []}
    try:
        zoo = zoo_worktree(work, args.zoo_pr, "head")
        obs["zoo_checkout"] = str(zoo)
        obs["copied_head_tests"] = copy_head_tests(checkout, args.unsloth_pr, args.unsloth_tests)
        for t in args.zoo_tests:
            if not (zoo / t).exists():
                src = work / "zoo_src"
                (zoo / t).write_text(git(["show", f"zpr-head:{t}"], cwd = src), encoding = "utf-8")
                obs["copied_head_tests"].append("zoo/" + t)
    except Exception as e:
        obs["error"] = f"setup: {e}"
        args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
        return 0

    env = {k: v for k, v in os.environ.items() if k not in CLEAR}
    env["PYTHONPATH"] = os.pathsep.join([str(checkout), str(zoo)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["UNSLOTH_IS_PRESENT"] = "1"
    rcs, tails, errs = [], [], []
    failed, errors, counts = set(), set(), {"passed": 0, "failed": 0, "skipped": 0}
    for prefix, cwd, tests in (("unsloth", checkout, args.unsloth_tests), ("zoo", zoo, args.zoo_tests)):
        rc, out, err = pytest(args.python, cwd, tests, env, args.timeout)
        rcs.append(rc)
        tails.append(f"[{prefix}] " + out[-3000:])
        errs.append(f"[{prefix}] " + err[-800:])
        failed |= {f"{prefix}/{x}" for x in re.findall(r"^FAILED\s+(\S+)", out, re.M)}
        errors |= {f"{prefix}/{x}" for x in re.findall(r"^ERROR\s+(\S+)", out, re.M)}
        for k in counts:
            m = re.search(rf"(\d+) {k}", out)
            counts[k] += int(m.group(1)) if m else 0
    # 0 and 1 mean pytest ran and judged; report 1 if either suite failed, else the worst code.
    obs["rc"] = 1 if 1 in rcs and all(r in (0, 1) for r in rcs) else max(rcs, key = lambda r: (r not in (0, 1), r))
    obs["rcs"] = rcs
    obs["tail"] = "\n".join(tails)[-4000:]
    obs["stderr_tail"] = "\n".join(errs)[-2000:]
    obs["failed"], obs["errors"] = sorted(failed), sorted(errors)
    obs["n_passed"], obs["n_failed"], obs["n_skipped"] = counts["passed"], counts["failed"], counts["skipped"]
    args.out.write_text(json.dumps(obs, indent = 2), encoding = "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
