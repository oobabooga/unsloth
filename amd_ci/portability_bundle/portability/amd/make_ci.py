#!/usr/bin/env python3
"""Build (and optionally push) the AMD CI branches that run the portability harness on gfx1151, Linux and Windows.

    python make_ci.py                    # -> $WORKSPACE/ci_portability, ci_portability_win (lint-clean), prints push cmds
    python make_ci.py --push             # commit as Daniel Han and push to oobabooga/unsloth (the push starts the run)
    python make_ci.py --cases int8_gemm  # restrict the cases (both OSes)

Each branch holds ONLY the workflow plus amd_ci/ (the toolkit and amd_ci/portability_bundle, a fresh snapshot.py
bundle), per workflows/amd_ci_workflow.md. The workflow installs AMD's gfx1151 torch wheels (repo.amd.com), gates
on rocm + gfx1151, runs harness.py --quick, appends capability.py's "Not tested here" and uploads out/ as an artifact.
Collect with:  gh run download <run> --repo oobabooga/unsloth -D <dir> && python ../collect.py <dir>
No secrets are used: every case runs on synthetic tensors.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = os.path.dirname(HERE)
sys.path.insert(0, PORT)
import snapshot  # noqa: E402

WS = snapshot.WS
FORK = "https://github.com/oobabooga/unsloth.git"
LEGS = {"linux": ("ci_portability", "amd-ci-portability", "workflow_linux.yml", "amd-ci-portability.yml"),
        "windows": ("ci_portability_win", "amd-ci-portability-win", "workflow_windows.yml",
                    "amd-ci-portability-win.yml")}


def sh(*a, cwd = None):
    return subprocess.run(a, cwd = cwd, check = True, capture_output = True, text = True).stdout


def build_leg(leg: str, cases: str, bundle: str) -> str:
    d, branch, tmpl, wf = LEGS[leg]
    out = os.path.join(WS, d)
    git_dir = os.path.join(out, ".git")
    keep_git = os.path.isdir(git_dir)
    for name in os.listdir(out) if os.path.isdir(out) else []:
        if name != ".git":
            p = os.path.join(out, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(os.path.join(out, ".github", "workflows"), exist_ok = True)
    shutil.copytree(os.path.join(WS, "amd_ci"), os.path.join(out, "amd_ci"),
                    ignore = shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(bundle, os.path.join(out, "amd_ci", "portability_bundle"),
                    ignore = shutil.ignore_patterns("__pycache__", "*.pyc"))
    with open(os.path.join(HERE, tmpl), encoding = "utf-8") as f:
        text = f.read()
    if cases != "all":
        text = text.replace("--cases all", f"--cases {cases}")
    with open(os.path.join(out, ".github", "workflows", wf), "w", encoding = "utf-8") as f:
        f.write(text)
    lint = subprocess.run([sys.executable, os.path.join(WS, "amd_ci", "lib", "lint_workflow.py"),
                           os.path.join(out, ".github", "workflows", wf)], capture_output = True, text = True)
    print(lint.stdout.strip())
    if lint.returncode != 0:
        raise SystemExit(f"lint failed for {wf}")
    if not keep_git:
        sh("git", "init", "-q", cwd = out)
        sh("git", "checkout", "-q", "-b", branch, cwd = out)
        sh("git", "remote", "add", "ooba", FORK, cwd = out)
    return out


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legs", default = "linux,windows")
    ap.add_argument("--cases", default = "all")
    ap.add_argument("--push", action = "store_true")
    ap.add_argument("--message", default = "Portability harness on gfx1151")
    a = ap.parse_args()
    bundle = os.path.join(WS, "temp", "portability_bundle")
    snapshot.build(bundle, os.path.join(WS, "wt_port_h3"), "origin/studio-h3-vae-fast", "origin/studio-nvfp4-kernels",
                   "origin/main", fetch = True)
    for leg in a.legs.split(","):
        out = build_leg(leg, a.cases, bundle)
        branch = LEGS[leg][1]
        if a.push:
            sh("git", "add", "-A", cwd = out)
            subprocess.run(["git", "-c", "user.name=Daniel Han", "-c", "user.email=danielhanchen@gmail.com", "commit",
                            "-q", "-m", f"{a.message} ({leg})"], cwd = out, check = False)
            sh("git", "push", "-q", "ooba", f"HEAD:refs/heads/{branch}", cwd = out)
            print(f"pushed {branch}: gh run list --repo oobabooga/unsloth --branch {branch} --limit 1")
        else:
            print(f"{leg}: cd {out} && git add -A && git commit -m '{a.message}' && "
                  f"git push -q ooba HEAD:refs/heads/{branch}")


if __name__ == "__main__":
    main()
