#!/usr/bin/env python3
"""Push the bisect harness + workflow to a disposable branch on each staging repo.

Mirrors pr_review/staging_ci.py rules: branch from FRESH upstream main, never touch staging
main, push with the keyring identity (GH_TOKEN stripped), disposable branch only.

    python stage_branch.py push  [--repos a,b,c] [--branch bisect-update-ci]
    python stage_branch.py runs  [--repos ...]
    python stage_branch.py close [--repos ...]      # delete the branch
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys

WS = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "pr_review"))
import staging_ci as sc  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPOS = sc.STAGING["unslothai/unsloth"]
# report.py is on this list because the workflow's own jobs call it (--step-summary / --assert).
HARNESS_FILES = ["run_step.sh", "run_step.ps1", "connect_proxy.py", "snapshot.py", "ci_plan.py", "ci_plan.json",
                 "pr_wheel.py", "idem.py", "report.py", "pins"]


def push(repos, branch, source_repo=None, source_ref=None):
    """source_repo/source_ref: push THAT tree (a local PR worktree branch) instead of upstream/main."""
    for repo in repos:
        d = sc.ensure_staging_clone(repo, "unslothai/unsloth")
        if source_repo:
            sc._git(["fetch", "-q", str(source_repo), source_ref], d)
            sc._git(["checkout", "-B", branch, "FETCH_HEAD"], d)
        else:
            sc._git(["checkout", "-B", branch, "upstream/main"], d)
        dest = d / "scripts" / "bisect"
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        for name in HARNESS_FILES:
            src = HERE / name
            if src.is_dir():
                shutil.copytree(src, dest / name)
            else:
                shutil.copy2(src, dest / name)
        wf = d / ".github" / "workflows" / "bisect-update.yml"
        shutil.copy2(HERE / "bisect-update.yml", wf)
        sc._git(["add", "scripts/bisect", ".github/workflows/bisect-update.yml"], d)
        r = sc._run(["git", "-C", str(d), "commit", "-q", "-m", "Bisect harness: time Studio install and update across releases 800-807, and verify a PR wheel"])
        if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
            print(f"[{repo}] commit failed: {r.stderr.strip()[:300]}")
            continue
        sc._lease_ref(d, branch)
        p = sc._run(["git", "-C", str(d), "push", "--force-with-lease", "origin", f"{branch}:{branch}"])
        if p.returncode != 0:
            p = sc._run(["git", "-C", str(d), "push", "--force", "origin", f"{branch}:{branch}"])
        head = sc._git(["rev-parse", "--short", "HEAD"], d)
        print(f"[{repo}] branch {branch} @ {head} pushed={p.returncode == 0} {p.stderr.strip()[-200:] if p.returncode else ''}")


def runs(repos, branch):
    for repo in repos:
        r = sc._run(["gh", "run", "list", "-R", repo, "--branch", branch, "--limit", "5", "--json", "databaseId,status,conclusion,createdAt,url,workflowName"])
        print(f"[{repo}]")
        try:
            for run in json.loads(r.stdout or "[]"):
                print(f"   {run['databaseId']} {run['status']:12} {str(run['conclusion']):10} {run['createdAt']} {run['url']}")
        except json.JSONDecodeError:
            print("   ", r.stderr.strip()[:200])


def close(repos, branch):
    for repo in repos:
        r = sc._run(["git", "-C", str(sc._clone_dir(repo)), "push", "origin", "--delete", branch])
        print(f"[{repo}] delete {branch}: rc={r.returncode} {r.stderr.strip()[-160:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "runs", "close"])
    ap.add_argument("--repos", default=",".join(REPOS))
    ap.add_argument("--branch", default="bisect-update-ci")
    ap.add_argument("--source-repo", default=None, help="local git dir/worktree whose --source-ref becomes the branch base")
    ap.add_argument("--source-ref", default=None)
    a = ap.parse_args()
    repos = [r for r in a.repos.split(",") if r]
    if a.cmd == "push":
        push(repos, a.branch, a.source_repo, a.source_ref)
    else:
        {"runs": runs, "close": close}[a.cmd](repos, a.branch)


if __name__ == "__main__":
    main()
