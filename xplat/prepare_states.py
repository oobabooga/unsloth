#!/usr/bin/env python3
"""Materialise the base and head llama.cpp trees for a pin-set change from public sources. Stdlib + git.

base = ggml-org/llama.cpp at the upstream tag + pins_base.diff (the merged pin set as it is)
head = base + change.diff (the pin set with the one repin)
Each tree's git tree hash must equal the one recorded in states.json (the llama_pinset sim result),
so every platform builds byte-identical sources.

  python prepare_states.py --kit xplat --root WORKDIR   ->  WORKDIR/src/base, WORKDIR/src/head
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def git(*a, cwd):
    cmd = ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "-c", "user.name=xplat",
           "-c", "user.email=xplat@localhost", *a]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"FATAL: {' '.join(cmd)} -> {p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    kit = Path(a.kit).resolve()
    st = json.loads((kit / "states.json").read_text(encoding="utf-8"))
    src = Path(a.root).resolve() / "src"
    base, head = src / "base", src / "head"
    base.mkdir(parents=True, exist_ok=True)
    git("init", "-q", cwd=base)
    git("fetch", "-q", "--depth", "1", st["upstream"], f"refs/tags/{st['tag']}:refs/tags/{st['tag']}", cwd=base)
    git("checkout", "-q", "--detach", st["tag"], cwd=base)
    got = git("rev-parse", "HEAD^{tree}", cwd=base)
    if got != st["tag_tree"]:
        sys.exit(f"FATAL: upstream {st['tag']} tree {got} != {st['tag_tree']}")
    for diff, d, want, name in ((kit / "pins_base.diff", base, st["base_tree"], "base"),):
        git("apply", "--binary", "--whitespace=nowarn", str(diff), cwd=d)
        git("add", "-A", cwd=d)
        got = git("write-tree", cwd=d)
        if got != want:
            sys.exit(f"FATAL: {name} tree {got} != expected {want}")
        git("commit", "-q", "-m", name, cwd=d)
    git("worktree", "add", "-q", "--detach", str(head), "HEAD", cwd=base)
    git("apply", "--binary", "--whitespace=nowarn", str(kit / "change.diff"), cwd=head)
    git("add", "-A", cwd=head)
    got = git("write-tree", cwd=head)
    if got != st["head_tree"]:
        sys.exit(f"FATAL: head tree {got} != expected {st['head_tree']}")
    git("commit", "-q", "-m", "head", cwd=head)
    print(json.dumps({"base": str(base), "head": str(head), "tag": st["tag"],
                      "base_tree": st["base_tree"], "head_tree": st["head_tree"]}))


if __name__ == "__main__":
    main()
