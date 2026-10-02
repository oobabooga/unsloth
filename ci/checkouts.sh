#!/usr/bin/env bash
# main at the PR's base, and the same tree with the branch's change applied.
set -euxo pipefail
W="$1"
CI="$(cd "$(dirname "$0")" && pwd)"
git clone -q --filter=blob:none https://github.com/unslothai/unsloth "$W/main"
git -C "$W/main" checkout -q "$(cat "$CI/base.txt")"
cp -a "$W/main" "$W/fix"
git -C "$W/fix" apply --whitespace=nowarn "$CI/fix.patch"
git -C "$W/fix" status --short
