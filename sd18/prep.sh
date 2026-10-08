#!/usr/bin/env bash
# usage: prep.sh <patched|baseline> <workdir>
# patched : sd.cpp PR #18 head, all ggml-patches applied (what PR #18 ships)
# baseline: sd.cpp PR #17 head, its patches (0001 only) applied, plus only the
#           tests/test-backend-ops.cpp hunk of PR #18's 0002 so FA cases match.
set -euo pipefail
VARIANT=$1
W=$2
HERE=$(cd "$(dirname "$0")" && pwd)
PR18=5f222b85a9b52e7d6f43fdffcb0772806459154b
PR17=ca92d95ddf041bf885a36a7fdafbffd88706c28e
GGML=f583f393cd5dfdc129360bbf75cb3d49ccc837a8
mkdir -p "$W"
cd "$W"
git -c core.autocrlf=false clone -q https://github.com/unslothai/stable-diffusion.cpp sd
cd sd
git config core.autocrlf false
git fetch -q origin pull/18/head:pr18 pull/17/head:pr17
echo "pr18 ref now: $(git rev-parse pr18) (pinned $PR18)"
echo "pr17 ref now: $(git rev-parse pr17) (pinned $PR17)"
git show $PR18:scripts/unsloth/ggml-patches/0002-ggml-cuda-fa-longseq.patch > "$W/0002.patch"
if [ "$VARIANT" = patched ]; then git checkout -q $PR18; else git checkout -q $PR17; fi
test "$(git ls-tree HEAD ggml | awk '{print $3}')" = "$GGML"
if ! git -c core.autocrlf=false submodule update --init --depth 1 ggml; then
  rm -rf ggml && git init -q ggml
  git -C ggml fetch -q --depth 1 https://github.com/leejet/ggml.git $GGML
  git -C ggml checkout -q FETCH_HEAD
fi
test "$(git -C ggml rev-parse HEAD)" = "$GGML"
ls scripts/unsloth/ggml-patches/
for p in scripts/unsloth/ggml-patches/*.patch; do
  echo "== git -C ggml apply --verbose ../$p"
  git -C ggml apply --verbose "../$p" > "$W/apply.log" 2>&1 || { cat "$W/apply.log"; exit 1; }
  grep -c '^Applied patch' "$W/apply.log" | sed 's/^/files applied: /'
done
if [ "$VARIANT" = baseline ]; then
  echo "== baseline: test hunk of PR #18 0002 only"
  git -C ggml apply --verbose --include=tests/test-backend-ops.cpp "$W/0002.patch"
fi
"${PYTHON:-python3}" "$HERE/inject.py" ggml/tests/test-backend-ops.cpp
git -C ggml status --short
git -C ggml diff --stat | tail -n 1
