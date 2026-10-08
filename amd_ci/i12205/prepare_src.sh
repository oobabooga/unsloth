#!/usr/bin/env bash
# The proposed prebuild step's source half: rocm-systems pin + TheRock patches from the
# nightly's own manifest, sparse fetch of rocr-runtime, then an unpatched and a patched tree.
set -euo pipefail
ROCM="${ROCM:-/opt/rocm}"
WORK="${WORK:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
manifest="$ROCM/share/therock/therock_manifest.json"
test -f "$manifest"
sha="$(jq -r '.submodules[] | select(.submodule_name == "rocm-systems") | .pin_sha' "$manifest")"
therock_sha="$(jq -r '.the_rock_commit // empty' "$manifest")"
test -n "$sha" && test "$sha" != null
echo "rocm-systems pin: $sha (TheRock ${therock_sha:-unknown})"
src="$WORK/src_unpatched"
git init -q "$src"
git -C "$src" remote add origin https://github.com/ROCm/rocm-systems
git -C "$src" sparse-checkout set projects/rocr-runtime
git -C "$src" fetch -q --depth 1 --filter=blob:none origin "$sha"
git -C "$src" checkout -q FETCH_HEAD
if [ -n "$therock_sha" ]; then
  jq -r '.submodules[] | select(.submodule_name == "rocm-systems") | .patches[]?' "$manifest" | while read -r p; do
    [ -n "$p" ] || continue
    echo "applying TheRock patch $p"
    curl -fsSL --retry 5 "https://raw.githubusercontent.com/ROCm/TheRock/${therock_sha}/${p}" | git -C "$src" apply --include='projects/rocr-runtime/*' -
  done
fi
cp -a "$src" "$WORK/src_patched"
python3 "$HERE/patch_queues.py" "$WORK/src_patched/projects/rocr-runtime/libhsakmt/src/queues.c"
git -C "$WORK/src_patched" diff --stat
