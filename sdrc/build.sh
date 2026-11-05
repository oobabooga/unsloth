#!/usr/bin/env bash
# build.sh <label> <pr-number|-> <sha> <vk|cpu> <out-dir>
# Builds unslothai/stable-diffusion.cpp at <sha> exactly like the matching unsloth-sd-prebuilt.yml leg
# (Linux-Ubuntu-22.04-x86_64-vulkan or Linux-Ubuntu-22.04-x86_64), plus ggml's test-backend-ops.
# ggml-patches are applied in glob order like the resolve job.
set -euo pipefail
LABEL=$1; PR=$2; SHA=$3; KIND=$4; OUT=$5
H="$(cd "$(dirname "$0")" && pwd)"
SRC="$RUNNER_TEMP/src_$LABEL"
rm -rf "${SRC:?}"
git clone -q https://github.com/unslothai/stable-diffusion.cpp "$SRC"
cd "$SRC"
git fetch -q origin "$SHA" || git fetch -q origin "refs/pull/$PR/head"
git checkout -q "$SHA"
git submodule update --init --depth 1 ggml
for p in scripts/unsloth/ggml-patches/*.patch; do
  [ -e "$p" ] || continue
  git -C ggml apply --verbose "../$p"
done
# harness only: RMS_NORM perf cases for test-backend-ops (identical in every build)
python3 "$H/tbo_perf_cases.py" ggml/tests/test-backend-ops.cpp
EXTRA=""
if [ "$KIND" = vk ]; then
  EXTRA="-DSD_VULKAN=ON"
else
  EXTRA="-DSD_BUILD_SHARED_GGML_LIB=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON"
fi
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON -DSD_SERVER_BUILD_FRONTEND=OFF \
  -DSD_WEBP=OFF -DSD_WEBM=OFF -DGGML_NATIVE=OFF -DGGML_BUILD_TESTS=ON $EXTRA > "$RUNNER_TEMP/cmake_$LABEL.log" 2>&1 \
  || { tail -60 "$RUNNER_TEMP/cmake_$LABEL.log"; exit 1; }
T0=$(date +%s)
cmake --build build --config Release -j "$(nproc)" --target sd-cli sd-server test-backend-ops
echo "build $LABEL took $(( $(date +%s) - T0 )) s"
mkdir -p "$OUT"
cp -a build/bin/. "$OUT/"
cp "$(find build -name test-backend-ops -type f | head -1)" "$OUT/" 2>/dev/null || true
{ echo "label=$LABEL pr=$PR sha=$(git rev-parse HEAD) ggml=$(git -C ggml rev-parse HEAD) kind=$KIND"; ls scripts/unsloth/ggml-patches; } > "$OUT/srcinfo.txt"
ls -la "$OUT"
