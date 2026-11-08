#!/usr/bin/env bash
# build.sh <label> <sha> <vk|cpu|hipvk> <out-dir>
# Builds unslothai/stable-diffusion.cpp at <sha> like the matching unsloth-sd-prebuilt.yml leg
# (vk = Linux-Ubuntu-22.04-x86_64-vulkan, cpu = Linux-Ubuntu-22.04-x86_64 CPU-all-variants),
# hipvk = HIP (gfx1151 only) + Vulkan in one binary (built on the gfx1151 box, for the #51 two-backend check).
# ggml-patches are applied in glob order onto the ggml submodule, like the resolve job.
set -euo pipefail
LABEL=$1; SHA=$2; KIND=$3; OUT=$4
H="$(cd "$(dirname "$0")" && pwd)"
SRC="$RUNNER_TEMP/src_${KIND}_$LABEL"
rm -rf "${SRC:?}"
mkdir -p "$SRC"
cd "$SRC"
git init -q .
git remote add origin https://github.com/unslothai/stable-diffusion.cpp
git fetch -q --depth 1 origin "$SHA"
git checkout -q FETCH_HEAD
git submodule update --init --depth 1 ggml
for p in scripts/unsloth/ggml-patches/*.patch; do
  [ -e "$p" ] || continue
  git -C ggml apply "../$p"
done
TBO=OFF
if [ "$KIND" = vk ]; then
  # harness only: RMS_NORM perf cases for test-backend-ops (identical in every build)
  python3 "$H/tbo_perf_cases.py" ggml/tests/test-backend-ops.cpp
  TBO=ON
fi
case "$KIND" in
  vk) EXTRA=(-DSD_VULKAN=ON) ;;
  cpu) EXTRA=(-DSD_BUILD_SHARED_GGML_LIB=ON -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON) ;;
  hipvk) EXTRA=(-DSD_VULKAN=ON -DSD_HIPBLAS=ON -DHIP_PLATFORM=amd -DGPU_TARGETS=gfx1151
                "-DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang"
                "-DCMAKE_HIP_FLAGS=-mllvm --amdgpu-unroll-threshold-local=600") ;;
  *) echo "bad kind $KIND"; exit 2 ;;
esac
TARGETS=(sd-cli)
[ "$TBO" = ON ] && TARGETS+=(test-backend-ops)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON -DSD_SERVER_BUILD_FRONTEND=OFF \
  -DSD_WEBP=OFF -DSD_WEBM=OFF -DGGML_NATIVE=OFF -DGGML_BUILD_TESTS=$TBO "${EXTRA[@]}" > "$RUNNER_TEMP/cmake_${KIND}_$LABEL.log" 2>&1 \
  || { tail -60 "$RUNNER_TEMP/cmake_${KIND}_$LABEL.log"; exit 1; }
T0=$(date +%s)
cmake --build build --config Release -j "$(nproc)" --target "${TARGETS[@]}" > "$RUNNER_TEMP/build_${KIND}_$LABEL.log" 2>&1 \
  || { tail -80 "$RUNNER_TEMP/build_${KIND}_$LABEL.log"; exit 1; }
echo "build $KIND $LABEL took $(( $(date +%s) - T0 )) s"
mkdir -p "$OUT"
cp -a build/bin/. "$OUT/"
if [ "$TBO" = ON ]; then
  T="$(find build -name test-backend-ops -type f | head -1)"
  [ -n "$T" ] && cp "$T" "$OUT/"
fi
{ echo "label=$LABEL kind=$KIND sha=$(git rev-parse HEAD) ggml=$(git -C ggml rev-parse HEAD)"; ls scripts/unsloth/ggml-patches; } > "$OUT/srcinfo.txt"
ls -la "$OUT"
