#!/usr/bin/env bash
# build.sh <label> <sha> <apply-patch 0|1> <vk|rocm> <out-dir>
# Builds unslothai/stable-diffusion.cpp at <sha> (+ ggml-patches in glob order, as the prebuilt resolve job),
# optionally with igpu_fit.patch on top (the head arm). Only src/core/backend_fit.cpp differs between arms.
set -euo pipefail
LABEL=$1; SHA=$2; PATCH=$3; KIND=$4; OUT=$5
H="$(cd "$(dirname "$0")" && pwd)"
SRC="$RUNNER_TEMP/src_$LABEL"
rm -rf "${SRC:?}"
git clone -q https://github.com/unslothai/stable-diffusion.cpp "$SRC"
cd "$SRC"
git checkout -q "$SHA"
git submodule update --init --depth 1 ggml
for p in scripts/unsloth/ggml-patches/*.patch; do
  [ -e "$p" ] || continue
  git -C ggml apply "../$p"
done
if [ "$PATCH" = 1 ]; then git apply --verbose "$H/igpu_fit.patch"; fi
if [ "$KIND" = vk ]; then
  EXTRA=(-DSD_VULKAN=ON)
else
  EXTRA=(-DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang" -DCMAKE_HIP_FLAGS="-mllvm --amdgpu-unroll-threshold-local=600"
         -DSD_HIPBLAS=ON -DHIP_PLATFORM=amd -DGPU_TARGETS=gfx1151)
fi
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON -DSD_SERVER_BUILD_FRONTEND=OFF \
  -DSD_WEBP=OFF -DSD_WEBM=OFF -DGGML_NATIVE=OFF "${EXTRA[@]}" > "$RUNNER_TEMP/cmake_$LABEL.log" 2>&1 \
  || { tail -60 "$RUNNER_TEMP/cmake_$LABEL.log"; exit 1; }
T0=$(date +%s)
cmake --build build --config Release -j "$(nproc)" --target sd-cli > "$RUNNER_TEMP/build_$LABEL.log" 2>&1 \
  || { grep -E "error" "$RUNNER_TEMP/build_$LABEL.log" | head -40; exit 1; }
echo "build $LABEL took $(( $(date +%s) - T0 )) s"
mkdir -p "$OUT"
cp -a build/bin/. "$OUT/"
{ echo "label=$LABEL sha=$(git rev-parse HEAD) patched=$PATCH ggml=$(git -C ggml rev-parse HEAD) kind=$KIND"; git diff --stat; } > "$OUT/srcinfo.txt"
cat "$OUT/srcinfo.txt"
