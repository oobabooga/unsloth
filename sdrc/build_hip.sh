#!/usr/bin/env bash
# build_hip.sh <label> <sha> <out-dir> [extra ggml patch file ...]
# HIP (gfx1151 only) build of unslothai/stable-diffusion.cpp at <sha>, ggml-patches applied in glob order like the
# resolve job; extra patch files are added to scripts/unsloth/ggml-patches first (as a fix PR would add them).
# Builds sd-cli, test-backend-ops and the harness-only sdxp-fa-probe. Uses whatever hipconfig is first on PATH.
set -euo pipefail
LABEL=$1; SHA=$2; OUT=$3; shift 3
H="$(cd "$(dirname "$0")" && pwd)"
SRC="$RUNNER_TEMP/src_hip_$LABEL"
rm -rf "${SRC:?}"
mkdir -p "$SRC"
cd "$SRC"
git init -q .
git remote add origin "${SRC_REPO:-https://github.com/unslothai/stable-diffusion.cpp}"
git fetch -q --depth 1 origin "$SHA"
git checkout -q FETCH_HEAD
git submodule update --init --depth 1 ggml
for x in "$@"; do cp "$x" scripts/unsloth/ggml-patches/; done
for p in scripts/unsloth/ggml-patches/*.patch; do
  [ -e "$p" ] || continue
  git -C ggml apply "../$p"
done
# harness only (identical in every build): the FA probe as a ggml test target
cp "$H/fa_probe.cpp" ggml/tests/sdxp-fa-probe.cpp
printf '\nadd_executable(sdxp-fa-probe sdxp-fa-probe.cpp)\ntarget_link_libraries(sdxp-fa-probe PRIVATE ggml)\n' >> ggml/tests/CMakeLists.txt
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON -DSD_SERVER_BUILD_FRONTEND=OFF \
  -DSD_WEBP=OFF -DSD_WEBM=OFF -DGGML_NATIVE=OFF -DGGML_BUILD_TESTS=ON \
  -DSD_HIPBLAS=ON -DHIP_PLATFORM=amd -DGPU_TARGETS=gfx1151 \
  "-DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang" "-DCMAKE_HIP_FLAGS=-mllvm --amdgpu-unroll-threshold-local=600" \
  > "$RUNNER_TEMP/cmake_hip_$LABEL.log" 2>&1 || { tail -60 "$RUNNER_TEMP/cmake_hip_$LABEL.log"; exit 1; }
T0=$(date +%s)
cmake --build build --config Release -j "$(nproc)" --target sd-cli test-backend-ops sdxp-fa-probe \
  > "$RUNNER_TEMP/build_hip_$LABEL.log" 2>&1 || { grep -E "error|Error" "$RUNNER_TEMP/build_hip_$LABEL.log" | head -40; exit 1; }
echo "build hip $LABEL took $(( $(date +%s) - T0 )) s"
mkdir -p "$OUT"
cp -a build/bin/. "$OUT/"
for t in test-backend-ops sdxp-fa-probe; do
  f="$(find build -name "$t" -type f | head -1)"
  [ -n "$f" ] && cp "$f" "$OUT/"
done
{ echo "label=$LABEL kind=hip sha=$(git rev-parse HEAD) ggml=$(git -C ggml rev-parse HEAD) hipcc=$(hipconfig -l)"; ls scripts/unsloth/ggml-patches; } > "$OUT/srcinfo.txt"
ls -la "$OUT"
