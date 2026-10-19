#!/bin/bash
# linux_build.sh: k6 = upstream ggml-org b11721 (ff5a2337) + k6p/k6.patch.gz (= moe-combined-k6 tip, K6_SHA), HIP gfx1151 build
# with the release's backend flags (unsloth-prebuilt-rocm.yml linux leg: shared libs, GGML_BACKEND_DL, CPU_ALL_VARIANTS, RPC,
# OPENMP off, ROCWMMA FA off, NATIVE off), system /opt/rocm. Tests on for test-backend-ops. Output: $AMD_CI_WORK/k6bin (+ TAG).
set +e
W="$AMD_CI_WORK"; OUT="$W/out"; J="$GITHUB_WORKSPACE/k6p"; mkdir -p "$OUT"
BASE=ff5a23376725e5a889fa0bfb987b7541dc304da0
{ echo "runner ${RUNNER_NAME:-?}"; uname -a; nproc; free -m; hipconfig --version; ls -d /opt/rocm*; cmake --version | head -1; ninja --version; } > "$OUT/build_diag.txt" 2>&1
cat "$OUT/build_diag.txt"
S="$W/src_k6"; mkdir -p "$S"
git -C "$S" init -q; git -C "$S" remote add origin https://github.com/ggml-org/llama.cpp
git -C "$S" fetch -q --depth 1 origin "$BASE" || { echo "FATAL: fetch $BASE"; exit 1; }
git -C "$S" checkout -q --detach "$BASE"
[ "$(git -C "$S" rev-parse HEAD)" = "$BASE" ] || { echo "FATAL: checkout"; exit 1; }
gunzip -c "$J/k6.patch.gz" > "$W/k6.patch"
git -C "$S" apply --whitespace=nowarn "$W/k6.patch" || { echo "FATAL: k6 patch does not apply"; exit 1; }
git -C "$S" diff --shortstat | tee "$OUT/source.txt"
echo "k6 = $(cat "$J/K6_SHA") as b11721 + patch" | tee -a "$OUT/source.txt"
export HIPCXX=/opt/rocm/llvm/bin/clang HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm PATH="/opt/rocm/bin:$PATH"
T0=$(date +%s)
cmake -S "$S" -B "$S/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_CXX_FLAGS="-I/opt/rocm/include" \
  -DCMAKE_HIP_COMPILER="$HIPCXX" -DCMAKE_PREFIX_PATH=/opt/rocm -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 \
  -DBUILD_SHARED_LIBS=ON -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_RPC=ON -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DGGML_NATIVE=OFF -DGGML_STATIC=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF > "$OUT/k6_configure.log" 2>&1
crc=$?
tail -30 "$OUT/k6_configure.log"
[ $crc = 0 ] || { echo "FATAL: configure rc=$crc"; exit 1; }
cmake --build "$S/build" -j "$(nproc)" > "$OUT/k6_build.log" 2>&1
brc=$?
echo "build rc=$brc in $(( $(date +%s) - T0 )) s" | tee -a "$OUT/source.txt"
if [ $brc != 0 ]; then grep -E -B2 -A6 "error:|FAILED" "$OUT/k6_build.log" | head -120; exit 1; fi
# release config above (tests off); then test-backend-ops alone
cmake -S "$S" -B "$S/build" -DLLAMA_BUILD_TESTS=ON >> "$OUT/k6_configure.log" 2>&1 && \
  cmake --build "$S/build" -j "$(nproc)" --target test-backend-ops > "$OUT/k6_build_tbo.log" 2>&1
echo "test-backend-ops build rc=$?" | tee -a "$OUT/source.txt"
grep -c -i "warning:" "$OUT/k6_build.log" | sed 's/^/build warnings: /' | tee -a "$OUT/source.txt"
mkdir -p "$W/k6bin"
cp -a "$S"/build/bin/* "$W/k6bin/"
echo "k6 $(cat "$J/K6_SHA" | cut -c1-10) hip gfx1151 $(hipconfig --version 2>/dev/null)" > "$W/k6bin/TAG"
ls -la "$W/k6bin" | head -80
du -sh "$W/k6bin"
test -x "$W/k6bin/llama-server" && test -x "$W/k6bin/test-backend-ops" && ls "$W"/k6bin/libggml-hip.so || { echo "FATAL: missing outputs"; exit 1; }
# upload-artifact drops the executable bit: ship a tarball
mkdir -p "$W/art"; tar czf "$W/art/k6bin_linux.tgz" -C "$W" k6bin && ls -la "$W/art"
