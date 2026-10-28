#!/bin/bash
# hosted_linux_build.sh NAME: the release's Linux ROCm recipe (unsloth-prebuilt-rocm.yml linux leg) on a GitHub-hosted ubuntu-22.04, as in
# the k9 / k10 / k10s runs: TheRock dist tarball (ROCM_VER, the weekly pin of b11734-mix-c208e5f), clang from /opt/rocm/llvm, release
# cmake flags (GGML_HIP_GRAPHS at its default), the ROCm runtime libs + rocblas / hipblaslt kernels (kpack step) beside the binaries,
# RPATH $ORIGIN. Trees (all from upstream ggml-org ff5a2337, the merge base of both heads):
#   up  = ff5a2337 stock;  k10 = + k15/k10.patch.gz (a162dce854);  k15 = + k10.patch.gz + k15/k15d.patch.gz (95b695923af)
#   k15cpu = the k15 tree, CPU only, GGML_BACKEND_DL=OFF static (test-mmid-multi only builds without dynamic backends), two ISA
#            variants: avx512 (AVX-512 F/VNNI/VBMI/BF16, Zen 5) and avx2. Targets test-mmid-multi + test-backend-ops only.
# HIP trees build all tests (LLAMA_BUILD_TESTS=ON) and run `ctest -L main` on the hosted CPU (no GPU). The applied tree is checked
# against the SHA's tree hash (git write-tree), so a patch that applies but yields another tree fails here.
# Output: $RUNNER_TEMP/art/bin_<NAME>.tgz (dir bin_<NAME>/) + $RUNNER_TEMP/info/ (logs, SRC_CHECK) for the verdict job.
set +e
N=$1; J="$GITHUB_WORKSPACE/k15"; T="$RUNNER_TEMP"; ROCM_VER=${ROCM_VER:-10.2.0a20261017}
BASE=$(cat "$J/UP_SHA")
mkdir -p "$T/art" "$T/info"; I="$T/info"
sudo apt-get update -qq >/dev/null; sudo apt-get install -y -qq cmake ninja-build curl patchelf libatomic1 >/dev/null 2>&1
nproc; free -m | head -2; df -h "$T" | tail -1
S="$T/src"; mkdir -p "$S"
git -C "$S" init -q; git -C "$S" config core.autocrlf false; git -C "$S" remote add origin https://github.com/ggml-org/llama.cpp
git -C "$S" fetch -q --depth 1 origin "$BASE" && git -C "$S" checkout -q --detach "$BASE" || { echo "FATAL: fetch"; exit 1; }
git -C "$S" config user.email ci@localhost; git -C "$S" config user.name ci
case $N in up) PS=""; WANT=$(cat "$J/UP_SHA");; k10) PS="k10"; WANT=$(cat "$J/K10_SHA");; k15|k15cpu) PS="k10 k15d"; WANT=$(cat "$J/K15_SHA");;
  *) echo "FATAL: unknown $N"; exit 1;; esac
for p in $PS; do
  gunzip -c "$J/$p.patch.gz" > "$T/$p.patch"
  git -C "$S" apply --index --whitespace=nowarn "$T/$p.patch" || { echo "FATAL: $p.patch does not apply"; exit 1; }
done
# the expected tree of each head (git ls-tree of the SHA in wt_k15, recorded when the patches were made)
TREE=$(git -C "$S" write-tree); EXP=$(grep "^$WANT " "$J/TREES" | cut -d' ' -f2)
echo "tree $N $TREE expect $EXP (${WANT:0:11})" | tee "$I/SRC_CHECK_$N"
[ -n "$EXP" ] && [ "$TREE" = "$EXP" ] || { echo "FATAL: the applied tree is not ${WANT:0:11}"; exit 1; }
# 1. the eager-fallback error mapping in the HIP vendor header (be6d2579151): present on k15, absent on the base trees
c=$(grep -c "^#define cudaErrorInvalidValue hipErrorInvalidValue" "$S/ggml/src/ggml-cuda/vendors/hip.h"); echo "hip.h cudaErrorInvalidValue mapping lines: $c" | tee -a "$I/SRC_CHECK_$N"
git -C "$S" diff --cached --shortstat "$BASE" | tee -a "$I/SRC_CHECK_$N"

if [ "$N" = k15cpu ]; then
  COM="-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF -DGGML_OPENMP=OFF -DLLAMA_BUILD_TESTS=ON \
       -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF"
  declare -A F
  F[avx512]="-DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_BF16=ON"
  F[avx2]="-DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX512=OFF"
  mkdir -p "$T/bin_$N"; ok=1
  for v in avx512 avx2; do
    B="$S/build_$v"; t0=$(date +%s)
    cmake -S "$S" -B "$B" $COM ${F[$v]} > "$I/cfg_${N}_$v.log" 2>&1 && cmake --build "$B" -j "$(nproc)" --target test-mmid-multi test-backend-ops > "$I/build_${N}_$v.log" 2>&1
    brc=$?; echo "build $N $v rc=$brc in $(( $(date +%s) - t0 )) s" | tee -a "$I/SRC_CHECK_$N"
    [ $brc = 0 ] || { grep -E "error:|FAILED" "$I/build_${N}_$v.log" | head -20; ok=0; continue; }
    mkdir -p "$T/bin_$N/$v"; cp "$B/bin/test-mmid-multi" "$B/bin/test-backend-ops" "$T/bin_$N/$v/"
  done
  echo "$N $(cut -c1-11 "$J/K15_SHA") cpu-only static avx512 + avx2 (hosted ubuntu-22.04)" > "$T/bin_$N/TAG"
  tar czf "$T/art/bin_$N.tgz" -C "$T" "bin_$N"; ls -la "$T/art"
  [ $ok = 1 ] && test -x "$T/bin_$N/avx512/test-mmid-multi" && test -x "$T/bin_$N/avx2/test-mmid-multi"; exit $?
fi

t0=$(date +%s)
sudo mkdir -p /opt/rocm
curl -sfL "https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-$ROCM_VER.tar.gz" | sudo tar --use-compress-program=gzip -xf - -C /opt/rocm --strip-components=1
ls /opt/rocm/llvm/bin/clang || { echo "FATAL: TheRock $ROCM_VER"; exit 1; }
echo "TheRock $ROCM_VER in $(( $(date +%s) - t0 )) s"
export HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm HIP_PLATFORM=amd HIP_CLANG_PATH=/opt/rocm/llvm/bin HIP_INCLUDE_PATH=/opt/rocm/include \
  HIP_LIB_PATH=/opt/rocm/lib HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
t0=$(date +%s)
cmake -S "$S" -B "$S/build" -G Ninja -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_FLAGS="-I/opt/rocm/include" -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1151 -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_RPC=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=OFF -DGGML_STATIC=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_CCACHE=OFF > "$I/cfg_$N.log" 2>&1 || { tail -40 "$I/cfg_$N.log"; exit 1; }
grep -E "^GGML_HIP_GRAPHS|^GGML_HIP:|^GPU_TARGETS" "$S/build/CMakeCache.txt" | tee -a "$I/SRC_CHECK_$N"
cmake --build "$S/build" -j "$(nproc)" -- -k 0 > "$I/build_$N.log" 2>&1
brc=$?
echo "build $N rc=$brc in $(( $(date +%s) - t0 )) s" | tee -a "$I/SRC_CHECK_$N"
grep -E "error:|FAILED|warning:" "$I/build_$N.log" | sort | uniq -c | sort -rn | head -40
[ $brc = 0 ] || exit 1
t0=$(date +%s)
cmake -S "$S" -B "$S/build" -DLLAMA_BUILD_TESTS=ON >> "$I/cfg_$N.log" 2>&1 && cmake --build "$S/build" -j "$(nproc)" -- -k 0 > "$I/build_tests_$N.log" 2>&1
trc=$?; echo "tests build $N rc=$trc in $(( $(date +%s) - t0 )) s" | tee -a "$I/SRC_CHECK_$N"; grep -E "error:|FAILED" "$I/build_tests_$N.log" | head -20
( cd "$S/build" && LD_LIBRARY_PATH=/opt/rocm/lib:$S/build/bin HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= timeout 2400 ctest -L main --output-on-failure --timeout 900 ) > "$I/ctest_$N.log" 2>&1
echo "ctest $N rc=$?" | tee -a "$I/SRC_CHECK_$N"; grep -E "tests passed|tests failed|\*\*\*Failed|Failed|Not Run|Timeout|^ *[0-9]+/[0-9]+ Test" "$I/ctest_$N.log" | tail -80
B="$S/build/bin"
mkdir -p "$B/rocblas" "$B/hipblaslt"
[ -d /opt/rocm/lib/rocblas/library ] && cp -r /opt/rocm/lib/rocblas/library "$B/rocblas/"
[ -d /opt/rocm/lib/hipblaslt/library ] && cp -r /opt/rocm/lib/hipblaslt/library "$B/hipblaslt/"
for p in libhipblas.so* librocblas.so* libamdhip64.so* librocsolver.so* libroctx64.so* libhipblaslt.so* rocm_sysdeps/lib/librocm_sysdeps_liblzma.so* \
         librocprofiler-register.so* libamd_comgr.so* libamd_comgr_loader.so* libhsa-runtime64.so* rocm_sysdeps/lib/librocm_sysdeps_numa.so* \
         librocroller.so* liborigami.so* libtensilelite-host.so* librocm_kpack.so* rocm_sysdeps/lib/librocm_sysdeps_z.so* \
         rocm_sysdeps/lib/librocm_sysdeps_zstd.so* llvm/lib/libLLVM.so* llvm/lib/libclang-cpp.so* llvm/lib/libomp.so \
         rocm_sysdeps/lib/librocm_sysdeps_elf.so* rocm_sysdeps/lib/librocm_sysdeps_drm.so* rocm_sysdeps/lib/librocm_sysdeps_drm_amdgpu.so* \
         rocm_sysdeps/lib/librocm_sysdeps_bz2.so*; do cp -L /opt/rocm/lib/$p "$B/" 2>/dev/null; done  # -L: no symlink may point outside the bundle
[ -e "$B/libomp.so" ] || { rm -f "$B/libomp.so"; lo=$(find /opt/rocm -name 'libomp.so' 2>/dev/null | head -1); echo "libomp from ${lo:-none}"; [ -n "$lo" ] && cp -L "$lo" "$B/libomp.so"; }
( cd "$B" && for f in *.so* llama-* test-*; do [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null; done )
# the release's kpack step (unslothai/unsloth#13663): TheRock keeps the rocBLAS / hipBLASLt kernels in /opt/rocm/.kpack; a flat
# bundle without them aborts in hipblasSgemm (hipErrorInvalidKernelFile). Same script and pinned commit as the k9 / k10 runs.
KP=https://raw.githubusercontent.com/unslothai/llama.cpp/f95b52162ed4b916a58d4f16b72662483cc55075/scripts/unsloth/relocate_kpack.py
curl -fsSL "$KP" -o "$T/relocate_kpack.py" && python3 "$T/relocate_kpack.py" --bundle "$B" --kpack-src /opt/rocm/.kpack --archs gfx1151 || { echo "FATAL: kpack"; exit 1; }; ls -la "$B/rocblas/k"
la="$(ldconfig -p | awk -F'=> ' '/libatomic\.so\.1/{print $2; exit}')"; [ -n "$la" ] && cp -L "$la" "$B/"
K=$(echo $N | tr a-z A-Z); echo "$N $(cut -c1-11 "$J/${K}_SHA") hip gfx1151 TheRock $ROCM_VER (hosted ubuntu-22.04)" > "$B/TAG"
( cd "$B" && ldd ./llama-server ./libggml-hip.so 2>/dev/null | grep "not found" ) && echo "WARNING: unresolved libs"
for t in test-backend-ops test-graph-src-type; do echo "$t: $([ -x "$B/$t" ] && echo present || echo absent)" | tee -a "$I/SRC_CHECK_$N"; done
mkdir -p "$T/bin_$N"; cp -a "$B"/. "$T/bin_$N/"
tar czf "$T/art/bin_$N.tgz" -C "$T" "bin_$N"; ls -la "$T/art"; du -sh "$T/bin_$N"
test -x "$T/bin_$N/llama-server" && test -x "$T/bin_$N/test-backend-ops" && ls "$T/bin_$N/libggml-hip.so"
