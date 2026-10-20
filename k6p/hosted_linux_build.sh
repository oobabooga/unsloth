#!/bin/bash
# hosted_linux_build.sh NAME: the release's Linux ROCm recipe (unsloth-prebuilt-rocm.yml linux leg) on a GitHub-hosted ubuntu-22.04:
# TheRock dist tarball (ROCM_VER, the weekly pin of the b11696 release), clang from /opt/rocm/llvm, release cmake flags, the ROCm
# runtime libs + rocblas / hipblaslt kernels copied beside the binaries, RPATH $ORIGIN. NAME: k9 / k8 (b11721 + <NAME>.patch.gz,
# gfx1151 bundle), k9x (k9 compile check for gfx1100 with the gfx110X-all toolchain, no bundle). Every tree then builds all tests
# (LLAMA_BUILD_TESTS=ON) and runs `ctest -L main` on the hosted CPU (no GPU). Output: $RUNNER_TEMP/art/bin_<NAME>.tgz (dir bin_<NAME>/).
set +e
N=$1; J="$GITHUB_WORKSPACE/k6p"; T="$RUNNER_TEMP"; ROCM_VER=${ROCM_VER:-10.2.0a20261017}
BASE=ff5a23376725e5a889fa0bfb987b7541dc304da0
GFX=gfx1151; FAM=gfx1151; P=$N
if [ "$N" = k9x ]; then GFX=gfx1100; FAM=gfx110X-all; P=k9; fi
sudo apt-get update -qq >/dev/null; sudo apt-get install -y -qq cmake ninja-build curl patchelf libatomic1 >/dev/null 2>&1
nproc; free -m | head -2; df -h "$T" | tail -1
t0=$(date +%s)
sudo mkdir -p /opt/rocm
curl -sfL "https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-$FAM-$ROCM_VER.tar.gz" | sudo tar --use-compress-program=gzip -xf - -C /opt/rocm --strip-components=1
ls /opt/rocm/llvm/bin/clang || { echo "FATAL: TheRock $ROCM_VER"; exit 1; }
echo "TheRock $ROCM_VER in $(( $(date +%s) - t0 )) s"
export HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm HIP_PLATFORM=amd HIP_CLANG_PATH=/opt/rocm/llvm/bin HIP_INCLUDE_PATH=/opt/rocm/include \
  HIP_LIB_PATH=/opt/rocm/lib HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
S="$T/src"; mkdir -p "$S"
git -C "$S" init -q; git -C "$S" remote add origin https://github.com/ggml-org/llama.cpp
git -C "$S" fetch -q --depth 1 origin $BASE && git -C "$S" checkout -q --detach $BASE || { echo "FATAL: fetch"; exit 1; }
gunzip -c "$J/$P.patch.gz" > "$T/$P.patch"; git -C "$S" apply --whitespace=nowarn "$T/$P.patch" || { echo "FATAL: patch"; exit 1; }
git -C "$S" diff --shortstat
t0=$(date +%s)
cmake -S "$S" -B "$S/build" -G Ninja -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_FLAGS="-I/opt/rocm/include" -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=$GFX -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_RPC=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=OFF -DGGML_STATIC=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_CCACHE=OFF > "$T/cfg_$N.log" 2>&1 || { tail -40 "$T/cfg_$N.log"; exit 1; }
cmake --build "$S/build" -j "$(nproc)" -- -k 0 > "$T/build_$N.log" 2>&1
brc=$?
echo "build $N rc=$brc in $(( $(date +%s) - t0 )) s"
grep -E "error:|FAILED|warning:" "$T/build_$N.log" | sort | uniq -c | sort -rn | head -40
mkdir -p "$T/art"; cp "$T/build_$N.log" "$T/cfg_$N.log" "$T/art/" 2>/dev/null
[ $brc = 0 ] || exit 1
t0=$(date +%s)
cmake -S "$S" -B "$S/build" -DLLAMA_BUILD_TESTS=ON >> "$T/cfg_$N.log" 2>&1 && cmake --build "$S/build" -j "$(nproc)" -- -k 0 > "$T/build_tests_$N.log" 2>&1
trc=$?; echo "tests build $N rc=$trc in $(( $(date +%s) - t0 )) s"; grep -E "error:|FAILED" "$T/build_tests_$N.log" | head -20
( cd "$S/build" && LD_LIBRARY_PATH=/opt/rocm/lib:$S/build/bin HIP_VISIBLE_DEVICES= timeout 2400 ctest -L main --output-on-failure --timeout 900 ) > "$T/ctest_$N.log" 2>&1
echo "ctest $N rc=$?"; grep -E "tests passed|tests failed|\*\*\*Failed|Failed|Not Run|Timeout|^ *[0-9]+/[0-9]+ Test" "$T/ctest_$N.log" | tail -80
cp "$T/build_tests_$N.log" "$T/ctest_$N.log" "$T/art/"
if [ "$N" = k9x ]; then echo "compile check only"; ls -la "$S/build/bin/libggml-hip.so" && exit 0; exit 1; fi
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
la="$(ldconfig -p | awk -F'=> ' '/libatomic\.so\.1/{print $2; exit}')"; [ -n "$la" ] && cp -L "$la" "$B/"
( cd "$B" && for f in *.so* llama-* test-*; do [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null; done )
K=$(echo $N | tr a-z A-Z); echo "$N $(cut -c1-10 "$J/${K}_SHA") hip gfx1151 TheRock $ROCM_VER (hosted ubuntu-22.04)" > "$B/TAG"
( cd "$B" && ldd ./llama-server ./libggml-hip.so 2>/dev/null | grep "not found" ) && echo "WARNING: unresolved libs"
mkdir -p "$T/art" "$T/bin_$N"; cp -a "$B"/. "$T/bin_$N/"
tar czf "$T/art/bin_$N.tgz" -C "$T" "bin_$N"; ls -la "$T/art"; du -sh "$T/bin_$N"
test -x "$T/bin_$N/llama-server" && ls "$T/bin_$N/libggml-hip.so"
