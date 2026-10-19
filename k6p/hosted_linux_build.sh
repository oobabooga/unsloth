#!/bin/bash
# hosted_linux_build.sh NAME: the release's Linux ROCm recipe (unsloth-prebuilt-rocm.yml linux leg) on a GitHub-hosted ubuntu-22.04:
# TheRock dist tarball (ROCM_VER, the weekly pin of the b11696 release), clang from /opt/rocm/llvm, release cmake flags, the ROCm
# runtime libs + rocblas / hipblaslt kernels copied beside the binaries, RPATH $ORIGIN. NAME: k6 (b11721 + k6.patch.gz), ub (plain
# upstream b11721), rs (the release's own source tarball). Output: $RUNNER_TEMP/art/bin_<NAME>.tgz (dir bin_<NAME>/).
set +e
N=$1; J="$GITHUB_WORKSPACE/k6p"; T="$RUNNER_TEMP"; ROCM_VER=${ROCM_VER:-10.2.0a20261010}
BASE=ff5a23376725e5a889fa0bfb987b7541dc304da0; TAG=b11696-mix-7baeb9a
sudo apt-get update -qq >/dev/null; sudo apt-get install -y -qq cmake ninja-build curl patchelf libatomic1 >/dev/null 2>&1
nproc; free -m | head -2; df -h "$T" | tail -1
t0=$(date +%s)
sudo mkdir -p /opt/rocm
curl -sfL "https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-$ROCM_VER.tar.gz" | sudo tar --use-compress-program=gzip -xf - -C /opt/rocm --strip-components=1
ls /opt/rocm/llvm/bin/clang || { echo "FATAL: TheRock $ROCM_VER"; exit 1; }
echo "TheRock $ROCM_VER in $(( $(date +%s) - t0 )) s"
export HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm HIP_PLATFORM=amd HIP_CLANG_PATH=/opt/rocm/llvm/bin HIP_INCLUDE_PATH=/opt/rocm/include \
  HIP_LIB_PATH=/opt/rocm/lib HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
S="$T/src"; mkdir -p "$S"
if [ "$N" = rs ]; then
  curl -fsSL "https://github.com/unslothai/llama.cpp/releases/download/$TAG/llama.cpp-source-$TAG.tar.gz" | tar xz -C "$S" --strip-components=1 || exit 1
else
  git -C "$S" init -q; git -C "$S" remote add origin https://github.com/ggml-org/llama.cpp
  git -C "$S" fetch -q --depth 1 origin $BASE && git -C "$S" checkout -q --detach $BASE || { echo "FATAL: fetch"; exit 1; }
  if [ "$N" = k6 ]; then gunzip -c "$J/k6.patch.gz" > "$T/k6.patch"; git -C "$S" apply --whitespace=nowarn "$T/k6.patch" || { echo "FATAL: patch"; exit 1; }; fi
fi
t0=$(date +%s)
cmake -S "$S" -B "$S/build" -G Ninja -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_FLAGS="-I/opt/rocm/include" -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1151 -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_RPC=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=OFF -DGGML_STATIC=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_CCACHE=OFF > "$T/cfg_$N.log" 2>&1 || { tail -40 "$T/cfg_$N.log"; exit 1; }
cmake --build "$S/build" -j "$(nproc)" -- -k 0 > "$T/build_$N.log" 2>&1
brc=$?
echo "build $N rc=$brc in $(( $(date +%s) - t0 )) s"
grep -E "error:|FAILED" "$T/build_$N.log" | head -40
[ $brc = 0 ] || exit 1
if [ "$N" = k6 ] || [ "$N" = rs ]; then
  cmake -S "$S" -B "$S/build" -DLLAMA_BUILD_TESTS=ON >> "$T/cfg_$N.log" 2>&1 && cmake --build "$S/build" -j "$(nproc)" --target test-backend-ops >> "$T/build_$N.log" 2>&1
  echo "test-backend-ops build rc=$?"
fi
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
( cd "$B" && for f in *.so* llama-* test-backend-ops; do [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null; done )
case $N in
  k6) echo "k6 $(cut -c1-10 "$J/K6_SHA") hip gfx1151 TheRock $ROCM_VER (hosted ubuntu-22.04)" > "$B/TAG" ;;
  ub) echo "upstream b11721 ff5a2337 hip gfx1151 TheRock $ROCM_VER" > "$B/TAG" ;;
  rs) echo "$TAG source hip gfx1151 TheRock $ROCM_VER" > "$B/TAG" ;;
esac
( cd "$B" && ldd ./llama-server ./libggml-hip.so 2>/dev/null | grep "not found" ) && echo "WARNING: unresolved libs"
mkdir -p "$T/art" "$T/bin_$N"; cp -a "$B"/. "$T/bin_$N/"
tar czf "$T/art/bin_$N.tgz" -C "$T" "bin_$N"; ls -la "$T/art"; du -sh "$T/bin_$N"
cp "$T/build_$N.log" "$T/cfg_$N.log" "$T/art/" 2>/dev/null
test -x "$T/bin_$N/llama-server" && ls "$T/bin_$N/libggml-hip.so"
