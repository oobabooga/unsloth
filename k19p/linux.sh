#!/bin/bash
# linux.sh: gfx1151 Linux ROCm. Trees: i3 = the public CI copy of k19-int3 1f09fc4a8fb (fetched by commit), d6 = i3 + d6.patch
# (the 11-06 tree dc86a8f271b), rp = i3 + rp.patch (the k19-gdnreplay trial merge 2b1738df6c3). Each tree hash is checked.
# Build: TheRock dist ROCM_VER, its clang, the release cmake flags with tests ON, runtime libs + rocBLAS / hipBLASLt kernels beside the
# binaries (kpack relocation), RPATH $ORIGIN. Then run.py. Writes only under $AMD_CI_WORK.
set +e
W="$AMD_CI_WORK"; J="$GITHUB_WORKSPACE/k19p"; O="$W/out"; I="$O/build"; ROCM_VER=${ROCM_VER:-10.2.0a20261017}
mkdir -p "$I"; RR="$W/rocm"; NJ=${K19_BUILD_J:-32}
{ echo "runner ${RUNNER_NAME:-?}"; uname -a; nproc; free -m | head -2; grep -m1 "model name" /proc/cpuinfo
  R0=$(command -v rocminfo || echo /opt/rocm/bin/rocminfo); "$R0" 2>/dev/null | grep -m3 -E "Marketing Name|gfx1"; df -h "$W" | tail -1; } > "$O/diag.txt" 2>&1
cat "$O/diag.txt"
if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null || ! command -v patchelf >/dev/null; then
  python3 -m venv "$W/bvenv" && "$W/bvenv/bin/pip" -q install cmake ninja patchelf > "$I/pip.log" 2>&1; export PATH="$W/bvenv/bin:$PATH"
fi
cmake --version | head -1 | tee -a "$O/diag.txt"
mkdir -p "$RR"; t0=$(date +%s)
curl -sfL "https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-$ROCM_VER.tar.gz" | tar --use-compress-program=gzip -xf - -C "$RR" --strip-components=1
CL="$RR/llvm/bin"; [ -x "$CL/clang" ] || CL="$RR/lib/llvm/bin"
[ -x "$CL/clang" ] || { echo "FATAL: TheRock $ROCM_VER"; exit 1; }
echo "TheRock $ROCM_VER in $(( $(date +%s) - t0 )) s" | tee -a "$O/diag.txt"
export HIP_PATH="$RR" ROCM_PATH="$RR" HIP_PLATFORM=amd HIP_CLANG_PATH="$CL" HIP_INCLUDE_PATH="$RR/include" HIP_LIB_PATH="$RR/lib" \
  HIP_DEVICE_LIB_PATH="$RR/lib/llvm/amdgcn/bitcode" PATH="$RR/bin:$CL:$PATH"
curl -fsSL "https://raw.githubusercontent.com/unslothai/llama.cpp/f95b52162ed4b916a58d4f16b72662483cc55075/scripts/unsloth/relocate_kpack.py" -o "$W/relocate_kpack.py"

bundle() {   # $1 build bin dir: runtime libs + kernels beside the binaries, RPATH $ORIGIN (in place)
  local B=$1
  mkdir -p "$B/rocblas" "$B/hipblaslt"
  [ -d "$RR/lib/rocblas/library" ] && cp -r "$RR/lib/rocblas/library" "$B/rocblas/"
  [ -d "$RR/lib/hipblaslt/library" ] && cp -r "$RR/lib/hipblaslt/library" "$B/hipblaslt/"
  for p in libhipblas.so* librocblas.so* libamdhip64.so* librocsolver.so* libroctx64.so* libhipblaslt.so* rocm_sysdeps/lib/librocm_sysdeps_liblzma.so* \
           librocprofiler-register.so* libamd_comgr.so* libamd_comgr_loader.so* libhsa-runtime64.so* rocm_sysdeps/lib/librocm_sysdeps_numa.so* \
           librocroller.so* liborigami.so* libtensilelite-host.so* librocm_kpack.so* rocm_sysdeps/lib/librocm_sysdeps_z.so* \
           rocm_sysdeps/lib/librocm_sysdeps_zstd.so* llvm/lib/libLLVM.so* llvm/lib/libclang-cpp.so* llvm/lib/libomp.so \
           rocm_sysdeps/lib/librocm_sysdeps_elf.so* rocm_sysdeps/lib/librocm_sysdeps_drm.so* rocm_sysdeps/lib/librocm_sysdeps_drm_amdgpu.so* \
           rocm_sysdeps/lib/librocm_sysdeps_bz2.so*; do cp -L "$RR"/lib/$p "$B/" 2>/dev/null; done
  [ -e "$B/libomp.so" ] || { lo=$(find "$RR" -name 'libomp.so' 2>/dev/null | head -1); [ -n "$lo" ] && cp -L "$lo" "$B/libomp.so"; }
  ( cd "$B" && for f in *.so* llama-* test-*; do [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null; done )
  python3 "$W/relocate_kpack.py" --bundle "$B" --kpack-src "$RR/.kpack" --archs gfx1151 >> "$I/kpack.log" 2>&1 || echo "WARNING: kpack $B"
  la="$(ldconfig -p | awk -F'=> ' '/libatomic\.so\.1/{print $2; exit}')"; [ -n "$la" ] && cp -L "$la" "$B/"
}

I3C=$(awk '$1=="i3"{print $2}' "$J/TREES")
TARGS=()
for N in ${K19_TREES:-fx d6 rpf}; do
  S="$W/src_$N"; mkdir -p "$S"; t0=$(date +%s); EXP=$(awk -v n=$N '$1==n{print $3}' "$J/TREES")
  git -C "$S" init -q; git -C "$S" config core.autocrlf false; git -C "$S" remote add origin https://github.com/danielhanchen/llama.cpp-staging-ci
  git -C "$S" fetch -q --depth 1 origin "$I3C" && git -C "$S" checkout -q --detach "$I3C" || { echo "FATAL: fetch $N" | tee "$I/SRC_CHECK_$N"; continue; }
  PS=$(awk -v n=$N '$1==n{print $5}' "$J/TREES" | tr ',' ' '); [ -z "$PS" ] && [ "$N" != i3 ] && PS=$N
  for p in $PS; do gunzip -c "$J/$p.patch.gz" > "$W/$p.patch"; git -C "$S" apply --index --whitespace=nowarn "$W/$p.patch" || echo "FATAL: $p.patch ($N)"; done
  TREE=$(git -C "$S" write-tree)
  echo "tree $N $TREE expect $EXP" | tee "$I/SRC_CHECK_$N"
  [ "$TREE" = "$EXP" ] || { echo "FATAL: the applied tree $N is not the expected one" | tee -a "$I/SRC_CHECK_$N"; continue; }
  cmake -S "$S" -B "$S/build" -G Ninja -DCMAKE_C_COMPILER="$CL/clang" -DCMAKE_CXX_COMPILER="$CL/clang++" \
    -DCMAKE_CXX_FLAGS="-I$RR/include" -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1151 -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_RPC=ON \
    -DGGML_HIP_ROCWMMA_FATTN=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=OFF -DGGML_STATIC=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_CCACHE=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DCMAKE_PREFIX_PATH="$RR" > "$I/cfg_$N.log" 2>&1 || { tail -30 "$I/cfg_$N.log"; echo "FATAL: configure $N" | tee -a "$I/SRC_CHECK_$N"; continue; }
  cmake --build "$S/build" -j "$NJ" -- -k 0 > "$I/build_$N.log" 2>&1
  brc=$?; echo "build $N rc=$brc in $(( $(date +%s) - t0 )) s" | tee -a "$I/SRC_CHECK_$N"
  grep -E " error:|FAILED:" "$I/build_$N.log" | sort | uniq -c | sort -rn | head -30 | tee -a "$I/SRC_CHECK_$N"
  echo "warnings $N: $(grep -c ' warning:' "$I/build_$N.log")" | tee -a "$I/SRC_CHECK_$N"
  [ $brc = 0 ] || continue
  bundle "$S/build/bin"
  c=$(cd "$S/build/bin" && ./llama-server --list-devices 2>&1 | grep -c "ROCm0"); echo "devices $N: ROCm0 x$c" | tee -a "$I/SRC_CHECK_$N"
  TARGS+=(--tree "$N=$S/build|$S/build/bin")
done
gunzip -c "$W/src_fx/ci/tiny-qwen3moe-f16.gguf.gz" > "$W/tiny-qwen3moe-f16.gguf"
python3 -u "$J/run.py" --out "$O" "${TARGS[@]}" --ctest fx,d6,rpf --ctest-replay fx --tbo fx,d6 --replay fx,rpf --smoke fx,d6 \
  --models "$W/src_fx/build/tests/test-models" --tinymoe "$W/tiny-qwen3moe-f16.gguf" 2>&1 | tee "$O/run.txt"
test -s "$O/rows.jsonl"
