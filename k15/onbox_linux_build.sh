#!/bin/bash
# onbox_linux_build.sh: k19 r3. The hosted ubuntu runners of the fork were saturated (r1 / r2 hosted builds queued 30+ min behind other
# sessions' matrices), so the Linux bins are built on the gfx1151 box itself, inside the GPU job, with the same recipe as
# hosted_linux_build.sh: TheRock dist tarball ROCM_VER (extracted under $AMD_CI_WORK, no root), clang from it, release cmake flags,
# tests ON, the runtime libs + rocblas / hipblaslt kernels (kpack step) beside the binaries, RPATH $ORIGIN; and PolyStrata (its Strix
# Halo recipe, no source change). Trees as the hosted script (k10 / pfd / i3d patches on ff5a2337, tree hashes checked).
# Output: $AMD_CI_WORK/k15/bins/<name>/ (i3, up, pf, poly); logs in $AMD_CI_WORK/out/build/. No ctest here (the box has the GPU).
set +e
W="$AMD_CI_WORK"; J="$GITHUB_WORKSPACE/k15"; I="$W/out/build"; R="$W/k15"; ROCM_VER=${ROCM_VER:-10.2.0a20261017}
mkdir -p "$I" "$R/bins"; RR="$W/rocm"
NJ=${K19_BUILD_J:-32}
t0=$(date +%s)
for t in cmake ninja git patchelf c++ g++ gcc; do echo "$t: $(command -v $t)"; done | tee "$I/tools.txt"
if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null; then
  python3 -m venv "$W/bvenv" && "$W/bvenv/bin/pip" -q install cmake ninja patchelf > "$I/pip.log" 2>&1; export PATH="$W/bvenv/bin:$PATH"
fi
command -v patchelf >/dev/null || { python3 -m venv "$W/bvenv" 2>/dev/null; "$W/bvenv/bin/pip" -q install patchelf >> "$I/pip.log" 2>&1; export PATH="$W/bvenv/bin:$PATH"; }
cmake --version | head -1 | tee -a "$I/tools.txt"; ninja --version | tee -a "$I/tools.txt"
mkdir -p "$RR"
curl -sfL "https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-$ROCM_VER.tar.gz" | tar --use-compress-program=gzip -xf - -C "$RR" --strip-components=1
ls "$RR/llvm/bin/clang" > /dev/null || ls "$RR/lib/llvm/bin/clang" > /dev/null || { echo "FATAL: TheRock $ROCM_VER"; exit 1; }
CL="$RR/llvm/bin"; [ -x "$CL/clang" ] || CL="$RR/lib/llvm/bin"
echo "TheRock $ROCM_VER in $(( $(date +%s) - t0 )) s, clang $CL" | tee -a "$I/tools.txt"
export HIP_PATH="$RR" ROCM_PATH="$RR" HIP_PLATFORM=amd HIP_CLANG_PATH="$CL" HIP_INCLUDE_PATH="$RR/include" HIP_LIB_PATH="$RR/lib" \
  HIP_DEVICE_LIB_PATH="$RR/lib/llvm/amdgcn/bitcode" PATH="$RR/bin:$CL:$PATH"
BASE=$(cat "$J/UP_SHA")

bundle() {   # $1 build bin dir, $2 output dir
  local B=$1 D=$2
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
  ( cd "$B" && for f in *.so* llama-* test-* strata-*; do [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null; done )
  python3 "$W/relocate_kpack.py" --bundle "$B" --kpack-src "$RR/.kpack" --archs gfx1151 >> "$I/kpack.log" 2>&1 || echo "WARNING: kpack $B"
  la="$(ldconfig -p | awk -F'=> ' '/libatomic\.so\.1/{print $2; exit}')"; [ -n "$la" ] && cp -L "$la" "$B/"
  mkdir -p "$D"; cp -a "$B"/. "$D/"
}
curl -fsSL "https://raw.githubusercontent.com/unslothai/llama.cpp/f95b52162ed4b916a58d4f16b72662483cc55075/scripts/unsloth/relocate_kpack.py" -o "$W/relocate_kpack.py"

for N in ${K19_TREES:-i3 up pf}; do
  case $N in up) PS=""; WANT=$(cat "$J/UP_SHA");; pf) PS="k10 pfd"; WANT=$(cat "$J/PF_SHA");; i3) PS="k10 pfd i3d"; WANT=$(cat "$J/I3_SHA");; esac
  S="$W/src_$N"; mkdir -p "$S"; t0=$(date +%s)
  git -C "$S" init -q; git -C "$S" config core.autocrlf false; git -C "$S" remote add origin https://github.com/ggml-org/llama.cpp
  git -C "$S" fetch -q --depth 1 origin "$BASE" && git -C "$S" checkout -q --detach "$BASE" || { echo "FATAL: fetch $N"; continue; }
  for p in $PS; do gunzip -c "$J/$p.patch.gz" > "$W/$p.patch"; git -C "$S" apply --index --whitespace=nowarn "$W/$p.patch" || echo "FATAL: $p.patch ($N)"; done
  TREE=$(git -C "$S" write-tree); EXP=$(grep "^$WANT " "$J/TREES" | cut -d' ' -f2)
  echo "tree $N $TREE expect $EXP (${WANT:0:11})" | tee "$I/SRC_CHECK_$N"
  [ -n "$EXP" ] && [ "$TREE" = "$EXP" ] || { echo "FATAL: the applied tree is not ${WANT:0:11}" | tee -a "$I/SRC_CHECK_$N"; continue; }
  cmake -S "$S" -B "$S/build" -G Ninja -DCMAKE_C_COMPILER="$CL/clang" -DCMAKE_CXX_COMPILER="$CL/clang++" \
    -DCMAKE_CXX_FLAGS="-I$RR/include" -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1151 -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_RPC=ON \
    -DGGML_HIP_ROCWMMA_FATTN=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=OFF -DGGML_STATIC=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_CCACHE=OFF -DCMAKE_PREFIX_PATH="$RR" > "$I/cfg_$N.log" 2>&1 || { tail -30 "$I/cfg_$N.log"; echo "FATAL: configure $N" | tee -a "$I/SRC_CHECK_$N"; continue; }
  grep -E "^GGML_HIP_GRAPHS|^GGML_HIP:|^GPU_TARGETS" "$S/build/CMakeCache.txt" | tee -a "$I/SRC_CHECK_$N"
  cmake --build "$S/build" -j "$NJ" -- -k 0 > "$I/build_$N.log" 2>&1
  brc=$?; echo "build $N rc=$brc in $(( $(date +%s) - t0 )) s" | tee -a "$I/SRC_CHECK_$N"
  grep -E " error:|FAILED" "$I/build_$N.log" | sort | uniq -c | sort -rn | head -20 | tee -a "$I/SRC_CHECK_$N"
  [ $brc = 0 ] || continue
  K=$(echo $N | tr a-z A-Z); echo "$N $(cut -c1-11 "$J/${K}_SHA") hip gfx1151 TheRock $ROCM_VER (on-box)" > "$S/build/bin/TAG"
  bundle "$S/build/bin" "$R/bins/$N"
  for t in llama-server test-backend-ops test-llama-archs test-llama-context; do echo "$N $t: $([ -x "$R/bins/$N/$t" ] && echo present || echo absent)" | tee -a "$I/SRC_CHECK_$N"; done
done

if [ -z "${K19_NO_POLY:-}" ]; then
  S="$W/poly"; t0=$(date +%s); SHA=$(cat "$J/POLY_SHA")
  git init -q "$S"; git -C "$S" remote add origin https://github.com/VecSzn/PolyStrata
  if git -C "$S" fetch -q --depth 1 origin "$SHA" && git -C "$S" checkout -q --detach FETCH_HEAD; then
    echo "poly $(git -C "$S" rev-parse HEAD)" | tee "$I/SRC_CHECK_poly"
    cmake -S "$S" -B "$S/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON \
      -DSTRATA_BUILD_TESTS=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_HIP_COMPILER="$CL/clang++" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$RR" \
      "-DCMAKE_PREFIX_PATH=$RR;$RR/lib/rocm_sysdeps;$RR/lib/llvm" \
      "-DCMAKE_HIP_FLAGS=--rocm-path=$RR --rocm-device-lib-path=$RR/lib/llvm/amdgcn/bitcode" > "$I/cfg_poly.log" 2>&1
    crc=$?; echo "configure poly rc=$crc" | tee -a "$I/SRC_CHECK_poly"; grep -i -E "warning.*gfx|hipblaslt|ggml commit" "$I/cfg_poly.log" | head -8 | tee -a "$I/SRC_CHECK_poly"
    if [ $crc = 0 ]; then
      cmake --build "$S/build" --target strata-poly -j "$NJ" > "$I/build_poly.log" 2>&1
      brc=$?; echo "build poly rc=$brc in $(( $(date +%s) - t0 )) s" | tee -a "$I/SRC_CHECK_poly"; grep -E " error:|FAILED" "$I/build_poly.log" | head -20 | tee -a "$I/SRC_CHECK_poly"
      if [ $brc = 0 ]; then
        B="$W/bin_poly"; mkdir -p "$B/tools"; cp "$(find "$S/build" -maxdepth 3 -type f -name strata-poly | head -1)" "$B/"
        cp -r "$S/tools/hip" "$B/tools/" 2>/dev/null; find "$S/build" -maxdepth 3 -name "*.so*" -exec cp -L {} "$B/" \; 2>/dev/null
        echo "poly ${SHA:0:8} hip gfx1151 TheRock $ROCM_VER (on-box)" > "$B/TAG"
        bundle "$B" "$R/bins/poly"
      fi
    fi
  else
    echo "FATAL: fetch PolyStrata" | tee "$I/SRC_CHECK_poly"
  fi
fi
ls "$R/bins"
