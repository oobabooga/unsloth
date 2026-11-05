#!/bin/bash
# hosted_poly_build.sh: PolyStrata f472d85a (k15/POLY_SHA) for gfx1151 on a GitHub-hosted ubuntu-22.04, its own docs/STRIX_HALO.md recipe
# (STRATA_ENABLE_HIP=ON, STRATA_PREFILL_MMQ=ON, CMAKE_HIP_ARCHITECTURES=gfx1151), no source change, with the same TheRock dist tarball as
# the llama builds (ROCM_VER). ggml comes from the revision its CMake pins. Target strata-poly (the engine the scoreboard's "poly" arms
# run). ggml-cpu: GGML_NATIVE=OFF + the Zen 5 ISA (AVX-512 F/VBMI/VNNI/BF16), since -march=native would be the hosted CPU's. Runtime libs + rocblas / hipblaslt kernels beside the exe (kpack step), RPATH $ORIGIN, its tools/hip tuning tables copied along.
# Output: $RUNNER_TEMP/art/bin_poly.tgz (dir bin_poly/) + $RUNNER_TEMP/info/ (logs). A failed build uploads no bin: the poly step skips.
set +e
J="$GITHUB_WORKSPACE/k15"; T="$RUNNER_TEMP"; ROCM_VER=${ROCM_VER:-10.2.0a20261017}; SHA=$(cat "$J/POLY_SHA")
mkdir -p "$T/art" "$T/info"; I="$T/info"
sudo apt-get update -qq >/dev/null; sudo apt-get install -y -qq cmake ninja-build curl patchelf libatomic1 g++-12 >/dev/null 2>&1
nproc; free -m | head -2; df -h "$T" | tail -1; cmake --version | head -1
S="$T/poly"; git init -q "$S"; git -C "$S" remote add origin https://github.com/VecSzn/PolyStrata
git -C "$S" fetch -q --depth 1 origin "$SHA" && git -C "$S" checkout -q --detach FETCH_HEAD || { echo "FATAL: fetch PolyStrata $SHA"; exit 1; }
echo "poly $(git -C "$S" rev-parse HEAD)" | tee "$I/SRC_CHECK_poly"
t0=$(date +%s)
sudo mkdir -p /opt/rocm
curl -sfL "https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-$ROCM_VER.tar.gz" | sudo tar --use-compress-program=gzip -xf - -C /opt/rocm --strip-components=1
ls /opt/rocm/lib/llvm/bin/clang++ || { echo "FATAL: TheRock $ROCM_VER"; exit 1; }
echo "TheRock $ROCM_VER in $(( $(date +%s) - t0 )) s"
SDK=/opt/rocm
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd PATH="$SDK/bin:$SDK/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
t0=$(date +%s)
cmake -S "$S" -B "$S/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON \
  -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON -DCMAKE_HIP_COMPILER=$SDK/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode" > "$I/cfg_poly.log" 2>&1 || { tail -40 "$I/cfg_poly.log"; echo "FATAL: poly configure"; exit 1; }
grep -i -E "warning|gfx1151|hipblaslt" "$I/cfg_poly.log" | head -20
cmake --build "$S/build" --target strata-poly -j "$(nproc)" > "$I/build_poly.log" 2>&1
brc=$?; echo "build poly rc=$brc in $(( $(date +%s) - t0 )) s" | tee -a "$I/SRC_CHECK_poly"
grep -E "error:|FAILED" "$I/build_poly.log" | head -30
[ $brc = 0 ] || exit 1
B="$T/bin_poly"; mkdir -p "$B/rocblas" "$B/hipblaslt" "$B/tools"
EXE=$(find "$S/build" -maxdepth 3 -type f -name strata-poly | head -1); cp "$EXE" "$B/"
cp -r "$S/tools/hip" "$B/tools/" 2>/dev/null
find "$S/build" -maxdepth 3 -name "*.so*" -exec cp -L {} "$B/" \; 2>/dev/null
[ -d /opt/rocm/lib/rocblas/library ] && cp -r /opt/rocm/lib/rocblas/library "$B/rocblas/"
[ -d /opt/rocm/lib/hipblaslt/library ] && cp -r /opt/rocm/lib/hipblaslt/library "$B/hipblaslt/"
for p in libhipblas.so* librocblas.so* libamdhip64.so* librocsolver.so* libroctx64.so* libhipblaslt.so* rocm_sysdeps/lib/librocm_sysdeps_liblzma.so* \
         librocprofiler-register.so* libamd_comgr.so* libamd_comgr_loader.so* libhsa-runtime64.so* rocm_sysdeps/lib/librocm_sysdeps_numa.so* \
         librocroller.so* liborigami.so* libtensilelite-host.so* librocm_kpack.so* rocm_sysdeps/lib/librocm_sysdeps_z.so* \
         rocm_sysdeps/lib/librocm_sysdeps_zstd.so* llvm/lib/libLLVM.so* llvm/lib/libclang-cpp.so* llvm/lib/libomp.so \
         rocm_sysdeps/lib/librocm_sysdeps_elf.so* rocm_sysdeps/lib/librocm_sysdeps_drm.so* rocm_sysdeps/lib/librocm_sysdeps_drm_amdgpu.so* \
         rocm_sysdeps/lib/librocm_sysdeps_bz2.so*; do cp -L /opt/rocm/lib/$p "$B/" 2>/dev/null; done
( cd "$B" && for f in *.so* strata-poly; do [ -f "$f" ] && patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null; done )
KP=https://raw.githubusercontent.com/unslothai/llama.cpp/f95b52162ed4b916a58d4f16b72662483cc55075/scripts/unsloth/relocate_kpack.py
curl -fsSL "$KP" -o "$T/relocate_kpack.py" && python3 "$T/relocate_kpack.py" --bundle "$B" --kpack-src /opt/rocm/.kpack --archs gfx1151 || echo "WARNING: kpack"
la="$(ldconfig -p | awk -F'=> ' '/libatomic\.so\.1/{print $2; exit}')"; [ -n "$la" ] && cp -L "$la" "$B/"
echo "poly $(cut -c1-8 "$J/POLY_SHA") hip gfx1151 TheRock $ROCM_VER (hosted ubuntu-22.04)" > "$B/TAG"
( cd "$B" && ldd ./strata-poly | grep "not found" ) && echo "WARNING: unresolved libs"
tar czf "$T/art/bin_poly.tgz" -C "$T" "bin_poly"; ls -la "$T/art"; du -sh "$B"
test -x "$B/strata-poly"
