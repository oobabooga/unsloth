#!/usr/bin/env bash
# Reproduces build-linux-rocm from unslothai/stable-diffusion.cpp#14 (head d2ae8f8) on this box.
set -euo pipefail
W="$RUNNER_TEMP/sd14-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"; rm -rf "$W"; mkdir -p "$W"/{tmp,cache}; export TMPDIR="$W/tmp" PIP_CACHE_DIR="$W/cache/pip"
cd "$W"
PR=d2ae8f84e61753ebd8cc0bf5582d4217452f4dd9
git clone -q https://github.com/unslothai/stable-diffusion.cpp src
git -C src fetch -q origin "refs/pull/14/head"; git -C src checkout -q "$PR"
git -C src submodule update -q --init --depth 1 ggml
cp -r src tooling
export ROCM_VERSION=7.14.0
export GPU_TARGETS="gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201"
python3 -m venv .rocm; . .rocm/bin/activate
python -m pip install -q --upgrade pip
# hosted runner gets these from apt; no sudo here
python -m pip install -q cmake ninja patchelf
DEVICE="$(echo "$GPU_TARGETS" | sed 's/;/,device-/g; s/^/device-/')"
python -m pip install -q --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,devel,${DEVICE}]==${ROCM_VERSION}"
export ROCM_PATH="$(rocm-sdk path --root)" HIP_PATH="$(rocm-sdk path --root)" CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}" PATH="$(rocm-sdk path --bin):$PATH"
t0=$SECONDS
( cd src
  cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang" \
    -DCMAKE_HIP_FLAGS="-mllvm --amdgpu-unroll-threshold-local=600" \
    -DSD_BUILD_EXAMPLES=ON -DSD_SERVER_BUILD_FRONTEND=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF \
    -DGGML_NATIVE=OFF -DSD_HIPBLAS=ON -DHIP_PLATFORM=amd -DGPU_TARGETS="$GPU_TARGETS" > "$W/cmake.log"
  cmake --build build --config Release -j "$(nproc)" --target sd-cli sd-server > "$W/build.log" 2>&1 || { tail -50 "$W/build.log"; exit 1; } )
echo "build took $((SECONDS - t0))s"
export GITHUB_WORKSPACE="$W"
# ---- verbatim from the PR's "Stage the bundle with the ROCm runtime" step ----
BIN="${GITHUB_WORKSPACE}/src/build/bin"
STAGE="${GITHUB_WORKSPACE}/rocm-bundle"
rm -rf "$STAGE" && mkdir -p "$STAGE/lib" "$STAGE/.kpack"
cp "$BIN/sd-cli" "$BIN/sd-server" "$STAGE/"
ldd "$STAGE/sd-cli" "$STAGE/sd-server" \
  | awk -v r="$ROCM_PATH/" 'index($3, r) == 1 { print $3 }' | sort -u > "$RUNNER_TEMP/rocm-libs.txt"
if [ ! -s "$RUNNER_TEMP/rocm-libs.txt" ]; then
  echo "ERROR: nothing resolved from $ROCM_PATH; is this a HIP build?" >&2
  exit 1
fi
while read -r so; do cp -L "$so" "$STAGE/lib/"; done < "$RUNNER_TEMP/rocm-libs.txt"
for t in rocblas hipblaslt; do
  compgen -G "$STAGE/lib/lib$t.so*" > /dev/null || continue
  mkdir -p "$STAGE/lib/$t"
  cp -rL "$ROCM_PATH/lib/$t/library" "$STAGE/lib/$t/"
done
for so in "$STAGE"/lib/*.so*; do
  objcopy -O binary --only-section=.rocm_kpack_ref "$so" "$RUNNER_TEMP/kpack_ref.bin"
  for name in $(strings "$RUNNER_TEMP/kpack_ref.bin" | grep -o '\.kpack/[A-Za-z0-9_]*_@GFXARCH@' \
      | sed 's#^\.kpack/##; s#_@GFXARCH@$##' | sort -u); do
    for gfx in ${GPU_TARGETS//;/ }; do
      cp -L "$ROCM_PATH/.kpack/${name}_${gfx}.kpack" "$STAGE/.kpack/"
    done
  done
done
patchelf --set-rpath '$ORIGIN/lib' "$STAGE/sd-cli" "$STAGE/sd-server"
for f in "$STAGE"/lib/*.so*; do patchelf --set-rpath '$ORIGIN' "$f"; done
echo "bundled $(wc -l < "$RUNNER_TEMP/rocm-libs.txt") ROCm libraries, $(ls "$STAGE/.kpack" | wc -l) kpack archives"
du -sh "$STAGE"
if LD_LIBRARY_PATH= ldd "$STAGE/sd-cli" "$STAGE/sd-server" | grep -i 'not found'; then
  echo "ERROR: unresolved libraries after bundling" >&2
  exit 1
fi
# ---- end verbatim ----
deactivate
env -i PATH="$PATH" BIN_DIR="$STAGE" KEEP_LAYOUT=1 OUT_DIR="$W/dist" TAG=pr14 LABEL=Linux-Ubuntu-24.04-x86_64-rocm-7.14.0 \
  COMMIT="$PR" SOURCE_REPO=unslothai/stable-diffusion.cpp LICENSE_FILE="$W/src/LICENSE" \
  python3 "$W/tooling/scripts/unsloth/package_bundle.py"
ls -la "$W/dist"
