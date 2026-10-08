#!/usr/bin/env bash
# Rebuild libhsa-runtime64 from the rocm-systems pin of a TheRock nightly, unpatched
# and with the KFD wave-count patch (rocm-systems#11299), using identical flags, then
# apply the proposed prebuild step's three gates against the shipped library.
#   ROCM   extracted TheRock tarball (default /opt/rocm)
#   WORK   holds src_unpatched/, src_patched/, bundle_stock/, topo_spoof.c
# Runs on a dev host or on the prebuild workflow's ubuntu-22.04 runner (APT=1 installs tools).
set -euo pipefail
ROCM="${ROCM:-/opt/rocm}"
WORK="${WORK:?}"
if [ "${APT:-0}" = "1" ]; then
  sudo apt-get update -qq >/dev/null
  sudo apt-get install -y -qq ninja-build pkg-config xxd patchelf binutils >/dev/null
fi
# TheRock ships its own sysdeps headers + pkgconfig (libdrm, libdrm_amdgpu, numa, libelf),
# so the rebuilt runtime links the bundle's librocm_sysdeps_* exactly like the shipped one.
export PKG_CONFIG_PATH="$ROCM/lib/rocm_sysdeps/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
cmake --version | head -1

for flavor in unpatched patched; do
  bdir="$WORK/build_$flavor"
  rm -rf "${bdir:?}"
  cmake -S "$WORK/src_$flavor/projects/rocr-runtime" -B "$bdir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_PREFIX_PATH="$ROCM;$ROCM/lib/rocm_sysdeps" \
    -DCMAKE_C_COMPILER="$ROCM/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$ROCM/llvm/bin/clang++" \
    > "$WORK/cmake_$flavor.log" 2>&1 || { tail -40 "$WORK/cmake_$flavor.log"; exit 1; }
  cmake --build "$bdir" --target hsa-runtime64 > "$WORK/build_$flavor.log" 2>&1 \
    || { tail -40 "$WORK/build_$flavor.log"; exit 1; }
  echo "built $flavor: $(find "$bdir" -name 'libhsa-runtime64.so.1.*' -type f)"
done

old_lib="$WORK/bundle_stock/libhsa-runtime64.so.1"
exports() { objdump -T "$1" | awk 'NF > 2 && !/\*UND\*/ && $(NF-1) ~ /ROCR_/ {print $(NF-1), $NF}' | sort -u; }

for flavor in unpatched patched; do
  echo "=== gates: $flavor"
  new_lib="$(find "$WORK/build_$flavor" -name 'libhsa-runtime64.so.1.*' -type f | head -1)"
  out="$WORK/bundle_$flavor"
  rm -rf "${out:?}"
  mkdir -p "$out"
  # Gate 1: every ROCR_* export the shipped library has, at the same version node.
  missing="$(comm -23 <(exports "$old_lib") <(exports "$new_lib"))"
  echo "exports shipped=$(exports "$old_lib" | wc -l) rebuilt=$(exports "$new_lib" | wc -l) missing=$(printf '%s' "$missing" | grep -c . || true)"
  if [ -n "$missing" ]; then echo "$missing"; exit 1; fi
  cp "$new_lib" "$out/libhsa-runtime64.so.1"
  patchelf --set-rpath '$ORIGIN' "$out/libhsa-runtime64.so.1"
  # Gate 2: NEEDED set is satisfied by the bundle (or glibc / libstdc++).
  for dep in $(readelf -d "$out/libhsa-runtime64.so.1" | awk -F'[][]' '/NEEDED/ {print $2}'); do
    case "$dep" in libc.so.*|libm.so.*|libdl.so.*|librt.so.*|libpthread.so.*|ld-linux*|libstdc++.so.*|libgcc_s.so.*) echo "NEEDED $dep (system)"; continue ;; esac
    if [ -e "$WORK/bundle_stock/$dep" ]; then echo "NEEDED $dep (bundled)"; continue; fi
    echo "NEEDED $dep: NOT IN BUNDLE"; exit 1
  done
  # Gate 3: HIP and llama-completion resolve against the swapped runtime.
  stage="$WORK/stage_$flavor"
  rm -rf "${stage:?}"
  cp -a "$WORK/bundle_stock" "$stage"
  cp "$out/libhsa-runtime64.so.1" "$stage/libhsa-runtime64.so.1"
  unresolved="$(LD_LIBRARY_PATH="$stage" ldd -r "$stage/libamdhip64.so.7" 2>&1 | grep -E 'undefined symbol|not found' || true)"
  unresolved2="$(LD_LIBRARY_PATH="$stage" ldd -r "$stage/llama-completion" 2>&1 | grep -E 'undefined symbol|not found' || true)"
  if [ -n "$unresolved$unresolved2" ]; then echo "$unresolved"; echo "$unresolved2"; exit 1; fi
  rm -rf "${stage:?}"
  echo "ldd -r clean (libamdhip64.so.7, llama-completion)"
  echo "glibc max: $(objdump -T "$out/libhsa-runtime64.so.1" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"
  sha256sum "$out/libhsa-runtime64.so.1"
done
echo "glibc max (shipped): $(objdump -T "$old_lib" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"

# The fopen shim for the topology spoof, built for the same glibc floor.
gcc -O2 -shared -fPIC -o "$WORK/topo_spoof.so" "$WORK/topo_spoof.c" -ldl
echo "shim glibc max: $(objdump -T "$WORK/topo_spoof.so" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"
echo "=== BUILD OK"
