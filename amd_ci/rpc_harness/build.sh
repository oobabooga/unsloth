#!/bin/bash
# build.sh OUT BACKEND ARCHS NAME=REF [NAME=REF ...]
#   BACKEND: cuda | hip | cpu      ARCHS: CUDA archs ("75") or AMDGPU targets ("gfx1151"), ignored for cpu
#   REF: a branch, pull/N/head, or a commit sha of unslothai/llama.cpp
# Binaries land in OUT/wt_NAME/build/bin. One ccache across trees: only files a PR changes recompile.
set -uo pipefail
OUT=$1; BACKEND=$2; ARCHS=$3; shift 3
SRC=$OUT/src
mkdir -p "$OUT"
[ -d "$SRC/.git" ] || git clone -q --filter=blob:none https://github.com/unslothai/llama.cpp "$SRC" || exit 1
export CCACHE_DIR=$OUT/ccache CCACHE_BASEDIR=$OUT CCACHE_NOHASHDIR=true CCACHE_MAXSIZE=30G CCACHE_SLOPPINESS=time_macros
JOBS=${JOBS:-$(nproc)}
LAUNCH=""
command -v ccache >/dev/null 2>&1 && LAUNCH="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache"
COMMON="-DCMAKE_BUILD_TYPE=Release -DGGML_RPC=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF"
case "$BACKEND" in
  cuda) BK="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCHS" ;;
  hip)
    ROCM=${ROCM_PATH:-/opt/rocm}
    export HIPCXX="$ROCM/llvm/bin/clang" HIP_PATH="$ROCM" ROCM_PATH="$ROCM"
    BK="-DGGML_HIP=ON -DGPU_TARGETS=$ARCHS -DAMDGPU_TARGETS=$ARCHS" ;;
  cpu) BK="" ;;
  *) echo "unknown backend $BACKEND"; exit 2 ;;
esac
rc_all=0
for spec in "$@"; do
  name=${spec%%=*}; ref=${spec#*=}
  wt=$OUT/wt_$name
  t0=$(date +%s)
  if [ ! -x "$wt/build/bin/llama-server" ]; then
    git -C "$SRC" fetch -q origin "$ref" || { echo "BUILD_FAIL $name fetch $ref"; rc_all=1; continue; }
    sha=$(git -C "$SRC" rev-parse FETCH_HEAD)
    [ -d "$wt" ] || git -C "$SRC" worktree add -q -f --detach "$wt" "$sha"
    FLAGS="$COMMON"
    # NAME_norpc: the same tree without the RPC backend (what a source build had before Studio turned RPC on).
    case "$name" in *_norpc) FLAGS="${COMMON/-DGGML_RPC=ON/-DGGML_RPC=OFF}" ;; esac
    # shellcheck disable=SC2086
    cmake -S "$wt" -B "$wt/build" $FLAGS $BK $LAUNCH > "$OUT/cmake_$name.log" 2>&1 \
      && cmake --build "$wt/build" -j "$JOBS" --target llama-server llama-bench $( [[ "$name" == *_norpc ]] || echo ggml-rpc-server ) > "$OUT/build_$name.log" 2>&1
    rc=$?
  else
    rc=0; sha=$(git -C "$wt" rev-parse HEAD)
  fi
  if [ $rc -eq 0 ] && [ -x "$wt/build/bin/llama-server" ]; then
    echo "BUILD_OK $name $sha $(( $(date +%s) - t0 ))s $wt/build/bin"
  else
    echo "BUILD_FAIL $name rc=$rc"; tail -30 "$OUT/build_$name.log" "$OUT/cmake_$name.log" 2>/dev/null; rc_all=1
  fi
done
command -v ccache >/dev/null 2>&1 && ccache -s | grep -iE "hit|miss" | head -4
exit $rc_all
