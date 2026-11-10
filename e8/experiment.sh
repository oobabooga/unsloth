#!/usr/bin/env bash
# Usage: experiment.sh <plain|depend>
# A CMake + Ninja CUDA object, rebuilt after an edit that only touches code under a __CUDA_ARCH__ guard
# and keeps the line count. Does ccache hand back the pre-edit object?
set -euo pipefail
mode="$1"
mkdir -p proj && cd proj
cat > CMakeLists.txt <<'CM'
cmake_minimum_required(VERSION 3.18)
project(t LANGUAGES CXX CUDA)
add_library(t OBJECT t.cu)
CM
write() { cat > common.cuh <<SRC
#pragma once
#include <cuda_runtime.h>
static __device__ __forceinline__ int f(int a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return a + $1;
#else
    return a + 7;
#endif
}
SRC
}
cat > t.cu <<'SRC'
#include "common.cuh"
__global__ void k(int *p) { p[0] = f(p[0]); }
SRC
ccache --version | head -1
ccache --set-config=compiler_check=content
ccache --set-config=sloppiness=include_file_mtime,include_file_ctime,pch_defines,locale
[ "$mode" = depend ] && ccache --set-config=depend_mode=true
ccache --show-config 2>/dev/null | grep -E 'depend_mode|direct_mode' || ccache -p | grep -E 'depend_mode|direct_mode'
write 1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > configure.log 2>&1 || { tail -40 configure.log; exit 1; }
obj="$(ninja -C build -t targets all | grep -E 't\.cu\.(o|obj):' | cut -d: -f1)"
echo "compile command: $(ninja -C build -t commands "$obj" | tail -1 | cut -c1-600)"
stats() { ccache --print-stats 2>/dev/null | grep -E '^(direct_cache_hit|preprocessed_cache_hit|cache_miss)\s' | tr '\n' ' ' | tr '\t' '=' || ccache -s | grep -E 'hit|miss' | tr '\n' ' '; }
h() { sha256sum "build/$obj" | cut -c1-16; }
ccache -z >/dev/null; ninja -C build "$obj" >/dev/null; h1="$(h)"; s1="$(stats)"
sleep 2; write 2
ccache -z >/dev/null; ninja -C build "$obj" >/dev/null; h2="$(h)"; s2="$(stats)"
sleep 2; write 1
ccache -z >/dev/null; ninja -C build "$obj" >/dev/null; h3="$(h)"; s3="$(stats)"
echo "RESULT mode=$mode build1 $h1 [$s1]"
echo "RESULT mode=$mode build2 $h2 [$s2]"
echo "RESULT mode=$mode build3 (source restored) $h3 [$s3] $([ "$h3" = "$h1" ] && echo same-as-build1 || echo DIFFERENT-from-build1)"
if [ "$h1" = "$h2" ]; then echo "RESULT mode=$mode VERDICT: STALE (ccache returned the pre-edit object)"; else echo "RESULT mode=$mode VERDICT: rebuilt"; fi
