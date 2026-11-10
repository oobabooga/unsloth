#!/usr/bin/env bash
# Run after the fork's CUDA setup action configured src/build. Builds one real ggml-cuda object,
# changes one line under a __CUDA_ARCH__ guard (same line count), rebuilds, restores, rebuilds.
set -euo pipefail
cd src
obj="$(ninja -C build -t targets all | grep -E 'ggml-cuda\.dir/(.*/)?mmvq\.cu\.(o|obj):' | cut -d: -f1)"
[ -n "$obj" ] || { echo "no mmvq object target"; exit 1; }
ccache --version | head -1
ccache -p | grep -E 'depend_mode|compiler_check' | tr '\n' ' '; echo
f=ggml/src/ggml-cuda/common.cuh
grep -c 'return __dp4a(a, b, c);' "$f"
stats() { ccache --print-stats | grep -E '^(direct_cache_hit|preprocessed_cache_hit|cache_miss)\s' | tr '\n' ' ' | tr '\t' '='; }
h() { sha256sum "build/$obj" | cut -c1-16; }
ccache -z >/dev/null; ninja -C build "$obj" >/dev/null; h1="$(h)"; s1="$(stats)"
sleep 2; sed -i 's|    return __dp4a(a, b, c);|    return __dp4a(a, b, c) + 12345;|' "$f"; grep -c '+ 12345;' "$f"
ccache -z >/dev/null; ninja -C build "$obj" >/dev/null; h2="$(h)"; s2="$(stats)"
sleep 2; sed -i 's|    return __dp4a(a, b, c) + 12345;|    return __dp4a(a, b, c);|' "$f"
ccache -z >/dev/null; ninja -C build "$obj" >/dev/null; h3="$(h)"; s3="$(stats)"
echo "RESULT build1 (original) $h1 [$s1]"
echo "RESULT build2 (edited)   $h2 [$s2]"
echo "RESULT build3 (restored) $h3 [$s3]"
if [ "$h1" = "$h2" ]; then echo "RESULT VERDICT: STALE (ccache returned the pre-edit object for the edited source)"; else echo "RESULT VERDICT: rebuilt"; fi
