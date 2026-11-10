#!/usr/bin/env bash
# Run after the fork's CUDA setup action configured src/build. Builds one real ggml-cuda object,
# changes one line that only the architectures after the first one compile (same line count),
# rebuilds, restores, rebuilds. nvcc -E preprocesses for the first listed architecture only.
set -euo pipefail
cd src
obj="$(ninja -C build -t targets all | grep -E 'ggml-cuda\.dir/(.*/)?mmvq\.cu\.(o|obj):' | cut -d: -f1)"
[ -n "$obj" ] || { echo "no mmvq object target"; exit 1; }
ccache --version | head -1
ccache -p | grep -E 'depend_mode|compiler_check' | tr '\n' ' '; echo
f=ggml/src/ggml-cuda/mmvq.cu
guard="$(grep -n '^#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE$' "$f" | head -1 | cut -d: -f1)"
[ -n "$guard" ] || { echo "guard not found"; exit 1; }
line="$(awk -v g="$guard" 'NR > g && /^    return MMVQ_PARAMETERS_GENERIC;$/ { print NR; exit }' "$f")"
[ -n "$line" ] || { echo "target line not found"; exit 1; }
echo "RESULT edit: $f line $line (the #else of the guard at line $guard; compiled for 80, 86 and 89, not for 70)"
sed -n "${guard},${line}p" "$f"
stats() { ccache --print-stats | grep -E '^(direct_cache_hit|preprocessed_cache_hit|cache_miss)\s' | tr '\n' ' ' | tr '\t' '='; }
h() { sha256sum "build/$obj" | cut -c1-16; }
build() { rm -f "build/$obj"; ninja -C build "$obj" >/dev/null; }
ccache -z >/dev/null; build; h1="$(h)"; s1="$(stats)"
sleep 2; sed -i "${line}s|GENERIC|TURING|" "$f"; sed -n "${line}p" "$f"
ccache -z >/dev/null; build; h2="$(h)"; s2="$(stats)"
CCACHE_DISABLE=1 build; hc="$(h)"
sleep 2; sed -i "${line}s|TURING|GENERIC|" "$f"
ccache -z >/dev/null; build; h3="$(h)"; s3="$(stats)"
echo "RESULT build1 (original)            $h1 [$s1]"
echo "RESULT build2 (edited)              $h2 [$s2]"
echo "RESULT control (edited, no ccache)  $hc"
echo "RESULT build3 (restored)            $h3 [$s3]"
if [ "$hc" = "$h1" ]; then echo "RESULT VERDICT: INCONCLUSIVE (the edit does not change the object)"
elif [ "$h2" = "$h1" ]; then echo "RESULT VERDICT: STALE (ccache returned the pre-edit object for the edited source)"
else echo "RESULT VERDICT: rebuilt"; fi
