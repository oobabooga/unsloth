#!/usr/bin/env bash
# Usage: experiment.sh <source-dir> <build-dir>
# Builds three ggml-hip objects, edits GPU-only code, rebuilds with ccache and without, compares.
set -euo pipefail
SRC="$1"; BUILD="$2"
objs() { ninja -C "$BUILD" -t targets all | grep -E 'ggml-hip\.dir/(mmvq|mmq|sum)\.cu\.(o|obj):' | cut -d: -f1 | sort; }
OBJS="$(objs)"
[ "$(echo "$OBJS" | wc -l)" -eq 3 ] || { echo "expected 3 object targets, got: $OBJS"; exit 1; }
build() { ninja -C "$BUILD" $OBJS; }
wipe()  { (cd "$BUILD" && rm -f $OBJS); }
hashes() { (cd "$BUILD" && sha256sum $OBJS | awk '{print $1}' | tr '\n' ' '); }
stats() { ccache --print-stats | grep -E '^(direct_cache_hit|preprocessed_cache_hit|cache_miss|direct_cache_miss|preprocessed_cache_miss|unsupported_source_language|local_storage_hit|local_storage_miss)\s' | tr '\n' ' ' | tr '\t' '='; }

ccache --show-config | grep -E 'depend_mode|direct_mode|compiler_check|sloppiness' || true
ccache -z >/dev/null

echo "== build 1: original source, cold cache"
build; H1="$(hashes)"; S1="$(stats)"; ccache -z >/dev/null

echo "== edit: GPU-only code in common.cuh (inside the per-arch #if of ggml_cuda_dp4a)"
f="$SRC/ggml/src/ggml-cuda/common.cuh"
PYBIN="$(command -v python3 || command -v python)"
"$PYBIN" - "$f" <<'PY'
import sys
p = sys.argv[1]
t = open(p, encoding="utf-8").read()
a = "    c = __builtin_amdgcn_sdot4(a, b, c, false);\n"
b = "    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);\n"
assert t.count(a) == 1 and t.count(b) == 1, (t.count(a), t.count(b))
t = t.replace(a, a + "    c += 12345;\n").replace(b, b + "    c += 12345;\n")
open(p, "w", encoding="utf-8", newline="").write(t)
PY
(cd "$SRC" && git diff --stat 2>/dev/null || true)

echo "== build 2: edited source, warm ccache"
wipe; build; H2="$(hashes)"; S2="$(stats)"; ccache -z >/dev/null

echo "== build 3: edited source, ccache disabled"
wipe; CCACHE_DISABLE=1 build; H3="$(hashes)"
echo "== build 4: edited source, ccache disabled (determinism control)"
wipe; CCACHE_DISABLE=1 build; H4="$(hashes)"

{
  echo "objects: $(echo $OBJS | tr '\n' ' ')"
  echo "H1 original          : $H1"
  echo "H2 edited, ccache    : $H2"
  echo "H3 edited, no ccache : $H3"
  echo "H4 edited, no ccache : $H4"
  echo "stats build 1: $S1"
  echo "stats build 2: $S2"
  i=0
  for o in $OBJS; do
    i=$((i+1))
    h1=$(echo $H1 | cut -d' ' -f$i); h2=$(echo $H2 | cut -d' ' -f$i); h3=$(echo $H3 | cut -d' ' -f$i); h4=$(echo $H4 | cut -d' ' -f$i)
    det=$([ "$h3" = "$h4" ] && echo deterministic || echo NONDETERMINISTIC)
    changed=$([ "$h1" = "$h3" ] && echo "edit does not change this object" || echo "edit changes this object")
    if [ "$h2" = "$h3" ]; then verdict="ccache result is CORRECT"; elif [ "$h2" = "$h1" ]; then verdict="ccache result is STALE (pre-edit object)"; else verdict="ccache result differs from both"; fi
    echo "RESULT $(basename $o): $det; $changed; $verdict"
  done
} | tee result.txt
[ -z "${GITHUB_STEP_SUMMARY:-}" ] || { echo '```'; cat result.txt; echo '```'; } >> "$GITHUB_STEP_SUMMARY"
