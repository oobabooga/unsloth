#!/usr/bin/env bash
# Usage: experiment2.sh <source-dir> <build-dir> [objects-regex]
# Does ccache serve a pre-edit object after an edit that only touches per-GPU-arch code and keeps every line number?
set -euo pipefail
SRC="$1"; BUILD="$2"; PAT="${3:-mmvq|sum}"
OBJS="$(ninja -C "$BUILD" -t targets all | grep -E "ggml-hip\.dir/(.*/)?($PAT)\.cu\.(o|obj):" | cut -d: -f1 | sort || true)"
[ -n "$OBJS" ] || { echo "no object targets matched"; exit 1; }
echo "object targets: $(echo $OBJS)"
build() { ninja -C "$BUILD" $OBJS >/dev/null; }
wipe()  { (cd "$BUILD" && rm -f $OBJS); }
hashes() { (cd "$BUILD" && sha256sum $OBJS | awk '{print substr($1,1,16)}' | tr '\n' ' '); }
stats() { ccache --print-stats | grep -E '^(direct_cache_hit|preprocessed_cache_hit|cache_miss)\s' | tr '\n' ' ' | tr '\t' '='; }
ccache --show-config | grep -E 'depend_mode|compiler_check|sloppiness' | tr '\n' ' '; echo
PYBIN="$(command -v python3 || command -v python)"
edit() {
"$PYBIN" - "$SRC/ggml/src/ggml-cuda/common.cuh" "$1" <<'PY'
import sys
p, suffix = sys.argv[1], sys.argv[2]
t = open(p, encoding="utf-8", newline="").read()
a = "    c = __builtin_amdgcn_sdot4(a, b, c, false);"
b = "    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);"
assert t.count(a) == 1 and t.count(b) == 1, (t.count(a), t.count(b))
n = t.count("\n")
t = t.replace(a, a[:-1] + suffix + ";").replace(b, b[:-1] + suffix + ";")
assert t.count("\n") == n
open(p, "w", encoding="utf-8", newline="").write(t)
PY
}
echo "== build 0: original source, ccache disabled"
wipe; CCACHE_DISABLE=1 build; H0="$(hashes)"
echo "== build 1: original source, cold ccache"
ccache -z >/dev/null; wipe; build; H1="$(hashes)"; S1="$(stats)"
echo "== edit: per-arch branches of ggml_cuda_dp4a (common.cuh), same line count"
edit " + 12345"
echo "== build 2: edited source, warm ccache"
ccache -z >/dev/null; wipe; build; H2="$(hashes)"; S2="$(stats)"
echo "== build 3: edited source, ccache disabled"
wipe; CCACHE_DISABLE=1 build; H3="$(hashes)"
{
  echo "H0 original, no ccache : $H0"
  echo "H1 original, ccache    : $H1"
  echo "H2 edited,   ccache    : $H2"
  echo "H3 edited,   no ccache : $H3"
  echo "stats build 1: $S1"
  echo "stats build 2: $S2"
  i=0
  for o in $OBJS; do
    i=$((i+1))
    h0=$(echo $H0 | cut -d' ' -f$i); h1=$(echo $H1 | cut -d' ' -f$i); h2=$(echo $H2 | cut -d' ' -f$i); h3=$(echo $H3 | cut -d' ' -f$i)
    changed=$([ "$h0" = "$h3" ] && echo "edit does not change this object" || echo "edit changes this object")
    served=$([ "$h2" = "$h1" ] && echo "ccache served the PRE-EDIT object" || echo "ccache produced a new object")
    verdict=ok; [ "$h0" != "$h3" ] && [ "$h2" = "$h1" ] && verdict=STALE
    echo "RESULT $(basename $o): $changed; $served; verdict=$verdict"
  done
} | tee -a result2.txt
[ -z "${GITHUB_STEP_SUMMARY:-}" ] || { echo '```'; cat result2.txt; echo '```'; } >> "$GITHUB_STEP_SUMMARY"
