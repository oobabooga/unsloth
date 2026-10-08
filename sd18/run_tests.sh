#!/usr/bin/env bash
# usage: run_tests.sh <test-backend-ops binary> <backend device name> <patched|baseline> <logdir>
set -uo pipefail
BIN=$1
DEV=$2
VARIANT=$3
LOGS=$4
mkdir -p "$LOGS"
TO=()
if timeout --version >/dev/null 2>&1; then TO=(timeout 7200); fi
SUMMARY="$LOGS/summary.txt"
: > "$SUMMARY"

run() {
  local name=$1; shift
  local log="$LOGS/$name.log"
  local t0=$SECONDS
  echo "::group::$name: $BIN $* -b $DEV"
  "${TO[@]}" "$BIN" "$@" -b "$DEV" > "$log" 2>&1
  local rc=$?
  sed -i 's/\x1b\[[0-9;]*m//g' "$log"
  grep -E 'FAIL|not supported|SUPPORTED|tests passed|backends passed|Backend |Device description|error|Assert|assert|abort' "$log" \
    | grep -vE ': OK$' | awk '!seen[$0]++' | head -n 80
  echo "::endgroup::"
  local passed ok_n fail_n nsup_n sup_n
  passed=$(grep -E '^  [0-9]+/[0-9]+ tests passed' "$log" | tail -n 1 | sed 's/^ *//')
  ok_n=$(grep -cE ': OK$' "$log")
  fail_n=$(grep -cE ': FAIL$|FAIL$' "$log")
  nsup_n=$(grep -ciE 'not supported' "$log")
  sup_n=$(grep -cE ': SUPPORTED$' "$log")
  printf '%-22s rc=%-3s %-28s ok=%-5s fail=%-4s not_supported=%-5s supported=%-5s %ss\n' \
    "$name" "$rc" "${passed:-no-summary}" "$ok_n" "$fail_n" "$nsup_n" "$sup_n" "$((SECONDS - t0))" | tee -a "$SUMMARY"
}

if [ "$VARIANT" = patched ]; then
  run new_support support -o ROPE_PE_PERMUTE,MODULATE_ROWS,SWIGLU_SCALED
  run new_ops test -o ROPE_PE_PERMUTE,MODULATE_ROWS,SWIGLU_SCALED
fi
run epilogue test -o SD18_MM_EPILOGUE
GGML_CUDA_CUBLAS_EPILOGUE_FUSION=0 run epilogue_fusion_off test -o SD18_MM_EPILOGUE
run rms_norm test -o RMS_NORM,RMS_NORM_MUL_ADD,ADD_RMS_NORM,RMS_NORM_MUL_ROPE
run add_mul_glu test -o ADD,MUL,GLU
run mul_mat test -o MUL_MAT,MUL_MAT_VEC_FUSION -j 8
run fa_longseq test -o FLASH_ATTN_EXT -p 'kv=(260|1000|4100),nb=(64|200|1000),mask=0' -j 8
run fa_all test -o FLASH_ATTN_EXT -j 8

echo "===== SUMMARY $VARIANT $DEV ====="
cat "$SUMMARY"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  { echo "### $VARIANT $DEV"; echo '```'; cat "$SUMMARY"; echo '```'; } >> "$GITHUB_STEP_SUMMARY"
fi
