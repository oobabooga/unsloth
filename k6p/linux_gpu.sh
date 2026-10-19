#!/bin/bash
# linux_gpu.sh: gfx1151 Linux measurement. Arms rel / rel2 = release b11696-mix-7baeb9a linux-x64-rocm-gfx1151 bundle (A/A),
# k6 / k6b = this run's k6 build (A/A). Cells, KLD and test-backend-ops via k6p_bench.py. Rows -> $AMD_CI_WORK/out/k6p/out/rows.jsonl
set +e
W="$AMD_CI_WORK"; OUT="$W/out"; J="$GITHUB_WORKSPACE/k6p"; R="$W/k6p"; mkdir -p "$OUT" "$R/bins/rel" "$R/bins/k6" "$R/models"
TAG=b11696-mix-7baeb9a
{ echo "runner ${RUNNER_NAME:-?}"; uname -a; nproc; free -m; cat /proc/meminfo | head -5; ls /sys/class/drm/; for f in /sys/class/drm/card*/device/mem_info_*_total; do echo "$f $(cat $f)"; done; hipconfig --version; rocminfo 2>/dev/null | grep -m3 -E "Marketing Name|gfx1"; } > "$OUT/gpu_diag.txt" 2>&1
cat "$OUT/gpu_diag.txt"
export K6P_DIR="$R" K6P_DEV=ROCm0 ARMS="rel:rel,rel2:rel,k6:k6,k6b:k6" HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm
( python3 -u "$J/k6p_bench.py" dl --models q35,lfm2,gptoss > "$OUT/dl.log" 2>&1 ) &
DLP=$!
curl -fL --retry 5 -s -S "https://github.com/unslothai/llama.cpp/releases/download/$TAG/app-$TAG-linux-x64-rocm-gfx1151.tar.gz" | tar xz -C "$R/bins/rel"
echo "release $TAG rc=$?" | tee -a "$OUT/gpu_diag.txt"; echo "$TAG linux-x64-rocm-gfx1151" > "$R/bins/rel/TAG"
tar xzf "$W/art/k6bin_linux.tgz" -C "$W" && cp -a "$W"/k6bin/* "$R/bins/k6/"
ls "$R/bins/rel" | head -50; ls "$R/bins/k6" | head -20
wait $DLP; cat "$OUT/dl.log" | grep -v '^ROW'
B="python3 -u $J/k6p_bench.py"
$B devinfo > "$OUT/devinfo.log" 2>&1; grep -v '^ROW' "$OUT/devinfo.log"
for c in q35:full lfm2:full lfm2:spill1536 gptoss:full gptoss:spill6144; do
  $B time --cell $c --reps 3 > "$OUT/time_${c//:/_}.log" 2>&1; echo "time $c rc=$?"; grep -v '^ROW' "$OUT/time_${c//:/_}.log" | tail -14
done
KLD_ARMS="rel,rel2,k6"
$B kld --model q35 --modes full --arms $KLD_ARMS > "$OUT/kld_q35.log" 2>&1; grep KLD "$OUT/kld_q35.log"
$B kld --model lfm2 --modes full,spill1536 --arms $KLD_ARMS > "$OUT/kld_lfm2.log" 2>&1; grep KLD "$OUT/kld_lfm2.log"
$B kld --model gptoss --modes full,spill6144 --arms $KLD_ARMS > "$OUT/kld_gptoss.log" 2>&1; grep KLD "$OUT/kld_gptoss.log"
OPS=RMS_NORM,RMS_NORM_MUL_ADD,ADD_RMS_NORM,RMS_NORM_MUL_ROPE,MUL_MAT_VEC_SHARED_SRC1,MUL_MAT_ID_FALLBACK_SHARED_SRC1,MUL_MAT_ID_SHARED,MUL_MAT_VEC_FUSION,MUL_MAT_ID_FUSION,HC_POST_NORM,EW_CHAIN,TOPK_MOE,TOP_K,GET_ROWS,SET_ROWS,CPY,MUL_MAT_ID,MUL_MAT
$B tbo --bin k6 --ops $OPS --timeout 2400 > "$OUT/tbo.log" 2>&1; grep TBO "$OUT/tbo.log"
cp "$R/out/rows.jsonl" "$OUT/rows_linux.jsonl"; mkdir -p "$OUT/logs"; cp "$R"/out/logs/*.filtered.log "$OUT/logs/" 2>/dev/null
test -s "$OUT/rows_linux.jsonl"
