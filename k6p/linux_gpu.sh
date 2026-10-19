#!/bin/bash
# linux_gpu.sh: gfx1151 Linux measurement (run 4). bins: rel = release b11696-mix-7baeb9a linux-x64-rocm-gfx1151 bundle; k6 / rs / ub =
# the hosted ubuntu-22.04 builds with the release's recipe and TheRock pin (artifacts bin_<name>.tgz in $AMD_CI_WORK/art). Matrix in
# k6p_plan.py; rows -> $AMD_CI_WORK/out/rows_linux.jsonl
set +e
W="$AMD_CI_WORK"; OUT="$W/out"; J="$GITHUB_WORKSPACE/k6p"; R="$W/k6p"; mkdir -p "$OUT" "$R/bins/rel" "$R/models"
TAG=b11696-mix-7baeb9a
{ echo "runner ${RUNNER_NAME:-?}"; uname -a; nproc; free -m; cat /sys/fs/cgroup/cpu.max 2>/dev/null; head -5 /proc/meminfo; for f in /sys/class/drm/card*/device/mem_info_*_total; do echo "$f $(cat $f)"; done; ls /opt/rocm* -d; rocminfo 2>/dev/null | grep -m3 -E "Marketing Name|gfx1"; } > "$OUT/gpu_diag.txt" 2>&1
cat "$OUT/gpu_diag.txt"
export K6P_DIR="$R" K6P_DEV=ROCm0 ARMS="rel:rel,rs:rs,ub:ub,k6:k6"
( python3 -u "$J/k6p_bench.py" dl --models q35,lfm2,gptoss > "$OUT/dl.log" 2>&1 ) &
DLP=$!
curl -fL --retry 5 -s -S "https://github.com/unslothai/llama.cpp/releases/download/$TAG/app-$TAG-linux-x64-rocm-gfx1151.tar.gz" | tar xz -C "$R/bins/rel"
echo "release $TAG rc=$?" | tee -a "$OUT/gpu_diag.txt"; echo "$TAG linux-x64-rocm-gfx1151" > "$R/bins/rel/TAG"
for n in k6 rs ub; do
  tar xzf "$W/art/bin_$n/bin_$n.tgz" -C "$R/bins" 2>/dev/null || tar xzf "$W/art/bin_$n.tgz" -C "$R/bins"
  mv "$R/bins/bin_$n" "$R/bins/$n"; echo "bin $n: $(cat "$R/bins/$n/TAG") $(ls "$R/bins/$n" | wc -l) files" | tee -a "$OUT/gpu_diag.txt"
done
# a bundle without libomp.so (TheRock 10.2.0a20261010 keeps it elsewhere): the release bundle's copy (LLVM OpenMP runtime)
for n in k6 rs ub; do [ -e "$R/bins/$n/libomp.so" ] || { rm -f "$R/bins/$n/libomp.so"; cp -L "$R/bins/rel/libomp.so" "$R/bins/$n/" && echo "libomp.so for $n from rel" | tee -a "$OUT/gpu_diag.txt"; }; done
for n in rel k6; do ( cd "$R/bins/$n" && LD_LIBRARY_PATH=. ldd ./libggml-hip.so | grep -E "amdhip|rocblas|not found" ) | sed "s/^/$n: /" | tee -a "$OUT/gpu_diag.txt"; done
for n in rel k6 rs ub; do d=$(cd "$R/bins/$n" && LD_LIBRARY_PATH=. ./llama-server --list-devices 2>&1 | grep -c "ROCm0"); echo "devices $n: ROCm0 x$d" | tee -a "$OUT/gpu_diag.txt"; [ "$d" -ge 1 ] || { echo "FATAL: bundle $n sees no ROCm0"; exit 1; }; done
wait $DLP; grep -v '^ROW' "$OUT/dl.log" | tail -5
OPS=RMS_NORM,RMS_NORM_MUL_ADD,ADD_RMS_NORM,RMS_NORM_MUL_ROPE,MUL_MAT_VEC_SHARED_SRC1,MUL_MAT_ID_FALLBACK_SHARED_SRC1,MUL_MAT_ID_SHARED,MUL_MAT_VEC_FUSION,MUL_MAT_ID_FUSION,HC_POST_NORM,EW_CHAIN,TOPK_MOE,TOP_K,GET_ROWS,SET_ROWS,CPY,MUL_MAT_ID,MUL_MAT
python3 -u "$J/k6p_plan.py" "$OUT/plan" "$OPS"
# MUL_MAT on the release source built with the same toolchain: tells a k6 failure from a toolchain / rocBLAS one
[ -x "$R/bins/rs/test-backend-ops" ] && python3 -u "$J/k6p_bench.py" tbo --bin rs --ops MUL_MAT,MUL_MAT_ID --timeout 2400 > "$OUT/plan/tbo_rs.log" 2>&1; grep TBO "$OUT/plan/tbo_rs.log"
cp "$R/out/rows.jsonl" "$OUT/rows_linux.jsonl"; mkdir -p "$OUT/logs"; cp "$R"/out/logs/*.filtered.log "$OUT/logs/" 2>/dev/null
test -s "$OUT/rows_linux.jsonl"
