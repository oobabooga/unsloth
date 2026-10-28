#!/bin/bash
# linux_gpu.sh (k15): gfx1151 Linux measurement. Bins k15 / k10 / up = the hosted ubuntu-22.04 builds with the release's recipe and
# TheRock pin (artifacts bin_<name>.tgz in $AMD_CI_WORK/art). Matrix in k15_plan.py (K15_STEPS); rows -> $AMD_CI_WORK/out/rows_linux.jsonl.
# Writes only under $AMD_CI_WORK (= $RUNNER_TEMP).
set +e
export K15_DEADLINE=$(( $(date +%s) + ${K15_STEP_MIN:-345} * 60 - 300 ))
W="$AMD_CI_WORK"; OUT="$W/out"; J="$GITHUB_WORKSPACE/k15"; R="$W/k15"; mkdir -p "$OUT" "$R/bins" "$R/models"
{ echo "runner ${RUNNER_NAME:-?}"; uname -a; nproc; free -m; grep -m1 "model name" /proc/cpuinfo; grep -o -m1 -w avx512f /proc/cpuinfo; head -5 /proc/meminfo
  for f in /sys/class/drm/card*/device/mem_info_*_total; do echo "$f $(cat $f)"; done; ls /opt/rocm* -d; rocminfo 2>/dev/null | grep -m3 -E "Marketing Name|gfx1"
  df -h "$W" | tail -1; } > "$OUT/gpu_diag.txt" 2>&1
cat "$OUT/gpu_diag.txt"
export K6P_DIR="$R" K6P_DEV=ROCm0 K6P_OUT="$OUT/bench"
( python3 -u "$J/k15_bench.py" dl --models q35,q3m > "$OUT/dl.log" 2>&1 ) &
DLP=$!
for n in k15 k10 up; do
  tar xzf "$W/art/bin_$n/bin_$n.tgz" -C "$R/bins" 2>/dev/null || tar xzf "$W/art/bin_$n.tgz" -C "$R/bins"
  [ -d "$R/bins/bin_$n" ] || { echo "bin $n: missing" | tee -a "$OUT/gpu_diag.txt"; [ $n = up ] && continue; echo "FATAL: no bin $n"; exit 1; }
  mv "$R/bins/bin_$n" "$R/bins/$n"; echo "bin $n: $(cat "$R/bins/$n/TAG") $(ls "$R/bins/$n" | wc -l) files" | tee -a "$OUT/gpu_diag.txt"
done
for n in k15 k10 up; do [ -d "$R/bins/$n" ] || continue; d="$R/bins/$n"
  ( cd "$d" && LD_LIBRARY_PATH=. ldd ./libggml-hip.so | grep -E "amdhip|rocblas|libomp|not found" ) | sed "s/^/$n: /" | tee -a "$OUT/gpu_diag.txt"
  c=$(cd "$d" && LD_LIBRARY_PATH=. ./llama-server --list-devices 2>&1 | grep -c "ROCm0"); echo "devices $n: ROCm0 x$c" | tee -a "$OUT/gpu_diag.txt"
  [ "$c" -ge 1 ] || { echo "FATAL: bundle $n sees no ROCm0"; exit 1; }
done
wait $DLP; grep -v '^ROW' "$OUT/dl.log" | tail -5
python3 -u "$J/k15_plan.py" "$OUT/plan"
test -s "$OUT/bench/rows.jsonl"
