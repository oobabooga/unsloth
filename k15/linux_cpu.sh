#!/bin/bash
# linux_cpu.sh (k15 item 9): the k17-ttft CPU multi-column MUL_MAT_ID kernels on the gfx1151 host CPU (Zen 5, AVX-512), HIP masked.
# bins: k15cpu avx512 / avx2 (static, GGML_BACKEND_DL=OFF; the hosted build). Per variant and GGML_CPU_MMID_MULTI_K in {1, 0}:
#   test-mmid-multi with GGML_CPU_TILED_MM=0 (bitwise vs one ggml_vec_dot per (weight row, routed row); any byte difference fails)
#   test-backend-ops -b CPU -o MUL_MAT_ID for Q4_K / Q5_K / Q6_K / IQ3_XXS / MXFP4 (crash / NaN check: the reference is the same CPU)
# A host without avx512f fails the gate (the AVX-512 variant would SIGILL, which is no answer about Zen 5).
set +e
W="$AMD_CI_WORK"; OUT="$W/out"; J="$GITHUB_WORKSPACE/k15"; R="$W/k15"; mkdir -p "$OUT" "$R/bins"
{ echo "runner ${RUNNER_NAME:-?}"; uname -a; nproc; grep -m1 "model name" /proc/cpuinfo; grep -o -w -E "avx512(f|_vnni|vbmi|_bf16)" /proc/cpuinfo | sort | uniq -c
  echo "HIP_VISIBLE_DEVICES='${HIP_VISIBLE_DEVICES-unset}' ROCR_VISIBLE_DEVICES='${ROCR_VISIBLE_DEVICES-unset}'"; } > "$OUT/cpu_diag.txt" 2>&1
cat "$OUT/cpu_diag.txt"
grep -q -w avx512f /proc/cpuinfo || { echo "GATE FAILED: no avx512f on this host"; exit 1; }
tar xzf "$W/art/bin_k15cpu/bin_k15cpu.tgz" -C "$R/bins" 2>/dev/null || tar xzf "$W/art/bin_k15cpu.tgz" -C "$R/bins"
mv "$R/bins/bin_k15cpu" "$R/bins/k15cpu" || { echo "FATAL: no k15cpu bin"; exit 1; }
export K6P_DIR="$R" K6P_OUT="$OUT/bench" K6P_DEV=CPU GGML_CPU_TILED_MM=0 PYTHONIOENCODING=utf-8
export ARMS="c512_k1:k15cpu/avx512:GGML_CPU_MMID_MULTI_K=1,c512_k0:k15cpu/avx512:GGML_CPU_MMID_MULTI_K=0,c2_k1:k15cpu/avx2:GGML_CPU_MMID_MULTI_K=1,c2_k0:k15cpu/avx2:GGML_CPU_MMID_MULTI_K=0"
T="MUL_MAT_ID@type_a=(q4_K|q5_K|q6_K|iq3_xxs|mxfp4),"
for a in c512_k1 c512_k0 c2_k1 c2_k0; do
  python3 -u "$J/k15_bench.py" exe --bin $a --exe test-mmid-multi --timeout 1800 > "$OUT/mmid_$a.log" 2>&1; echo "mmid $a rc=$? $(grep -a -E 'EXE' "$OUT/mmid_$a.log" | tail -1 | cut -c1-200)"
  python3 -u "$J/k15_bench.py" tbo --bin $a --ops "$T;" --timeout 3600 > "$OUT/tbocpu_$a.log" 2>&1; echo "tbo $a rc=$? $(grep -a -E 'TBO' "$OUT/tbocpu_$a.log" | tail -1 | cut -c1-200)"
done
test -s "$OUT/bench/rows.jsonl"
