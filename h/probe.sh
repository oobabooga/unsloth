#!/usr/bin/env bash
# Runs inside a clean Ubuntu container: no CUDA libraries, /driver holds a stand-in libcuda.so.1
# (the runner has no NVIDIA driver). Usage: probe.sh <label>
set -uo pipefail
label="$1"
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libgomp1 >/dev/null 2>&1
cd /b
echo "RESULT $label CUDA runtime libs on the system: $(ldconfig -p | grep -cE 'libcudart|libcublas')"
echo "RESULT $label runtime libs in bundle: [$(ls | grep -E '^libcu(dart|blas)' | tr '\n' ' ')]"
echo "RESULT $label ldd libggml-cuda.so not found (no driver): [$(ldd ./libggml-cuda.so | grep 'not found' | awk '{print $1}' | tr '\n' ' ')]"
export LD_LIBRARY_PATH=/driver
echo "RESULT $label ldd libggml-cuda.so not found (driver stand-in): [$(ldd ./libggml-cuda.so | grep 'not found' | awk '{print $1}' | tr '\n' ' ')]"
out="$(./llama-server --list-devices 2>&1)"
echo "RESULT $label llama-server --list-devices rc=$?:"
printf '%s\n' "$out" | tail -6 | sed 's/^/    | /'
