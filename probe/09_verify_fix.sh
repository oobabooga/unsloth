#!/usr/bin/env bash
set -uxo pipefail
SCRATCH="${RUNNER_TEMP:-$HOME}/strix-repro"
HERE="$(cd "$(dirname "$0")" && pwd)"
export UNSLOTH_STUDIO_HOME="$SCRATCH/studio-home"
PY="$UNSLOTH_STUDIO_HOME/unsloth_studio/bin/python"
test -x "$PY" || { echo "NO STUDIO VENV -- skipping"; exit 0; }

cd "$SCRATCH/repo"
git apply --stat "$HERE/fix.patch"
git apply "$HERE/fix.patch" && echo "FIX APPLIED" || { echo "FIX DID NOT APPLY"; exit 0; }
git apply "$HERE/matrix_test.patch" || echo "matrix test patch did not apply (non-fatal)"
cp "$HERE/test_windows_apu_fit_keeps_no_hip_context.py" studio/backend/tests/

# Same probe as before the patch: on Linux nothing about this path may move.
cd "$SCRATCH/repo/studio/backend"
echo "===================== llama.cpp GPU probe AFTER the fix (Linux gfx1151) ====================="
timeout 900 "$PY" "$HERE/py/llama_gpu_probe.py"
echo "--------------------- exit=$? ---------------------"

echo "===================== unit tests in the studio venv ====================="
"$PY" -m pip -q install pytest pytest-subtests 2>&1 | tail -2
timeout 1800 "$PY" -m pytest -q \
  tests/test_windows_apu_fit_keeps_no_hip_context.py \
  tests/test_gpu_arch_gate_os_matrix_7624.py \
  tests/test_amd_apu_unified_memory.py \
  tests/test_rocm_vram_probe_no_hip_context.py \
  tests/test_gpu_arch_gate_7624.py 2>&1 | tail -20
echo "--------------------- tests exit=$? ---------------------"
exit 0
