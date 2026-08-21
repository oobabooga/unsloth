#!/usr/bin/env bash
set -uxo pipefail
SCRATCH="${RUNNER_TEMP:-$HOME}/strix-repro"
export UNSLOTH_STUDIO_HOME="$SCRATCH/studio-home"
PY="$UNSLOTH_STUDIO_HOME/unsloth_studio/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
test -x "$PY" || { echo "NO STUDIO VENV -- skipping"; exit 0; }
BACKEND="$("$PY" -c 'import studio, os; print(os.path.join(os.path.dirname(studio.__file__), "backend"))' 2>/dev/null)"
echo "backend dir = $BACKEND"
test -d "$BACKEND" || BACKEND="$SCRATCH/repo/studio/backend"
cd "$BACKEND"

echo "===================== llama.cpp GPU probe (amd-smi ON PATH) ====================="
timeout 900 "$PY" "$HERE/py/llama_gpu_probe.py"
echo "--------------------- exit=$? ---------------------"

echo "===================== llama.cpp GPU probe (amd-smi BROKEN = Windows shape) ====================="
mkdir -p "$SCRATCH/nosmi"
printf '#!/bin/sh\nexit 127\n' > "$SCRATCH/nosmi/amd-smi"
chmod +x "$SCRATCH/nosmi/amd-smi"
timeout 900 env PATH="$SCRATCH/nosmi:$PATH" "$PY" "$HERE/py/llama_gpu_probe.py"
echo "--------------------- exit=$? ---------------------"
exit 0
