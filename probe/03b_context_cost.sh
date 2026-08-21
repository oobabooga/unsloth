#!/usr/bin/env bash
set -uxo pipefail
SCRATCH="${RUNNER_TEMP:-$HOME}/strix-repro"
PY="${RUNNER_TEMP:-$HOME}/strix-repro/studio-home/unsloth_studio/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "===================== HIP context cost on gfx1151 ====================="
timeout 300 "$PY" "$HERE/py/context_cost.py"
echo "--------------------- exit=$? ---------------------"
exit 0
