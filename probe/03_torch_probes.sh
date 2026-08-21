#!/usr/bin/env bash
set -uxo pipefail
SCRATCH="${RUNNER_TEMP:-$HOME}/strix-repro"
PY="$SCRATCH/venv/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
for f in "$HERE"/py/p_*.py; do
  name="$(basename "$f")"
  echo "===================== PROBE $name ====================="
  timeout 300 "$PY" "$f"
  echo "--------------------- PROBE $name exit=$? ---------------------"
done
echo "ALL TORCH PROBES DONE"
exit 0
