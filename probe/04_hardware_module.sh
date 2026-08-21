#!/usr/bin/env bash
set -uxo pipefail
SCRATCH="${RUNNER_TEMP:-$HOME}/strix-repro"
PY="$SCRATCH/venv/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRATCH"
rm -rf repo
git clone --depth 1 https://github.com/unslothai/unsloth.git repo
"$PY" -m pip -q install psutil structlog || true
cd "$SCRATCH/repo/studio/backend"

echo "===================== hardware module (amd-smi available) ====================="
timeout 600 "$PY" "$HERE/py/hw_module.py"
echo "--------------------- exit=$? ---------------------"

echo "===================== hardware module (amd-smi hidden = Windows shape) ====================="
mkdir -p "$SCRATCH/nosmi"
timeout 600 env PATH="$SCRATCH/nosmi:/usr/bin:/bin" "$PY" "$HERE/py/hw_module.py"
echo "--------------------- exit=$? ---------------------"
exit 0
