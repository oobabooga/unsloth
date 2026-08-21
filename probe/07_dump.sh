#!/usr/bin/env bash
set -uxo pipefail
SCRATCH="${RUNNER_TEMP:-$HOME}/strix-repro"
echo "===== studio.log ====="
tail -c 300000 "$SCRATCH/studio.log" 2>/dev/null
echo "===== install.log tail ====="
tail -100 "$SCRATCH/install.log" 2>/dev/null
echo "===== server logs ====="
for f in "$SCRATCH"/studio-home/logs/server/*.log; do
  echo "----- $f -----"; tail -200 "$f"
done 2>/dev/null
echo "===== dmesg tail ====="
dmesg 2>/dev/null | tail -60
exit 0
