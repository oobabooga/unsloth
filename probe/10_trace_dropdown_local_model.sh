#!/usr/bin/env bash
set -uxo pipefail
SCRATCH="${RUNNER_TEMP:-$HOME}/strix-repro"
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/auth.sh"
export UNSLOTH_STUDIO_HOME="$SCRATCH/studio-home"
export HF_HOME="$SCRATCH/cache/hf"
PY="$UNSLOTH_STUDIO_HOME/unsloth_studio/bin/python"
BIN="$UNSLOTH_STUDIO_HOME/unsloth_studio/bin/unsloth"
test -x "$PY" || { echo "NO STUDIO VENV -- skipping"; exit 0; }

MODELS="$SCRATCH/local-models"
timeout 1800 "$PY" "$HERE/py/fetch_small_gguf.py" "$MODELS" || echo "GGUF DOWNLOAD FAILED (continuing)"
find "$MODELS" -name '*.gguf' -printf '%p %s\n' 2>/dev/null | head

# Trace both native entry points, in whichever copy of studio the backend imports.
BACKEND_DIRS="$SCRATCH/repo/studio/backend"
VENV_BACKEND="$("$PY" -c 'import studio, os; print(os.path.join(os.path.dirname(studio.__file__), "backend"))' 2>/dev/null)"
[ -n "$VENV_BACKEND" ] && [ -d "$VENV_BACKEND" ] && BACKEND_DIRS="$BACKEND_DIRS $VENV_BACKEND"
MEMPY="$("$PY" -c 'import torch.cuda.memory as m; print(m.__file__)')"
for d in $BACKEND_DIRS; do
  echo "tracing $d/utils/hardware/amd.py"
  "$PY" "$HERE/py/patch_amd_smi_trace.py" "$d/utils/hardware/amd.py" "$MEMPY"
done

# Prove the trace fires before relying on its silence.
"$PY" -c "
import sys; sys.path.insert(0, '$SCRATCH/repo/studio/backend')
from utils.hardware import amd
amd._run_amd_smi('list')
" 2>&1 | grep -c '@@@AMD_SMI@@@' || echo "SELF CHECK FAILED"

LOG="$SCRATCH/traced2.log"
nohup "$BIN" studio --api-only -H 127.0.0.1 -p 8901 > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 150); do
  curl -sf http://127.0.0.1:8901/api/liveness >/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "BACKEND DIED DURING STARTUP"; break; }
  sleep 4
done
sleep 60   # let the background warm finish so its calls do not mix into the traces
studio_token 8901 || { tail -40 "$LOG"; exit 0; }

echo "@@@REQUEST register scan folder@@@" >> "$LOG"
curl -s -o /dev/null -w 'add scan folder status=%{http_code}\n' -X POST \
  "http://127.0.0.1:8901/api/models/scan-folders" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"path\": \"$MODELS\"}"
sleep 8

echo "@@@BASELINE_AFTER_WARM@@@" >> "$LOG"
hit () {
  echo "@@@REQUEST $1@@@" >> "$LOG"
  printf -- '--- %-52s ' "$1"
  curl -s -o /dev/null -w 'status=%{http_code} time=%{time_total}\n' \
    "http://127.0.0.1:8901$1" -H "Authorization: Bearer $TOKEN"
  sleep 4
}
for p in /api/models/list /api/inference/status /api/models/scan-folders \
         /api/models/recommended-folders /api/hub/local /api/hub/cached-gguf \
         /api/hub/cached-models /api/inference/monitor; do
  hit "$p"
done

if kill -0 $SRV 2>/dev/null; then echo "TRACED BACKEND ALIVE"; else wait $SRV; echo "TRACED BACKEND EXIT=$?"; fi

echo "===================== native calls attributed to a request ====================="
echo "amd-smi calls: $(grep -c '@@@AMD_SMI@@@' "$LOG")"
echo "mem_get_info calls: $(grep -c '@@@MEM_GET_INFO@@@' "$LOG")"
"$PY" "$HERE/py/summarise_traces.py" "$LOG"
kill $SRV 2>/dev/null
exit 0
