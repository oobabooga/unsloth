#!/usr/bin/env bash
set -uxo pipefail
SCRATCH="${RUNNER_TEMP:-$HOME}/strix-repro"
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/auth.sh"
export UNSLOTH_STUDIO_HOME="$SCRATCH/studio-home"
PY="$UNSLOTH_STUDIO_HOME/unsloth_studio/bin/python"
BIN="$UNSLOTH_STUDIO_HOME/unsloth_studio/bin/unsloth"
test -x "$PY" || { echo "NO STUDIO VENV -- skipping"; exit 0; }

MEMPY="$("$PY" -c 'import torch.cuda.memory as m; print(m.__file__)')"
echo "torch memory module: $MEMPY"
"$PY" "$HERE/py/patch_torch_trace.py" "$MEMPY"

LOG="$SCRATCH/traced.log"
nohup "$BIN" studio --api-only -H 127.0.0.1 -p 8899 > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 150); do
  curl -sf http://127.0.0.1:8899/api/liveness >/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "BACKEND DIED DURING STARTUP"; break; }
  sleep 4
done
# let the background torch warm settle so its own calls do not mix into the request traces
sleep 60
studio_token 8899 || { tail -50 "$LOG"; exit 0; }
echo "@@@BASELINE_AFTER_WARM@@@" >> "$LOG"

hit () {
  echo "@@@REQUEST $1@@@" >> "$LOG"
  printf -- '--- %-52s ' "$1"
  curl -s -o /dev/null -w 'status=%{http_code} time=%{time_total}\n' \
    "http://127.0.0.1:8899$1" -H "Authorization: Bearer $TOKEN"
  sleep 4
}
for p in /api/models/list /api/inference/status /api/models/scan-folders \
         /api/models/recommended-folders /api/hub/local /api/hub/cached-gguf \
         /api/hub/cached-models /api/inference/monitor /api/health \
         "/api/models/gguf-variants?model_path=unsloth/gemma-4-E2B-it-GGUF"; do
  hit "$p"
done

echo "@@@REQUEST POST /api/inference/validate@@@" >> "$LOG"
curl -s -o /dev/null -w 'validate status=%{http_code} time=%{time_total}\n' -X POST \
  "http://127.0.0.1:8899/api/inference/validate" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"model_path": "unsloth/gemma-4-E2B-it-GGUF", "gguf_variant": "Q4_K_M"}'
sleep 6
if kill -0 $SRV 2>/dev/null; then echo "TRACED BACKEND ALIVE"; else wait $SRV; echo "TRACED BACKEND EXIT=$?"; fi

echo "===================== mem_get_info calls, attributed to a request ====================="
grep -c '@@@MEM_GET_INFO@@@' "$LOG"
"$PY" "$HERE/py/summarise_traces.py" "$LOG"
kill $SRV 2>/dev/null
exit 0
