#!/usr/bin/env bash
set -uxo pipefail
SCRATCH="${RUNNER_TEMP:-$HOME}/strix-repro"
export UNSLOTH_STUDIO_HOME="$SCRATCH/studio-home"
BIN="$UNSLOTH_STUDIO_HOME/unsloth_studio/bin/unsloth"
test -x "$BIN" || { echo "NO STUDIO BINARY -- skipping"; exit 0; }
nohup "$BIN" studio --api-only -H 127.0.0.1 -p 8888 > "$SCRATCH/studio.log" 2>&1 &
SRV=$!
echo "backend pid=$SRV"
for i in $(seq 1 150); do
  curl -sf http://127.0.0.1:8888/api/liveness >/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "BACKEND DIED DURING STARTUP"; break; }
  sleep 4
done
curl -s http://127.0.0.1:8888/api/liveness; echo
SECRET="$(cat "$UNSLOTH_STUDIO_HOME/auth/.desktop_secret")"
TOKEN="$(curl -s http://127.0.0.1:8888/api/auth/desktop-login \
  -H 'Content-Type: application/json' -d "{\"secret\": \"$SECRET\"}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')"
hit () {
  printf -- '--- GET %-40s ' "$1"
  curl -s -o /dev/null -w 'status=%{http_code} time=%{time_total}\n' \
    "http://127.0.0.1:8888$1" -H "Authorization: Bearer $TOKEN"
}
for round in 1 2 3; do
  echo "########## dropdown round $round ##########"
  hit /api/models/list
  hit /api/inference/status
  hit /api/models/scan-folders
  hit /api/models/recommended-folders
  hit /api/hub/local
  hit /api/hub/cached-gguf
  hit /api/hub/cached-models
  hit /api/inference/monitor
  sleep 15
  if kill -0 $SRV 2>/dev/null; then echo "backend ALIVE after round $round"; else echo "BACKEND DEAD after round $round"; break; fi
done
sleep 30
if kill -0 $SRV 2>/dev/null; then
  echo "BACKEND STILL ALIVE at end"
else
  wait $SRV; echo "BACKEND EXIT=$?"
fi
exit 0
