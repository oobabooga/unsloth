#!/bin/bash
# usage: serve.sh <port> ; starts Studio from $W/src on localhost, waits for health
cd "$W/src/studio/backend"
export PATH="$HOME/.local/bin:$PATH"
# The initial password is accepted once; a restart on the same home must not pass it again.
if [ -e "$W/pw.set" ]; then unset UNSLOTH_STUDIO_PASSWORD; else export UNSLOTH_STUDIO_PASSWORD="$(cat "$W/pw")"; touch "$W/pw.set"; fi
nohup "$UNSLOTH_STUDIO_HOME/unsloth_studio/bin/python" run.py --host 127.0.0.1 --port "$1" > "$W/studio-$1.log" 2>&1 &
echo $! > "$W/studio.pid"
for i in $(seq 1 300); do curl -sf "http://127.0.0.1:$1/api/health" >/dev/null && { echo "up on $1"; exit 0; }; sleep 1; done
echo "Studio did not start"; tail -80 "$W/studio-$1.log"; exit 1
