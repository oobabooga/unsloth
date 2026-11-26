#!/usr/bin/env bash
# Studio device checks for Linux (DGX Spark and other NVIDIA boxes). Uses the Studio that is already installed.
#   bash run.sh --mode quick
#   bash run.sh --mode full [--yes] [--studio-home /path/to/studio]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDIO_HOME_ARG=""
prev=""
for a in "$@"; do
    if [ "$prev" = "--studio-home" ]; then STUDIO_HOME_ARG="$a"; fi
    case "$a" in --studio-home=*) STUDIO_HOME_ARG="${a#--studio-home=}" ;; esac
    prev="$a"
done
# Same order as install.sh: UNSLOTH_STUDIO_HOME > STUDIO_HOME > UNSLOTH_HOME/studio > $HOME/.unsloth/studio
HOMES=()
if [ -n "$STUDIO_HOME_ARG" ]; then
    HOMES+=("$STUDIO_HOME_ARG")
else
    [ -n "${UNSLOTH_STUDIO_HOME:-}" ] && HOMES+=("$UNSLOTH_STUDIO_HOME")
    [ -n "${STUDIO_HOME:-}" ] && HOMES+=("$STUDIO_HOME")
    [ -n "${UNSLOTH_HOME:-}" ] && HOMES+=("$UNSLOTH_HOME/studio")
    HOMES+=("$HOME/.unsloth/studio")
fi
PY=""
for h in "${HOMES[@]}"; do
    for v in unsloth_studio .venv; do
        if [ -x "$h/$v/bin/python" ]; then PY="$h/$v/bin/python"; break 2; fi
    done
done
EXTRA=()
if [ -n "$PY" ]; then
    echo "Studio Python: $PY"
    EXTRA=(--studio-python "$PY")
else
    PY="$(command -v python3 || command -v python || true)"
    if [ -z "$PY" ]; then
        echo "No Unsloth Studio install found (searched: ${HOMES[*]}) and no Python on PATH."
        echo "Pass --studio-home <folder that holds unsloth_studio>."
        exit 2
    fi
    echo "No Studio venv found under: ${HOMES[*]}. The driver will report what it searched."
fi
exec "$PY" -u "$HERE/devchecks.py" "$@" ${EXTRA[@]+"${EXTRA[@]}"}
