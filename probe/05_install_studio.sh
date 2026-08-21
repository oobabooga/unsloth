#!/usr/bin/env bash
set -uxo pipefail
SCRATCH="${RUNNER_TEMP:-$HOME}/strix-repro"
mkdir -p "$SCRATCH"
mkdir -p "$SCRATCH"/{tmp,cache/pip,cache/uv,cache/hf,cache/torch}
export TMPDIR="$SCRATCH/tmp"
export PIP_CACHE_DIR="$SCRATCH/cache/pip"
export UV_CACHE_DIR="$SCRATCH/cache/uv"
export HF_HOME="$SCRATCH/cache/hf"
export TORCH_HOME="$SCRATCH/cache/torch"
if [ ! -d "$SCRATCH/repo" ]; then git clone --depth 1 https://github.com/unslothai/unsloth.git "$SCRATCH/repo"; fi
cd "$SCRATCH/repo"
export UNSLOTH_STUDIO_HOME="$SCRATCH/studio-home"
export UNSLOTH_SKIP_AUTOSTART=1
export NO_COLOR=1
timeout 5400 sh ./install.sh --local > "$SCRATCH/install.log" 2>&1
rc=$?
echo "installer exit=$rc"
tail -250 "$SCRATCH/install.log"
ls -la "$UNSLOTH_STUDIO_HOME" || true
ls -la "$UNSLOTH_STUDIO_HOME/unsloth_studio/bin" || true
exit 0
