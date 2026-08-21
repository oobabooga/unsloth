#!/usr/bin/env bash
set -euxo pipefail
SCRATCH="${RUNNER_TEMP:-$HOME}/strix-repro"
mkdir -p "$SCRATCH"
mkdir -p "$SCRATCH"/{tmp,cache/pip,cache/uv,cache/hf,cache/torch}
export TMPDIR="$SCRATCH/tmp"
export PIP_CACHE_DIR="$SCRATCH/cache/pip"
export UV_CACHE_DIR="$SCRATCH/cache/uv"
export HF_HOME="$SCRATCH/cache/hf"
export TORCH_HOME="$SCRATCH/cache/torch"
mkdir -p "$SCRATCH"
python3 -m venv "$SCRATCH/venv"
"$SCRATCH/venv/bin/python" -m pip -q install --upgrade pip
"$SCRATCH/venv/bin/python" -m pip install \
  --index-url https://repo.amd.com/rocm/whl/gfx1151/ \
  --extra-index-url https://pypi.org/simple \
  'torch>=2.11.0,<2.12.0'
"$SCRATCH/venv/bin/python" -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip)"
