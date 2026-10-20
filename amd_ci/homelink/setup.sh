#!/usr/bin/env bash
# Setup for one side of the two-machine home cluster tryout, everything under $AMD_CI_WORK:
#   1. clone unslothai/unsloth at $HL_REF (the #14274 branch by default)
#   2. a throwaway venv with what the cluster CLI and Studio's planner import (probe loop installs the rest)
#   3. llama.cpp CPU + RPC from unslothai/llama.cpp pull/187/head: build-base, then the same tree with
#      amd_ci/homelink/coalesce_pr187.patch applied: build-coal (targets llama-server, ggml-rpc-server)
#   4. master only: the public GGUF (no token)
# Usage: bash amd_ci/homelink/setup.sh master|worker
set -uo pipefail
ROLE="$1"
W="$AMD_CI_WORK"
HERE="$(cd "$(dirname "$0")" && pwd)"
REF="${HL_REF:-cluster/home-link}"
mkdir -p "$W/src" "$W/out" "$W/home"
export HOME="$W/home" PIP_CACHE_DIR="$W/pipcache" HF_HOME="$W/hf"

echo "== clone unsloth@$REF"
git clone -q --depth 1 -b "$REF" https://github.com/unslothai/unsloth "$W/src/unsloth" || { echo "FATAL: clone"; exit 1; }
git -C "$W/src/unsloth" rev-parse HEAD | tee "$W/out/unsloth_sha.txt"

echo "== venv"
python3 -m venv "$W/venv" || { echo "FATAL: venv"; exit 1; }
. "$W/venv/bin/activate"
python3 -m pip install -q --upgrade pip >/dev/null 2>&1
python3 -m pip install -q cryptography psutil huggingface_hub structlog typer || { echo "FATAL: pip"; exit 1; }
REQ="$W/src/unsloth/studio/backend/requirements/no-torch-runtime.txt"
[ -f "$REQ" ] && python3 -m pip install -q -r "$REQ" 2>&1 | tail -3
python3 - "$W/src/unsloth" <<'EOF'
import re, subprocess, sys
repo = sys.argv[1]
probe = ("import sys; sys.path[:0] = [{r!r}, {r!r} + '/studio/backend']; "
         "from core.inference import cluster_serving, spark_serving; "
         "from studio.cluster_link import tls, identity; from unsloth_cli import app").format(r = repo)
names = {"yaml": "pyyaml", "jwt": "pyjwt", "dateutil": "python-dateutil", "multipart": "python-multipart",
         "dotenv": "python-dotenv", "google": "protobuf", "PIL": "pillow", "sklearn": "scikit-learn"}
for _ in range(40):
    r = subprocess.run([sys.executable, "-c", probe], capture_output = True, text = True)
    if r.returncode == 0:
        print("imports ok"); break
    m = re.search(r"No module named '([\w\.]+)'", r.stderr)
    if not m:
        print(r.stderr[-2000:]); sys.exit("FATAL: import probe")
    name = m.group(1).split(".")[0]
    pkg = names.get(name, name.replace("_", "-"))
    print("installing", pkg, flush = True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])
else:
    sys.exit("FATAL: import probe did not converge")
EOF
[ $? -eq 0 ] || exit 1

echo "== llama.cpp pull/187/head, CPU + RPC"
L="$W/llama"
git clone -q --filter=blob:none https://github.com/unslothai/llama.cpp "$L" || { echo "FATAL: llama clone"; exit 1; }
git -C "$L" fetch -q origin pull/187/head && git -C "$L" checkout -q FETCH_HEAD || { echo "FATAL: fetch 187"; exit 1; }
git -C "$L" rev-parse HEAD | tee "$W/out/llama_sha.txt"
FLAGS="-DCMAKE_BUILD_TYPE=Release -DGGML_RPC=ON -DGGML_NATIVE=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF"
command -v ccache >/dev/null && FLAGS="$FLAGS -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
J=$(nproc)
build() {
  cmake -S "$L" -B "$L/$1" $FLAGS > "$W/out/cmake_$1.log" 2>&1 \
    && cmake --build "$L/$1" -j "$J" --target llama-server ggml-rpc-server > "$W/out/make_$1.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] && [ -x "$L/$1/bin/ggml-rpc-server" ] && [ -x "$L/$1/bin/llama-server" ] \
    || { echo "FATAL: build $1 rc=$rc"; tail -30 "$W/out/make_$1.log"; exit 1; }
  echo "built $1"
}
build build-base
git -C "$L" apply "$HERE/coalesce_pr187.patch" || { echo "FATAL: coalesce patch does not apply"; exit 1; }
git -C "$L" diff --stat | tail -1
build build-coal

if [ "$ROLE" = master ]; then
  echo "== model"
  python3 - "$W/model.txt" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
p = hf_hub_download("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf")
open(sys.argv[1], "w").write(p)
print(p)
EOF
  [ -s "$W/model.txt" ] || { echo "FATAL: model download"; exit 1; }
fi
echo "SETUP_OK"
