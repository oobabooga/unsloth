#!/usr/bin/env bash
# Setup for one box of the N-machine pipeline-parallel run, everything under $AMD_CI_WORK:
#   1. a throwaway venv (cryptography for the run's private CA, psutil)
#   2. llama.cpp CPU + RPC: unslothai/llama.cpp pull/187/head (--pipeline-groups) with
#      amd_ci/homelink/tls_pr187.patch (llama.cpp's own mutual TLS RPC transport), one build: build-tls
#   3. master only: the public GGUF ($PP_MODEL_REPO / $PP_MODEL_FILE, no token)
# Usage: bash amd_ci/pp/setup.sh master|worker
set -uo pipefail
ROLE="$1"
W="$AMD_CI_WORK"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$W/out" "$W/home"
export HOME="$W/home" PIP_CACHE_DIR="$W/pipcache" HF_HOME="$W/hf"

echo "== box"
echo "runner: ${RUNNER_NAME:-?} host: $(hostname) cpus: $(nproc) load: $(cut -d' ' -f1-3 /proc/loadavg)"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core" || true
free -g | head -2

echo "== venv"
python3 -m venv "$W/venv" || { echo "FATAL: venv"; exit 1; }
. "$W/venv/bin/activate"
python3 -m pip install -q --upgrade pip >/dev/null 2>&1
python3 -m pip install -q cryptography psutil huggingface_hub || { echo "FATAL: pip"; exit 1; }

PR="${PP_LLAMA_PR:-187}"
echo "== llama.cpp pull/$PR/head + TLS, CPU + RPC"
L="$W/llama"
git clone -q --filter=blob:none https://github.com/unslothai/llama.cpp "$L" || { echo "FATAL: llama clone"; exit 1; }
git -C "$L" fetch -q origin "pull/$PR/head" && git -C "$L" checkout -q FETCH_HEAD || { echo "FATAL: fetch $PR"; exit 1; }
git -C "$L" rev-parse HEAD | tee "$W/out/llama_sha.txt"
git -C "$L" apply "$HERE/../homelink/tls_pr$PR.patch" || { echo "FATAL: TLS patch does not apply"; exit 1; }
# client forgets cached graph uids on FREE_BUFFER like the server does (else --backend-sampling + groups aborts)
git -C "$L" apply "$HERE/../homelink/rpc_graph_cache_free.patch" || { echo "FATAL: graph cache patch does not apply"; exit 1; }
# rpc-server --repack (experimental): CPU repack kernels on the server side
if [ "$PR" = 191 ]; then git -C "$L" apply "$HERE/../homelink/rpc_repack.patch" || { echo "FATAL: repack patch does not apply"; exit 1; }; fi
# groups submit one at a time (LLAMA_GROUP_SUBMIT_SERIAL=0 turns it off for A/B)
if [ "$PR" = 191 ]; then git -C "$L" apply "$HERE/../homelink/groups_submit_serial.patch" || { echo "FATAL: submit patch does not apply"; exit 1; }; fi
FLAGS="-DCMAKE_BUILD_TYPE=Release -DGGML_RPC=ON -DGGML_NATIVE=ON -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF"
command -v ccache >/dev/null && FLAGS="$FLAGS -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
if echo '#include <openssl/ssl.h>' | ${CC:-cc} -E -x c - >/dev/null 2>&1; then
  FLAGS="$FLAGS -DGGML_RPC_TLS=ON"; echo "TLS: system OpenSSL"
else
  FLAGS="$FLAGS -DGGML_RPC_TLS=ON -DGGML_RPC_TLS_BORINGSSL=ON"; echo "TLS: no OpenSSL headers, static BoringSSL"
fi
cmake -S "$L" -B "$L/build-tls" $FLAGS > "$W/out/cmake.log" 2>&1 \
  && cmake --build "$L/build-tls" -j "$(nproc)" --target llama-server ggml-rpc-server > "$W/out/make.log" 2>&1
rc=$?
[ $rc -eq 0 ] && [ -x "$L/build-tls/bin/ggml-rpc-server" ] && [ -x "$L/build-tls/bin/llama-server" ] \
  || { echo "FATAL: build rc=$rc"; tail -30 "$W/out/make.log"; exit 1; }
"$L/build-tls/bin/ggml-rpc-server" --help 2>&1 | grep -q -- --tls-cert || { echo "FATAL: no TLS flags"; exit 1; }
"$L/build-tls/bin/llama-server" --help 2>&1 | grep -q -- --pipeline-groups || { echo "FATAL: no --pipeline-groups"; exit 1; }
if [ "$PR" = 191 ]; then "$L/build-tls/bin/ggml-rpc-server" --help 2>&1 | grep -q -- --trace || { echo "FATAL: no --trace"; exit 1; }; fi

if [ "$ROLE" = master ]; then
  echo "== model"
  python3 - "$W/model.txt" <<'EOF'
import os, sys
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ.get("PP_MODEL_REPO", "unsloth/Qwen3-4B-GGUF"),
                    os.environ.get("PP_MODEL_FILE", "Qwen3-4B-Q4_K_M.gguf"))
open(sys.argv[1], "w").write(p)
print(p, os.path.getsize(p))
EOF
  [ -s "$W/model.txt" ] || { echo "FATAL: model download"; exit 1; }
fi
echo "SETUP_OK"
