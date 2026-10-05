#!/bin/bash
# Runs as root inside a fresh Ubuntu 24.04 WSL distro on the AMD Windows runner.
set -uo pipefail
STAGE=$1
export DEBIAN_FRONTEND=noninteractive
W=/opt/probe
mkdir -p $W
export HSA_ENABLE_DXG_DETECTION=1 LD_LIBRARY_PATH=/opt/rocm-wsl/lib ROCPROFILER_REGISTER_ENABLED=0
case "$STAGE" in
setup)
  python3 - <<'PY'
import hashlib, urllib.request
for url, sha, out in (
    ("https://repo.radeon.com/rocm/rocm.gpg.key", "2de99e2354646a90d9903e2a669fc4e36b02c1bbff7075c481e12d7edab2c88b", "/opt/probe/rocm.gpg.key"),
    ("https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb", "28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541", "/opt/probe/rocdxg-roct.deb"),
    ("https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-amd-smi-lib_1.2.2_amd64.deb", "f9b601457ea513a223c7dd2cc281fb0aef1d11f21f38ce6108dc76e665ad6cc5", "/opt/probe/rocdxg-amd-smi-lib.deb"),
):
    data = urllib.request.urlopen(url, timeout=120).read()
    assert hashlib.sha256(data).hexdigest() == sha, url
    open(out, "wb").write(data)
PY
  curl -fsSL -o $W/setup-rocm "$RAW/setup-rocm.sh"; chmod 755 $W/setup-rocm
  $W/setup-rocm $W/rocm.gpg.key $W/rocdxg-roct.deb $W/rocdxg-amd-smi-lib.deb > $W/setup.log 2>&1; echo "setup rc=$?"
  curl -fsSL -o $W/uv.tgz https://github.com/astral-sh/uv/releases/download/0.12.19/uv-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf $W/uv.tgz -C $W
  UV=$W/uv-x86_64-unknown-linux-gnu/uv
  export UV_CACHE_DIR=$W/uvc
  $UV venv -q --python 3.12 $W/env
  $UV pip install -q --python $W/env/bin/python --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/rocm/0.30.0/rocm723 'vllm==0.30.0+rocm723' 2>&1 | tail -3
  free -g; grep -E "MemTotal|MemAvailable|SwapTotal" /proc/meminfo
  $W/env/bin/python -c "import torch; f,t=torch.cuda.mem_get_info(0); print('torch free/total GiB', round(f/2**30,1), round(t/2**30,1)); p=torch.cuda.get_device_properties(0); print('props total GiB', round(p.total_memory/2**30,1))"
  ;;
serve)
  util=$2
  export HF_HOME=$W/hf VLLM_CACHE_ROOT=$W/vc-$util
  ( while true; do echo "$(date +%T) $(grep -E 'MemAvailable|SwapFree' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"; sleep 15; done ) > $W/mem-$util.log 2>&1 &
  MON=$!
  $W/env/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --max-model-len 4096 \
     --gpu-memory-utilization $util --host 127.0.0.1 --port 8011 > $W/serve-$util.log 2>&1 &
  PID=$!
  ok=0; start=$(date +%s)
  for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8011/health >/dev/null; then ok=1; break; fi
    kill -0 $PID 2>/dev/null || break
    sleep 5
  done
  echo "util=$util healthy=$ok after $(( $(date +%s) - start ))s"
  if [ $ok = 1 ]; then
    curl -s http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Say the capital of France in one word."}],"max_tokens":16,"temperature":0}' | head -c 300
    echo
  fi
  kill $PID 2>/dev/null; sleep 8; kill -9 $PID 2>/dev/null; pkill -9 -f vllm.entrypoints 2>/dev/null; kill $MON
  echo "--- memory"; cat $W/mem-$util.log | tail -20
  echo "--- serve log tail"; grep -vE 'Capturing|launcher.py:70' $W/serve-$util.log | tail -25 | cut -c1-300
  ;;
esac
