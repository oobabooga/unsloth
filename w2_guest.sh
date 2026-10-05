#!/bin/bash
# Runs as root inside a fresh Ubuntu 24.04 WSL distro on the AMD Windows runner.
set -uo pipefail
STAGE=$1
export DEBIAN_FRONTEND=noninteractive
W=/opt/probe
mkdir -p $W
case "$STAGE" in
rocm)
  apt-get update -y >/dev/null && apt-get install -y curl >/dev/null
  curl -fsSL -o $W/helper.sh "https://raw.githubusercontent.com/unslothai/unsloth/b1d829182f8c490a326cf7690f156d2de06381f2/scripts/install_rocm_wsl_strixhalo.sh"
  time UNSLOTH_WSL_SMOKE_TEST=0 bash $W/helper.sh 2>&1 | grep -vE '^(Get|Hit|Unpacking|Selecting|Preparing|Setting up|Processing)' | tail -80
  echo "helper rc=${PIPESTATUS[0]}"
  du -sh /opt/rocm-* 2>/dev/null
  ls /opt/rocm/lib | grep -i dxg
  ;;
vllm)
  export HSA_ENABLE_DXG_DETECTION=1
  rocminfo 2>/dev/null | grep -E 'Marketing Name:|Name: +gfx' | head
  curl -fsSL -o $W/uv.tgz https://github.com/astral-sh/uv/releases/download/0.12.19/uv-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf $W/uv.tgz -C $W
  UV=$W/uv-x86_64-unknown-linux-gnu/uv
  export UV_CACHE_DIR=$W/uvc
  $UV venv --python 3.12 $W/env
  time $UV pip install --python $W/env/bin/python --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/rocm/0.30.0/rocm723 'vllm==0.30.0+rocm723' 2>&1 | tail -3
  $W/env/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName, torch.cuda.mem_get_info()); a=torch.ones(1024,1024,device='cuda',dtype=torch.bfloat16); print((a@a).float().sum().item())"
  echo "smoke rc=$?"
  ;;
serve)
  export HSA_ENABLE_DXG_DETECTION=1 HF_HOME=$W/hf VLLM_CACHE_ROOT=$W/vc
  $W/env/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --max-model-len 4096 \
     --gpu-memory-utilization 0.3 --host 127.0.0.1 --port 8011 > $W/serve.log 2>&1 &
  PID=$!
  ok=0
  for i in $(seq 1 180); do
    if curl -sf http://127.0.0.1:8011/health >/dev/null; then ok=1; break; fi
    kill -0 $PID 2>/dev/null || break
    sleep 5
  done
  echo "healthy=$ok after $((i*5))s"
  if [ $ok = 1 ]; then
    curl -s http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Say the capital of France in one word."}],"max_tokens":16,"temperature":0}'
    echo
  fi
  kill $PID 2>/dev/null; sleep 5
  grep -vE 'Capturing|launcher.py:70' $W/serve.log | tail -60
  ;;
esac
