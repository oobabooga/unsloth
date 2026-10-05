#!/bin/bash
# Runs as root inside a fresh Ubuntu 24.04 WSL distro on the AMD Windows runner.
set -uo pipefail
STAGE=$1
RAW=https://raw.githubusercontent.com/oobabooga/unsloth/amd-vllm-k7q2xz-probew5
export DEBIAN_FRONTEND=noninteractive
W=/opt/probe
mkdir -p $W
case "$STAGE" in
rocm)
  python3 - <<'PY'
import hashlib, urllib.request
for url, sha, out in (
    ("https://repo.radeon.com/rocm/rocm.gpg.key", "2de99e2354646a90d9903e2a669fc4e36b02c1bbff7075c481e12d7edab2c88b", "/opt/probe/rocm.gpg.key"),
    ("https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb", "28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541", "/opt/probe/rocdxg-roct.deb"),
    ("https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-amd-smi-lib_1.2.2_amd64.deb", "f9b601457ea513a223c7dd2cc281fb0aef1d11f21f38ce6108dc76e665ad6cc5", "/opt/probe/rocdxg-amd-smi-lib.deb"),
    ("https://raw.githubusercontent.com/oobabooga/unsloth/amd-vllm-k7q2xz-probew5/setup-rocm.sh", None, "/opt/probe/setup-rocm"),
):
    data = urllib.request.urlopen(url, timeout=120).read()
    assert sha is None or hashlib.sha256(data).hexdigest() == sha, url
    open(out, "wb").write(data)
print("downloads ok")
PY
  chmod 755 $W/setup-rocm
  time $W/setup-rocm $W/rocm.gpg.key $W/rocdxg-roct.deb $W/rocdxg-amd-smi-lib.deb > $W/setup.log 2>&1
  echo "setup rc=$?"
  grep -E "^(Need to get|After this operation)|E:|gfx" $W/setup.log | tail -20
  tail -5 $W/setup.log
  cat /opt/rocm/.info/version
  ;;
vllm)
  export HSA_ENABLE_DXG_DETECTION=1 LD_LIBRARY_PATH=/opt/rocm-wsl/lib
  curl -fsSL -o $W/uv.tgz https://github.com/astral-sh/uv/releases/download/0.12.19/uv-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf $W/uv.tgz -C $W
  UV=$W/uv-x86_64-unknown-linux-gnu/uv
  export UV_CACHE_DIR=$W/uvc
  $UV venv -q --python 3.12 $W/env
  $UV pip install -q --python $W/env/bin/python --index-url https://pypi.org/simple \
    --extra-index-url https://wheels.vllm.ai/rocm/0.30.0/rocm723 'vllm==0.30.0+rocm723' 2>&1 | tail -3
  for f in $W/env/lib/python3.12/site-packages/torch/lib/*.so; do ldd "$f" | grep "not found"; done | grep -vE "libc10|libtorch" | sort -u
  $W/env/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName, torch.cuda.mem_get_info()); a=torch.ones(1024,1024,device='cuda',dtype=torch.bfloat16); print((a@a).float().sum().item())"
  echo "smoke rc=$?"
  echo "== amdsmi as shipped"
  $W/env/bin/python -c "import amdsmi; amdsmi.amdsmi_init(); print('handles', amdsmi.amdsmi_get_processor_handles())" 2>&1 | tail -3
  $W/env/bin/python -c "from vllm.platforms import current_platform as p; print('platform', type(p).__name__, p.is_rocm())" 2>&1 | tail -3
  ;;
amdsmi)
  export HSA_ENABLE_DXG_DETECTION=1 LD_LIBRARY_PATH=/opt/rocm-wsl/lib
  apt-get install -y $W/rocdxg-amd-smi-lib.deb > $W/smi.log 2>&1; echo "smi deb rc=$?"; tail -3 $W/smi.log
  ls /opt/rocm-wsl/lib
  for v in "ROCM_PATH=/opt/rocm-wsl" "LD_LIBRARY_PATH=/opt/rocm-wsl/lib"; do
    echo "== amdsmi with $v"
    env $v $W/env/bin/python -c "import amdsmi; amdsmi.amdsmi_init(); h=amdsmi.amdsmi_get_processor_handles(); print('handles', h)" 2>&1 | tail -3
    env $v $W/env/bin/python -c "from vllm.platforms import current_platform as p; print('platform', type(p).__name__, p.is_rocm())" 2>&1 | tail -3
  done
  ;;
serve)
  export LD_LIBRARY_PATH=/opt/rocm-wsl/lib HSA_ENABLE_DXG_DETECTION=1 HF_HOME=$W/hf VLLM_CACHE_ROOT=$W/vc
  [ -n "${2:-}" ] && export "$2"
  echo "extra env: ${2:-none}"
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
  grep -vE 'Capturing|launcher.py:70|Loading safetensors' $W/serve.log | grep -E "ERROR|Error|error|platform|Platform|rocm|KV cache|Free memory|Traceback" | tail -40
  ;;
esac
