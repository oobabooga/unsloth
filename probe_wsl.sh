#!/bin/bash
# Inside a throwaway WSL distro, as root: ROCm-on-WSL by the repo's helper, then vLLM's ROCm wheels.
set -x
HERE=$(dirname "$0")
export DEBIAN_FRONTEND=noninteractive
date
ls -l /dev/dxg
ls "/mnt/c/Program Files (x86)/Windows Kits/10/Include" 2>&1 | head
tr -d "\r" < "$HERE/install_rocm_wsl_strixhalo.sh" > /root/helper.sh; time bash /root/helper.sh > /root/rocm-wsl.log 2>&1
echo "helper exit $?"
tail -40 /root/rocm-wsl.log
export HSA_ENABLE_DXG_DETECTION=1 PATH=/opt/rocm/bin:$PATH
cat /opt/rocm/.info/version
ls /opt/rocm/lib | grep -E 'rocdxg|libhsa-runtime' | head
du -sh /opt/rocm-* 2>/dev/null
rocminfo 2>&1 | grep -E 'Marketing Name|Name: +gfx' | head -6
apt-get install -y -q libopenmpi3t64 libnuma1 python3.12-venv roctracer > /root/apt-extra.log 2>&1; echo "apt extra exit $?"; tail -3 /root/apt-extra.log
mkdir -p /root/stub
tr -d '\r' < /root/rocprofiler_stub.c > /root/stub.c
gcc -shared -fPIC -O2 -Wl,-soname,librocprofiler-sdk.so.1 -o /root/stub/librocprofiler-sdk.so.1 /root/stub.c; echo "stub exit $?"
for f in /opt/rocm/lib/*.so*; do readelf -d "$f" 2>/dev/null | grep -q 'librocprofiler-sdk.so' && echo "needs sdk: $f"; done
export LD_LIBRARY_PATH=/root/stub
for lib in libamdhip64.so.7 libhiprtc.so.7 libMIOpen.so.1 librocblas.so.5 libhipblas.so.3 libhipblaslt.so.1 libhipfft.so.0 libhiprand.so.1 libhipsparse.so.4 libhipsparselt.so.0 libhipsolver.so.1 librocsolver.so.0 librccl.so.1 libroctx64.so.4; do
  [ -e /opt/rocm/lib/$lib ] && echo "have $lib" || echo "MISSING $lib"
done
ldconfig -p | grep -E 'libmpi.so.40|libmpi_cxx.so.40|libnuma.so.1'
du -sh /opt/rocm-7.2.1
python3 -m venv /root/boot && /root/boot/bin/pip -q install uv
time /root/boot/bin/uv venv --python /usr/bin/python3.12 /root/venv
time /root/boot/bin/uv pip install --python /root/venv/bin/python "vllm==0.30.0+rocm723" \
  --extra-index-url https://wheels.vllm.ai/rocm/0.30.0/rocm723 --index-strategy unsafe-best-match > /root/pip.log 2>&1
echo "pip exit $?"; tail -3 /root/pip.log
/root/venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "hip", torch.version.hip)
print("avail", torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(p.name, getattr(p, "gcnArchName", None), p.total_memory >> 20, "MiB")
    print("mem_get_info", [x >> 20 for x in torch.cuda.mem_get_info()])
    a = torch.randn(1024, 1024, device = "cuda", dtype = torch.bfloat16)
    print("matmul", (a @ a).float().abs().mean().item())
EOF
/root/venv/bin/python -c "import vllm, vllm._C; print('vllm', vllm.__version__)"
/root/venv/bin/python -c "import amdsmi; amdsmi.amdsmi_init(); print('amdsmi ok', len(amdsmi.amdsmi_get_processor_handles()))" 2>&1 | tail -2
export HF_HOME=/root/hf
nproc; free -g
timeout 900 /root/venv/bin/python /root/alloc_probe.py; echo "alloc exit $?"
free -g
serve() {
  util=$1
  /root/venv/bin/python -m vllm.entrypoints.openai.api_server --model unsloth/Qwen2.5-0.5B-Instruct \
    --max-model-len 4096 --gpu-memory-utilization "$util" --host 127.0.0.1 --port 8011 > /root/serve.log 2>&1 &
  PID=$!
  for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8011/health >/dev/null; then echo "healthy after $((i*5))s"; break; fi
    if ! kill -0 $PID 2>/dev/null; then echo "server died"; break; fi
    [ $((i % 6)) = 0 ] && echo "t=$((i*5))s $(grep -E 'MemAvailable' /proc/meminfo) last: $(tail -c 200 /root/serve.log | tr '\n' ' ')"
    sleep 5
  done
  curl -s --max-time 60 http://127.0.0.1:8011/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"unsloth/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":16,"temperature":0}'; echo
  kill $PID 2>/dev/null; sleep 10; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  grep -E 'KV cache|memory|Memory|utilization|ERROR|Error' /root/serve.log | grep -v 'GET /health' | tail -20
}
echo "=== util 0.9"; serve 0.9
echo "=== util 0.5"; serve 0.5
date
