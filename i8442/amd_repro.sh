#!/usr/bin/env bash
# unslothai/unsloth#8442 repro on gfx1151: Studio's llama.cpp prebuilts (reporter-era and current), raw llama-server.
set +e
set -uo pipefail
W="$AMD_CI_WORK"
OUT="$W/out/i8442"; mkdir -p "$OUT/logs" "$W/models" "$W/builds"
MODEL="$W/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
if [ ! -s "$MODEL" ]; then
  curl -fsSL --retry 5 -o "$MODEL" "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" || { echo "FATAL: model download"; exit 1; }
fi
ls -la "$MODEL"
fetch() {  # tag asset_suffix -> prints llama-server path
  local tag="$1" suf="$2" d="$W/builds/$1-$2"
  mkdir -p "$d"
  if [ ! -f "$d/.ok" ]; then
    curl -fsSL --retry 5 -o "$d/a.tar.gz" "https://github.com/unslothai/llama.cpp/releases/download/$tag/app-$tag-linux-x64-$suf.tar.gz" || return 1
    tar -xzf "$d/a.tar.gz" -C "$d" && rm -f "$d/a.tar.gz" && touch "$d/.ok"
  fi
  find "$d" -name llama-server -type f -perm -u+x | head -1
}
SYS_ROCM=""
for d in /opt/rocm/lib /opt/rocm/lib64; do [ -d "$d" ] && SYS_ROCM="$SYS_ROCM:$d"; done
SYS_ROCM="${SYS_ROCM#:}"
echo "system ROCm lib dirs: $SYS_ROCM"; ls -d /opt/rocm* 2>/dev/null; cat /opt/rocm/.info/version 2>/dev/null
run() {  # label tag suffix configs
  local label="$1" bin libpath
  bin="$(fetch "$2" "$3")"
  if [ -z "$bin" ]; then echo "[$label] FETCH FAILED"; return; fi
  echo "=== $label: $bin"
  # Studio's Linux order (llama_cpp.py _build_llama_env): system ROCm before the bundle's HIP runtime.
  libpath="$(dirname "$bin")"
  case "$3" in rocm*) [ -n "$SYS_ROCM" ] && libpath="$SYS_ROCM:$libpath" ;; esac
  echo "LD_LIBRARY_PATH=$libpath"
  LD_LIBRARY_PATH="$libpath" "$bin" --version 2>&1 | tail -3
  mkdir -p "$OUT/logs/$label"
  LD_LIBRARY_PATH="$libpath" timeout 3000 python3 -u i8442/raw_server_diff.py \
    --binary "$bin" --model "$MODEL" --configs "$4" --repeats 3 --identity-rounds 10 \
    --out "$OUT/$label.json" --logdir "$OUT/logs/$label" 2>&1 | tee "$OUT/$label.txt"
}
run rocm_b11541_now "b11541-mix-2500203" "rocm-gfx1151" "studio_default,no_mtp_np4,no_mtp_np4_split,no_mtp_np1"
run rocm_b10342_aug "b10342-mix-e34b418" "rocm-gfx1151" "studio_default,no_mtp_np4,no_mtp_np4_split,no_mtp_np1"
run vulkan_b11541_now "b11541-mix-2500203" "vulkan" "studio_default"
echo "--- devices llama-server used:"; for f in "$OUT"/logs/*/studio_default_cold.log; do echo "$f"; grep -m3 -E "ROCm devices|Device 0:|ggml_vulkan: 0 =|ROCm error|rocBLAS error" "$f"; done
for f in "$OUT"/*.txt; do echo "$f"; grep -h "SUMMARY" "$f"; done | tee "$OUT/SUMMARY.txt"
exit 0
