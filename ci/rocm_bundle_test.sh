#!/bin/bash
# Env: BUNDLE_ZIP (the #14 artifact zip). Compares it with upstream master-813-bfbef5b's ROCm zip (Studio's current fallback).
set -uo pipefail
W="$RUNNER_TEMP/sdrev-${GITHUB_RUN_ID:-local}-rocmtest-$$"; mkdir -p "$W"/{tmp,hf,models,runs,art,A,B}
echo "ART=$W/art" >> "$GITHUB_ENV"
S="$W/art/summary.md"; CI="$(cd "$(dirname "$0")" && pwd)"
log() { echo "$*" | tee -a "$S"; }
export TMPDIR="$W/tmp" HF_HOME="$W/hf" HF_HUB_ENABLE_HF_TRANSFER=1
python3 -m venv "$W/venv"; . "$W/venv/bin/activate"; pip -q install numpy pillow huggingface_hub hf_transfer >/dev/null 2>&1
( MODELS_DIR="$W/models" python - <<'PY'
import os
from huggingface_hub import hf_hub_download as d
for r, f in [("unsloth/Z-Image-Turbo-GGUF","z-image-turbo-Q4_K_M.gguf"),("unsloth/Qwen3-4B-GGUF","Qwen3-4B-Q4_K_M.gguf"),("Comfy-Org/z_image_turbo","split_files/vae/ae.safetensors"),
             ("unsloth/MiniMax-H3-GGUF","minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf"),("unsloth/MiniMax-H3-GGUF","qwen3vl_32b_minimax_h3-Q2_K_M.gguf"),("unsloth/MiniMax-H3-GGUF","vae/minimax_h3_video_vae_fp16.safetensors"),("unsloth/MiniMax-H3-GGUF","vae/minimax_h3_audio_vae_fp32.safetensors")]:
    print(d(r, f, local_dir=os.environ["MODELS_DIR"]), flush=True)
print("DONE", flush=True)
PY
) > "$W/dl.log" 2>&1 &
log "## #14 ROCm bundle on $(hostname), system ROCm $(cat /opt/rocm/.info/version 2>/dev/null)"
cd "$W/B" && unzip -q "$BUNDLE_ZIP" && B="$(dirname "$(find "$W/B" -name sd-cli -type f | head -1)")"
curl -fsSL --retry 3 -o "$W/A.zip" https://github.com/leejet/stable-diffusion.cpp/releases/download/master-813-bfbef5b/sd-master-bfbef5b-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.14.0.zip
cd "$W/A" && unzip -q "$W/A.zip" && A="$(dirname "$(find "$W/A" -name sd-cli -type f | head -1)")"
chmod +x "$A/sd-cli" "$B/sd-cli" 2>/dev/null
log "bundle B: $(du -sh "$W/B" | cut -f1), $(find "$W/B" -name '*.so*' | wc -l) libs, $(find "$W/B" -name '*.kpack' | wc -l) kpacks; zip $(du -h "$BUNDLE_ZIP" | cut -f1)"
cat "$B/UNSLOTH_BUILD.txt" 2>/dev/null | head -5 | tee -a "$S"
for n in A B; do d=$([ $n = A ] && echo "$A" || echo "$B")
  log "$n ldd outside the bundle: $(env -i PATH=/usr/bin:/bin ldd "$d/sd-cli" | awk '$3 ~ /^\// {print $3}' | grep -v "^$d" | xargs -n1 basename 2>/dev/null | sort | tr '\n' ' ')"
  log "$n not found: $(env -i PATH=/usr/bin:/bin ldd "$d/sd-cli" | grep -c 'not found')"
done
for i in $(seq 1 240); do grep -q DONE "$W/dl.log" && break; sleep 15; done
M="$W/models"
run() { # name dir args...
  local name=$1 d=$2; shift 2; mkdir -p "$W/runs/$name"
  ( cd "$W/tmp" && exec env -i PATH=/usr/bin:/bin HOME="$W/tmp" "$d/sd-cli" "$@" -o "$W/runs/$name/f_%03d.png" -v > "$W/runs/$name.log" 2>&1 ) &
  local pid=$! sys=0
  while kill -0 $pid 2>/dev/null; do grep -q '/opt/rocm' /proc/$pid/maps 2>/dev/null && sys=1; sleep 5; done
  wait $pid; local rc=$?
  log "$name rc=$rc system-rocm-mapped=$sys device=$(grep -oE 'Device 0: [^,]*, gfx[0-9a-f]+' "$W/runs/$name.log" | head -1) | $(grep -E 'generate_image completed|generate_video completed|sampling completed|no GPU|available 0.00 MB|ASSERT' "$W/runs/$name.log" | sed -E 's/.*\] //' | head -4 | tr '\n' ' ')"
}
for n in A B; do d=$([ $n = A ] && echo "$A" || echo "$B")
  run z_$n "$d" --diffusion-model "$M/z-image-turbo-Q4_K_M.gguf" --llm "$M/Qwen3-4B-Q4_K_M.gguf" --vae "$M/split_files/vae/ae.safetensors" \
    -p "A lighthouse on a rocky coast at dusk, waves crashing, detailed photograph" --cfg-scale 1.0 --steps 8 --seed 7 -W 1024 -H 1024 --diffusion-fa
  run h3_$n "$d" -M vid_gen --diffusion-model "$M/minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf" --vae "$M/vae/minimax_h3_video_vae_fp16.safetensors" \
    --audio-vae "$M/vae/minimax_h3_audio_vae_fp32.safetensors" --llm "$M/qwen3vl_32b_minimax_h3-Q2_K_M.gguf" \
    -p "A red fox trots through fresh snow in a pine forest at sunrise." --cfg-scale 1.0 -W 640 -H 384 --video-frames 56 --steps 4 --seed 42 --rng cpu --fps 24 --diffusion-fa --offload-to-cpu --max-vram -1
done
cp "$W"/runs/*.log "$W/art/" 2>/dev/null; for n in z_A z_B; do f=$(ls "$W/runs/$n"/*.png 2>/dev/null | head -1); [ -n "$f" ] && cp "$f" "$W/art/$n.png"; done
cat "$S"
