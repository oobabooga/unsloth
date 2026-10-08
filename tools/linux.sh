#!/usr/bin/env bash
# AMD devlab legs for the H3 stack review. Usage: linux.sh <hip|vulkan> <stage>
set -euo pipefail
BK=$1; STAGE=$2
W="$RUNNER_TEMP/w"; T="$W/tools"
export PATH="$W/venv/bin:$PATH" TMPDIR="$W/tmp" HF_HOME="$W/hf" HF_HUB_ENABLE_HF_TRANSFER=1
REFS="m:6321a6935234c2bb09c77604874a439344faa7f6 pr17:ca92d95ddf041bf885a36a7fdafbffd88706c28e pr18:5f222b85a9b52e7d6f43fdffcb0772806459154b pr19:338079ece82649c7535da4bbba663f29b4bc1a25 pr20:aff34f1542edb66acf2f5bb5daae1c895db0afe4"
if [ "$BK" = hip ]; then
  export HIPCXX=/opt/rocm/llvm/bin/clang HIP_PATH=/opt/rocm
  DEF="-DSD_HIPBLAS=ON -DGPU_TARGETS=gfx1151"
else
  export VULKAN_SDK="$W/vk/x86_64" PATH="$W/vk/x86_64/bin:$PATH"
  DEF="-DSD_VULKAN=ON"
fi
case $STAGE in
setup)
  mkdir -p "$W"/{tmp,hf,models,runs}
  python3 -m venv "$W/venv"; "$W/venv/bin/pip" -q install cmake ninja huggingface_hub hf_transfer numpy
  if [ "$BK" = vulkan ]; then
    curl -fsSL --retry 3 -o "$W/vk.tar.xz" https://sdk.lunarg.com/sdk/download/1.4.328.1/linux/vulkansdk-linux-x86_64-1.4.328.1.tar.xz
    echo "241e75b56c91c0d210ed07a7c638ec05a3e5b0e4c66ba9f0ba0f102d823ad6bf  $W/vk.tar.xz" | sha256sum -c
    mkdir -p "$W/vk" && tar -xJf "$W/vk.tar.xz" -C "$W/vk" --strip-components=1
    "$W/vk/x86_64/bin/vulkaninfo" --summary 2>/dev/null | grep -E 'deviceName|driverName' || true
  fi
  git clone -q --filter=blob:none https://github.com/oobabooga/stable-diffusion.cpp "$W/src"
  for r in $REFS; do
    n=${r%%:*}; s=${r##*:}
    git -C "$W/src" worktree add -q --detach "$W/$n" "$s"
    git -C "$W/$n" submodule update -q --init --depth 1 ggml
    k=0; for p in "$W/$n"/scripts/unsloth/ggml-patches/*.patch; do [ -e "$p" ] || continue; git -C "$W/$n/ggml" apply "$p"; k=$((k+1)); done
    python3 "$T/sdrev_dump.py" "$W/$n" >/dev/null
    echo "$n $(git -C "$W/$n" log -1 --format=%h) patches=$k"
  done ;;
build)
  for r in $REFS; do
    n=${r%%:*}; extra=""; tgt=sd-cli
    [ "$n" = pr20 ] && { extra="-DGGML_BUILD_TESTS=ON"; tgt="sd-cli test-backend-ops"; }
    cmake -S "$W/$n" -B "$W/$n/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON -DSD_SERVER_BUILD_FRONTEND=OFF \
      -DSD_WEBP=OFF -DSD_WEBM=OFF -DGGML_NATIVE=OFF $DEF $extra > "$W/$n.cmake.log" 2>&1 || { tail -40 "$W/$n.cmake.log"; exit 1; }
    grep -E "fused" "$W/$n.cmake.log" || true
    cmake --build "$W/$n/build" -j "$(nproc)" --target $tgt > "$W/$n.build.log" 2>&1 || { grep -E "error|FAILED" "$W/$n.build.log" | head -40; exit 1; }
    echo "built $n"
  done ;;
tbo)
  set +e
  for op in ROPE_PE_PERMUTE MODULATE_ROWS SWIGLU_SCALED RMS_NORM CPY CONV_2D_DW GLU FLASH_ATTN_EXT SAGE_ATTN MUL_MAT; do
    timeout 2400 "$W/pr20/build/bin/test-backend-ops" test -o $op > "$W/tbo_$op.log" 2>&1; rc=$?
    echo "TBO $op rc=$rc ok=$(grep -c ' OK$' "$W/tbo_$op.log") fail=$(grep -c FAIL "$W/tbo_$op.log") not_supported=$(grep -c 'not supported' "$W/tbo_$op.log")"
    grep FAIL "$W/tbo_$op.log" | head -10
  done
  grep -h -E '^Backend|Device description' "$W"/tbo_*.log | sort -u | head ;;
smoke)
  M="$W/models/sd_turbo.gguf"
  [ -s "$M" ] || curl -fsSL --retry 5 -o "$M" https://huggingface.co/Green-Sky/SD-Turbo-GGUF/resolve/main/sd_turbo-f16-q8_0.gguf
  for r in $REFS; do
    n=${r%%:*}; o="$W/runs/sdt_$n"; mkdir -p "$o"
    c=(-m "$M" -p "a red fox sitting in a snowy pine forest, detailed fur" --steps 2 --cfg-scale 1 -W 512 -H 512 -s 42 --rng cpu)
    "$W/$n/build/bin/sd-cli" "${c[@]}" -o "$o/plain.png" > "$o/plain.log" 2>&1 || tail -20 "$o/plain.log"
    "$W/$n/build/bin/sd-cli" "${c[@]}" --diffusion-fa --vae-tiling --vae-tile-size 24x16 -o "$o/fa_tiled.png" > "$o/fa.log" 2>&1 || tail -20 "$o/fa.log"
    SD_TILE_ASYNC_MERGE=0 "$W/$n/build/bin/sd-cli" "${c[@]}" --diffusion-fa --vae-tiling --vae-tile-size 24x16 -o "$o/fa_tiled_sync.png" > "$o/fas.log" 2>&1 || tail -20 "$o/fas.log"
    echo "SDT $n $(for f in plain fa_tiled fa_tiled_sync; do printf '%s=%s ' $f "$(sha256sum "$o/$f.png" 2>/dev/null | cut -c1-16)"; done)"
  done ;;
h3)
  python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
W = os.environ["RUNNER_TEMP"] + "/w/models"
for f in ["vae/minimax_h3_audio_vae_fp32.safetensors", "vae/minimax_h3_video_vae_fp16.safetensors",
          "minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf", "qwen3vl_32b_minimax_h3-Q2_K_M.gguf"]:
    print(hf_hub_download("unsloth/MiniMax-H3-GGUF", f, local_dir=W), flush=True)
PY
  set +e
  M="$W/models"
  run() {  # tree tag env...
    local tree=$1 tag=$2; shift 2; local o="$W/runs/$tag"; mkdir -p "$o"
    env SDREV_DUMP="$o" "$@" "$W/$tree/build/bin/sd-cli" -M vid_gen --diffusion-model "$M/minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf" \
      --vae "$M/vae/minimax_h3_video_vae_fp16.safetensors" --audio-vae "$M/vae/minimax_h3_audio_vae_fp32.safetensors" \
      --llm "$M/qwen3vl_32b_minimax_h3-Q2_K_M.gguf" -p "A silver tabby kitten surfs a tropical ocean wave on a white surfboard. Upbeat surf-rock music." \
      --cfg-scale 1.0 --steps 2 -W 512 -H 288 --video-frames 22 --seed 42 --rng cpu --diffusion-fa -o "$o/out.avi" > "$o/log.txt" 2>&1
    echo "H3 $tag exit=$? $(grep -cE 'error|ERROR' "$o/log.txt") errors; $(ls "$o" | tr '\n' ' ')"
  }
  run m h3_m
  run pr17 h3_17_vfa0 SD_H3_VAE_FLASH_ATTN=0
  run pr17 h3_17
  run pr18 h3_18
  run pr19 h3_19
  run pr19 h3_19_dw0 SD_H3_AUDIO_DIRECT_DW=0
  run pr20 h3_20
  run pr20 h3_20b
  C="python3 $T/cmpdump.py"
  echo "--- master vs #17 with SD_H3_VAE_FLASH_ATTN=0 (tiling identity, kill switch)"; $C "$W/runs/h3_m" "$W/runs/h3_17_vfa0"
  echo "--- master vs #17 default (VAE FA on: expected to differ)"; $C "$W/runs/h3_m" "$W/runs/h3_17"
  echo "--- #17 vs #18 (claimed bit-identical)"; $C "$W/runs/h3_17" "$W/runs/h3_18"
  echo "--- #18 vs #19 (video claimed identical, audio intentionally differs)"; $C "$W/runs/h3_18" "$W/runs/h3_19"
  echo "--- #18 vs #19 SD_H3_AUDIO_DIRECT_DW=0"; $C "$W/runs/h3_18" "$W/runs/h3_19_dw0"
  echo "--- #19 vs #20 (cuDNN path inert off CUDA)"; $C "$W/runs/h3_19" "$W/runs/h3_20"
  echo "--- #20 run-to-run"; $C "$W/runs/h3_20" "$W/runs/h3_20b"
  grep -hiE 'fused|fallback|not supported|unsupported|flash|warn' "$W/runs/h3_20/log.txt" | sort | uniq -c | sort -rn | head -20 ;;
esac
