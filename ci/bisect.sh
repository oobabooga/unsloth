#!/bin/bash
# Env: BACKEND, VARIANTS = space list of name:sha:npatches (npatches = how many ggml patches of THAT tree, or "all"),
#      PATCH_SRC (sha whose scripts/unsloth/ggml-patches supplies the patches; default = variant sha), STEPS, REPS
set -uo pipefail
W="$RUNNER_TEMP/sdrev-${GITHUB_RUN_ID:-local}-bisect-$$"; mkdir -p "$W"/{tmp,hf,models,runs,art}
echo "ART=$W/art" >> "$GITHUB_ENV"
S="$W/art/summary.md"; CI="$(cd "$(dirname "$0")" && pwd)"
log() { echo "$*" | tee -a "$S"; }
export TMPDIR="$W/tmp" HF_HOME="$W/hf" HF_HUB_ENABLE_HF_TRANSFER=1
python3 -m venv "$W/venv"; . "$W/venv/bin/activate"; pip -q install cmake ninja numpy pillow huggingface_hub hf_transfer >/dev/null 2>&1
( MODELS_DIR="$W/models" python - <<'PY'
import os
from huggingface_hub import hf_hub_download as d
for f in ["minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf","qwen3vl_32b_minimax_h3-Q2_K_M.gguf","vae/minimax_h3_video_vae_fp16.safetensors","vae/minimax_h3_audio_vae_fp32.safetensors"]:
    print(d("unsloth/MiniMax-H3-GGUF", f, local_dir=os.environ["MODELS_DIR"]), flush=True)
print("DONE", flush=True)
PY
) > "$W/dl.log" 2>&1 &
case "$BACKEND" in
  hip) export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm PATH=/opt/rocm/bin:$PATH
       BFLAGS="-DSD_HIPBLAS=ON -DGPU_TARGETS=gfx1151 -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm";;
  vulkan) curl -fsSL --retry 3 -o "$W/vk.tar.xz" https://sdk.lunarg.com/sdk/download/1.4.328.1/linux/vulkansdk-linux-x86_64-1.4.328.1.tar.xz
       echo "241e75b56c91c0d210ed07a7c638ec05a3e5b0e4c66ba9f0ba0f102d823ad6bf  $W/vk.tar.xz" | sha256sum -c >/dev/null || exit 1
       mkdir -p "$W/vk" && tar -xJf "$W/vk.tar.xz" -C "$W/vk" --strip-components=1
       export VULKAN_SDK="$W/vk/x86_64" PATH="$W/vk/x86_64/bin:$PATH" LD_LIBRARY_PATH="$W/vk/x86_64/lib:${LD_LIBRARY_PATH:-}" CMAKE_PREFIX_PATH="$W/vk/x86_64"
       BFLAGS="-DSD_VULKAN=ON";;
esac
git clone -q --filter=blob:none "https://github.com/${SRC_REPO:-unslothai}/stable-diffusion.cpp" "$W/src"
git clone -q https://github.com/leejet/ggml "$W/ggml-clone"
for v in $VARIANTS; do
  IFS=: read -r name sha np psrc <<< "$v"; psrc=${psrc:-$sha}
  git -C "$W/src" worktree add -q --detach "$W/$name" "$sha" || { log "checkout $name failed"; exit 1; }
  g=$(git -C "$W/$name" ls-tree HEAD ggml | awk '{print $3}')
  rm -rf "$W/$name/ggml"; git clone -q --shared "$W/ggml-clone" "$W/$name/ggml"
  git -C "$W/$name/ggml" fetch -q https://github.com/leejet/ggml "$g" && git -C "$W/$name/ggml" checkout -q "$g" || exit 1
  mkdir -p "$W/p-$name"; git -C "$W/src" archive "$psrc" scripts/unsloth/ggml-patches 2>/dev/null | tar -x -C "$W/p-$name"
  i=0; for p in $(ls "$W/p-$name"/scripts/unsloth/ggml-patches/*.patch 2>/dev/null | sort); do
    [ "$np" != all ] && [ $i -ge $np ] && break
    git -C "$W/$name/ggml" apply "$p" || { log "PATCH FAIL $name $p"; exit 1; }; i=$((i+1)); done
  python "$CI/harness.py" "$W/$name/examples/cli/main.cpp" >/dev/null
  [ "$BACKEND" = hip ] && sed -i 's/if (total_bytes > 0 \&\& free_bytes > total_bytes) {/if (total_bytes > 0 \&\& free_bytes > total_bytes \&\& sd_backend_is(backend, "Vulkan")) {/' "$W/$name/src/model_manager.cpp"
  cmake -S "$W/$name" -B "$W/$name/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON -DSD_SERVER_BUILD_FRONTEND=OFF \
    -DSD_WEBP=OFF -DSD_WEBM=OFF -DGGML_NATIVE=OFF $BFLAGS > "$W/cmake-$name.log" 2>&1 && \
  cmake --build "$W/$name/build" -j "$(nproc)" --target sd-cli > "$W/build-$name.log" 2>&1 || { tail -30 "$W/build-$name.log"; log "BUILD FAIL $name"; exit 1; }
  log "variant $name: src $(git -C "$W/$name" log --oneline -1 | cut -c1-50) + $i patches from $psrc"
done
for i in $(seq 1 240); do grep -q DONE "$W/dl.log" && break; sleep 15; done
M="$W/models"
for r in $(seq 1 ${REPS:-2}); do for v in $VARIANTS; do name=${v%%:*}
  mkdir -p "$W/runs/$name$r"
  "$W/$name/build/bin/sd-cli" -M vid_gen --diffusion-model "$M/minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf" \
    --vae "$M/vae/minimax_h3_video_vae_fp16.safetensors" --audio-vae "$M/vae/minimax_h3_audio_vae_fp32.safetensors" \
    --llm "$M/qwen3vl_32b_minimax_h3-Q2_K_M.gguf" -p "A red fox trots through fresh snow in a pine forest at sunrise, breath steaming, soft crunching footsteps and distant birdsong." \
    --cfg-scale 1.0 -W 640 -H 384 --video-frames 56 --steps ${STEPS:-2} --seed 42 --rng cpu --fps 24 --diffusion-fa --offload-to-cpu --max-vram -1 -v \
    -o "$W/runs/$name$r/f_%03d.png" > "$W/runs/$name$r.log" 2>&1
  log "$name rep$r rc=$? steps: $(grep -oE '[0-9]+/[0-9]+ - [0-9.]+s/it' "$W/runs/$name$r.log" | awk '{print $3}' | tr '\n' ' ') | $(grep -E 'sampling completed|decode_first_stage completed' "$W/runs/$name$r.log" | sed -E 's/.*- //; s/ completed, taking / /' | tr '\n' ' ')"
done; done
first=${VARIANTS%% *}; first=${first%%:*}
for v in $VARIANTS; do name=${v%%:*}; log "cmp $first vs $name: $(python "$CI/compare.py" "$W/runs/${first}1" "$W/runs/${name}1")"; done
cp "$W"/runs/*.log "$W/art/" 2>/dev/null; cat "$S"
