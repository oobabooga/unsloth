#!/bin/bash
# Env: BACKEND (hip|vulkan|cpu) BASE HEAD (shas of unslothai/stable-diffusion.cpp) TBO_OPS (comma list or empty)
#      H3 (1 = H3 renders) H3_HEAD_ENVS (';'-separated extra head arms, each "name:VAR=1 VAR2=0") ZIMG (space list WxH[:flags])
#      HEAD_CMAKE (extra cmake args for head) SRC_REPO (default unslothai)
set -uo pipefail
W="$RUNNER_TEMP/w"; rm -rf "$W"; mkdir -p "$W"/{tmp,hf,models,runs}
S="$W/summary.md"; : > "$S"
CI="$(cd "$(dirname "$0")" && pwd)"
log() { echo "$*" | tee -a "$S"; }
export TMPDIR="$W/tmp" HF_HOME="$W/hf" HF_HUB_ENABLE_HF_TRANSFER=1
log "## $BACKEND base=$BASE head=$HEAD"
log "host $(hostname) $(uname -r) $(nproc) threads, $(free -g | awk '/Mem/{print $2}') GB"
python3 -m venv "$W/venv"; . "$W/venv/bin/activate"
pip -q install cmake ninja numpy pillow huggingface_hub hf_transfer >/dev/null 2>&1 || { log "pip failed"; exit 1; }

# models in background
DL=()
[ "${H3:-0}" = 1 ] && DL+=("unsloth/MiniMax-H3-GGUF:minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf" "unsloth/MiniMax-H3-GGUF:qwen3vl_32b_minimax_h3-Q2_K_M.gguf" "unsloth/MiniMax-H3-GGUF:vae/minimax_h3_video_vae_fp16.safetensors" "unsloth/MiniMax-H3-GGUF:vae/minimax_h3_audio_vae_fp32.safetensors")
[ -n "${ZIMG:-}" ] && DL+=("unsloth/Z-Image-Turbo-GGUF:z-image-turbo-Q4_K_M.gguf" "unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf" "Comfy-Org/z_image_turbo:split_files/vae/ae.safetensors")
( MODELS_DIR="$W/models" python - "${DL[@]}" <<'PY'
import sys, os
from huggingface_hub import hf_hub_download as d
for a in sys.argv[1:]:
    r, f = a.split(":", 1)
    for t in range(3):
        try: print(d(r, f, local_dir=os.environ["MODELS_DIR"]), flush=True); break
        except Exception as e: print("retry", f, e, flush=True)
print("DONE", flush=True)
PY
) > "$W/dl.log" 2>&1 &

# toolchains
case "$BACKEND" in
  hip)
    export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm PATH=/opt/rocm/bin:$PATH
    BFLAGS="-DSD_HIPBLAS=ON -DGPU_TARGETS=gfx1151 -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_PREFIX_PATH=/opt/rocm"
    DEV=ROCm0; log "rocm $(cat /opt/rocm/.info/version 2>/dev/null)";;
  vulkan)
    curl -fsSL --retry 3 -o "$W/vk.tar.xz" https://sdk.lunarg.com/sdk/download/1.4.328.1/linux/vulkansdk-linux-x86_64-1.4.328.1.tar.xz
    echo "241e75b56c91c0d210ed07a7c638ec05a3e5b0e4c66ba9f0ba0f102d823ad6bf  $W/vk.tar.xz" | sha256sum -c >/dev/null || { log "vulkan sdk sha mismatch"; exit 1; }
    mkdir -p "$W/vk" && tar -xJf "$W/vk.tar.xz" -C "$W/vk" --strip-components=1
    export VULKAN_SDK="$W/vk/x86_64" PATH="$W/vk/x86_64/bin:$PATH" LD_LIBRARY_PATH="$W/vk/x86_64/lib:${LD_LIBRARY_PATH:-}"
    export CMAKE_PREFIX_PATH="$VULKAN_SDK"
    BFLAGS="-DSD_VULKAN=ON"; DEV=Vulkan0;;
  cpu) BFLAGS=""; DEV=CPU;;
esac

# sources
git clone -q --filter=blob:none "https://github.com/${SRC_REPO:-unslothai}/stable-diffusion.cpp" "$W/src"
git -C "$W/src" fetch -q origin "$BASE" "$HEAD" 2>/dev/null || true
git clone -q https://github.com/leejet/ggml "$W/ggml-clone"
for n in base head; do
  sha=$([ $n = base ] && echo "$BASE" || echo "$HEAD")
  git -C "$W/src" worktree add -q --detach "$W/$n" "$sha" || { log "checkout $n $sha failed"; exit 1; }
  g=$(git -C "$W/$n" ls-tree HEAD ggml | awk '{print $3}')
  rm -rf "$W/$n/ggml"; git clone -q --shared "$W/ggml-clone" "$W/$n/ggml"
  git -C "$W/$n/ggml" fetch -q https://github.com/leejet/ggml "$g" && git -C "$W/$n/ggml" checkout -q "$g" || { log "ggml checkout $g failed"; exit 1; }
  [ "$(git -C "$W/$n/ggml" rev-parse HEAD)" = "$g" ] || { log "ggml at wrong commit"; exit 1; }
  np=0; for p in "$W/$n"/scripts/unsloth/ggml-patches/*.patch; do [ -f "$p" ] || continue; git -C "$W/$n/ggml" apply "$p" || { log "PATCH FAIL $n $p"; exit 1; }; np=$((np+1)); done
  python "$CI/harness.py" "$W/$n/examples/cli/main.cpp" >/dev/null
  log "$n $(git -C "$W/$n" log --oneline -1 | cut -c1-70) ggml=${g:0:8} patches=$np"
done

# build
for n in base head; do
  extra=""; [ $n = head ] && extra="${HEAD_CMAKE:-}"
  t0=$(date +%s)
  cmake -S "$W/$n" -B "$W/$n/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON -DSD_SERVER_BUILD_FRONTEND=OFF \
    -DSD_WEBP=OFF -DSD_WEBM=OFF -DGGML_NATIVE=OFF -DGGML_BUILD_TESTS=ON $BFLAGS $extra > "$W/cmake-$n.log" 2>&1 \
    || { tail -40 "$W/cmake-$n.log"; log "CMAKE FAIL $n"; exit 1; }
  cmake --build "$W/$n/build" -j "$(nproc)" --target sd-cli test-backend-ops > "$W/build-$n.log" 2>&1 \
    || { grep -E "error|FAILED" "$W/build-$n.log" | head -40; log "BUILD FAIL $n"; exit 1; }
  log "built $n in $(( $(date +%s) - t0 ))s; warnings: $(grep -c 'warning:' "$W/build-$n.log")"
done

# device present?
"$W/head/build/bin/test-backend-ops" -o ADD > "$W/devcheck.log" 2>&1
if [ "$DEV" != CPU ] && ! grep -q "$DEV" "$W/devcheck.log"; then cat "$W/devcheck.log"; log "DEVICE $DEV NOT FOUND"; exit 1; fi
log "device: $(grep -m1 -E "Backend name|$DEV" "$W/devcheck.log" | head -1)"

# test-backend-ops
if [ -n "${TBO_OPS:-}" ]; then
  for op in ${TBO_OPS//,/ }; do
    for n in base head; do
      to=$(timeout 3600 "$W/$n/build/bin/test-backend-ops" test -b "$DEV" -o "$op" 2>&1 | tee "$W/tbo-$op-$n.log" | grep -E "tests passed" | tail -1)
      nf=$(grep -c "\[FAIL\]" "$W/tbo-$op-$n.log"); ns=$(grep -c "not supported" "$W/tbo-$op-$n.log")
      log "tbo $op $n: ${to:-no result} fail=$nf unsupported=$ns"
      [ "$nf" != 0 ] && grep "\[FAIL\]" "$W/tbo-$op-$n.log" | head -5 | tee -a "$S"
    done
  done
  # extra env variants for head (e.g. "CPY:GGML_CUDA_CPY_ROWS=0")
  for v in ${TBO_ENV_VARIANTS:-}; do op=${v%%:*}; ev=${v#*:}
    to=$(env $ev timeout 3600 "$W/head/build/bin/test-backend-ops" test -b "$DEV" -o "$op" 2>&1 | tee "$W/tbo-$op-head-env.log" | grep -E "tests passed" | tail -1)
    log "tbo $op head [$ev]: ${to:-no result} fail=$(grep -c "\[FAIL\]" "$W/tbo-$op-head-env.log")"
  done
fi

wait_models() { for i in $(seq 1 240); do grep -q DONE "$W/dl.log" && return 0; sleep 15; done; cat "$W/dl.log"; log "MODEL DOWNLOAD TIMEOUT"; exit 1; }
M="$W/models"
times() { grep -E "sampling completed|decode_first_stage completed|decoding audio latent completed|generate_image completed|generate_video completed|vae decode|decode .*taking|encode_first_stage completed|get_learned_condition completed|tiles" "$1" | sed -E 's/.*\] //' | tr '\n' ' '; }

h3() { # name tree [env...]
  local name=$1 tree=$2; shift 2
  mkdir -p "$W/runs/$name"
  /usr/bin/time -f "WALL %e s MAXRSS %M KB" env "$@" "$W/$tree/build/bin/sd-cli" -M vid_gen --diffusion-model "$M/minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf" \
    --vae "$M/vae/minimax_h3_video_vae_fp16.safetensors" --audio-vae "$M/vae/minimax_h3_audio_vae_fp32.safetensors" \
    --llm "$M/qwen3vl_32b_minimax_h3-Q2_K_M.gguf" -p "A red fox trots through fresh snow in a pine forest at sunrise, breath steaming, soft crunching footsteps and distant birdsong." \
    --cfg-scale 1.0 -W 640 -H 384 --video-frames 56 --steps 4 --seed 42 --rng cpu --fps 24 --diffusion-fa --offload-to-cpu -v \
    -o "$W/runs/$name/f_%03d.png" > "$W/runs/$name.log" 2>&1
  local rc=$?
  log "h3 $name rc=$rc $(grep WALL "$W/runs/$name.log") | $(times "$W/runs/$name.log")"
  [ $rc != 0 ] && { grep -E "ASSERT|error|ERROR|abort" "$W/runs/$name.log" | head -5 | tee -a "$S"; }
}
if [ "${H3:-0}" = 1 ]; then
  wait_models
  h3 base1 base; h3 head1 head; h3 base2 base; h3 head2 head
  log "cmp base1 base2: $(python "$CI/compare.py" "$W/runs/base1" "$W/runs/base2")"
  log "cmp head1 head2: $(python "$CI/compare.py" "$W/runs/head1" "$W/runs/head2")"
  log "cmp base1 head1: $(python "$CI/compare.py" "$W/runs/base1" "$W/runs/head1")"
  IFS=';' read -ra ARMS <<< "${H3_HEAD_ENVS:-}"
  for a in "${ARMS[@]}"; do [ -z "$a" ] && continue; nm=${a%%:*}; ev=${a#*:}
    h3 "head_$nm" head $ev
    log "cmp base1 head_$nm [$ev]: $(python "$CI/compare.py" "$W/runs/base1" "$W/runs/head_$nm")"
  done
fi

zimg() { # name tree WxH flags
  local name=$1 tree=$2 wh=$3; shift 3
  mkdir -p "$W/runs/$name"
  /usr/bin/time -f "WALL %e s" "$W/$tree/build/bin/sd-cli" --diffusion-model "$M/z-image-turbo-Q4_K_M.gguf" --llm "$M/Qwen3-4B-Q4_K_M.gguf" \
    --vae "$M/split_files/vae/ae.safetensors" -p "A lighthouse on a rocky coast at dusk, waves crashing, warm light in the windows, detailed photograph" \
    --cfg-scale 1.0 --steps 8 -W "${wh%x*}" -H "${wh#*x}" --seed 7 --diffusion-fa "$@" -v -o "$W/runs/$name/out.png" > "$W/runs/$name.log" 2>&1
  log "zimg $name rc=$? $(grep WALL "$W/runs/$name.log") | $(times "$W/runs/$name.log")"
}
if [ -n "${ZIMG:-}" ]; then
  wait_models
  for spec in $ZIMG; do wh=${spec%%:*}; fl=""; [ "$spec" != "$wh" ] && fl="${spec#*:}"; fl=${fl//,/ }
    k="z_${wh}_$(echo "$fl" | tr -dc 'a-z0-9')"
    zimg "${k}_base" base "$wh" $fl; zimg "${k}_head" head "$wh" $fl
    log "cmp $k: $(python "$CI/compare.py" "$W/runs/${k}_base" "$W/runs/${k}_head")"
  done
fi

# artifacts: logs, summary, a few frames
mkdir -p "$W/art"; cp "$S" "$W"/*.log "$W/art/" 2>/dev/null
for d in "$W"/runs/*/; do n=$(basename "$d"); mkdir -p "$W/art/$n"; cp "$W/runs/$n.log" "$W/art/" 2>/dev/null; ls "$d"*.png 2>/dev/null | awk 'NR==1||NR%20==0' | xargs -r cp -t "$W/art/$n"; cp "$d"*.wav "$W/art/$n/" 2>/dev/null; done
echo "ART=$W/art" >> "$GITHUB_ENV"
cat "$S"
