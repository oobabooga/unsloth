#!/bin/bash
# PR 16857 cells on gfx1151 Linux: published unslothai/llama.cpp bundles (ROCm gfx1151 + Vulkan), no rebuild.
set +e
W="$AMD_CI_WORK"; OUT="$W/out"; R="$W/p16857"; mkdir -p "$OUT" "$R/bins/rocm" "$R/bins/vulkan" "$R/models"
TAG=${TAG:-b12128-mix-2f23ac9}
{ echo "runner ${RUNNER_NAME:-?}"; uname -a; free -m; nproc; df -h "$W";
  for f in /sys/class/drm/card*/device/mem_info_vram_total /sys/class/drm/card*/device/mem_info_gtt_total; do [ -r "$f" ] && echo "$f $(cat "$f")"; done;
  ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1; ldconfig -p | grep -i libvulkan; } > "$OUT/diag.txt" 2>&1
cat "$OUT/diag.txt"
M="$R/models/gemma-4-E2B-it-BF16.gguf"
curl -fL --retry 8 --retry-delay 10 -s -S -o "$M" \
  "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/0314792d7f1f7e229411f620751375812bb9faf2/gemma-4-E2B-it-BF16.gguf" &
MPID=$!
curl -fL --retry 5 -s -S "https://github.com/unslothai/llama.cpp/releases/download/$TAG/app-$TAG-linux-x64-rocm-gfx1151.tar.gz" | tar xz -C "$R/bins/rocm"
echo "rocm bundle rc=${PIPESTATUS[0]}" | tee -a "$OUT/diag.txt"
curl -fL --retry 5 -s -S "https://github.com/unslothai/llama.cpp/releases/download/$TAG/app-$TAG-linux-x64-vulkan.tar.gz" | tar xz -C "$R/bins/vulkan"
echo "vulkan bundle rc=${PIPESTATUS[0]}" | tee -a "$OUT/diag.txt"
wait $MPID; echo "model rc=$? size $(stat -c %s "$M")" | tee -a "$OUT/diag.txt"
test "$(stat -c %s "$M")" = 9311305568 || { echo "model size mismatch"; exit 1; }
cat "$M" > /dev/null   # warm the page cache so no cell pays the cold read alone
python3 -u amd_ci/pr16857/cells.py --bins "$R/bins" --backends rocm,vulkan --model "$M" --out "$OUT" \
  --ballast-backend "${BALLAST_BACKEND:-rocm}" 2>&1 | tee "$OUT/cells_console.log"
test -s "$OUT/cells.jsonl"
