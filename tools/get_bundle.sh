#!/usr/bin/env bash
# Fetch the Unsloth prebuilt llama.cpp for gfx1151 -- the same bundle Unsloth
# Studio installs, so the binary under test is the reporter's binary.
set -e
W="$RUNNER_TEMP/uma"
mkdir -p "$W/bin" "$W/models" "$W/out"
echo "W=$W" >> "$GITHUB_ENV"
curl -fL --retry 3 -o "$W/$BUNDLE" \
  "https://github.com/unslothai/llama.cpp/releases/download/$LLAMA_TAG/$BUNDLE"
tar xzf "$W/$BUNDLE" -C "$W/bin"
rm -f "$W/$BUNDLE"
{
  echo "## Host and build"
  echo
  echo '```'
  LD_LIBRARY_PATH="$W/bin" "$W/bin/llama-server" --version 2>&1 | head -4
  uname -r
  free -h | head -2
  for f in /sys/class/drm/card*/device/mem_info_vram_total \
           /sys/class/drm/card*/device/mem_info_gtt_total; do
    [ -r "$f" ] && echo "$f = $(cat "$f")"
  done
  rocminfo 2>/dev/null | grep -i xnack | head -2
  echo '```'
} >> "$GITHUB_STEP_SUMMARY"
