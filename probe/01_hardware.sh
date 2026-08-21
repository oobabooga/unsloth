#!/usr/bin/env bash
set -uxo pipefail
SCRATCH="${RUNNER_TEMP:-$HOME}/strix-repro"
uname -a
lscpu | head -25
free -h
df -h /mnt/scratch / || true
rocminfo 2>/dev/null | grep -E 'Name:|Marketing Name:|gfx|Compute Unit|Uuid' | head -40
rocm-smi 2>/dev/null | head -30
command -v amd-smi && amd-smi list 2>&1 | head -40 || echo "NO amd-smi on PATH"
command -v amd-smi && amd-smi static --asic 2>&1 | head -30
ls -l /dev/kfd /dev/dri /dev/accel* 2>/dev/null
grep -E 'gfx_target_version|simd_count' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | head
for d in /sys/class/drm/card*/device/mem_info_vram_total /sys/class/drm/card*/device/mem_info_gtt_total /sys/class/drm/card*/device/mem_info_vram_used; do
  [ -f "$d" ] && echo "$d = $(cat "$d")"
done
python3 -VV
exit 0
