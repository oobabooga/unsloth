#!/bin/sh
set -eu
export DEBIAN_FRONTEND=noninteractive
if [ ! -e /dev/dxg ]; then
  echo "WSL cannot see the AMD GPU. Update the AMD Adrenalin driver, restart Windows, then click Install again." >&2
  exit 3
fi
case "$(cat /opt/rocm/.info/version 2>/dev/null)" in
  7.2.1|7.2.1-*) installed=1 ;;
  *) installed=0 ;;
esac
if [ "$installed" = 0 ] || [ ! -e /opt/rocm/lib/librocdxg.so.1 ]; then
  install -d -m 755 /etc/apt/keyrings
  cp "$1" /etc/apt/keyrings/rocm.asc
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.asc] https://repo.radeon.com/rocm/apt/7.2.1 noble main" > /etc/apt/sources.list.d/rocm.list
  printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600
  apt-get update
  # rocm-libs is what PyTorch links (rocBLAS, MIOpen, RCCL, ...); vLLM's PyTorch also links
  # OpenMPI and the ROCm profiler SDK.
  apt-get install -y rocm-libs rocminfo hip-runtime-amd rocprofiler-sdk libopenmpi3t64
  dpkg -i "$2"
  ldconfig
fi
HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo | grep -E "Name:[[:space:]]+gfx[1-9]" || {
  echo "ROCm cannot reach the AMD GPU through WSL. Update the AMD Adrenalin driver, restart Windows, then click Install again." >&2
  exit 3
}
