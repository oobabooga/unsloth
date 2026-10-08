#!/usr/bin/env bash
# The PR bundle vs what Studio main installs today on this host (upstream ROCm, then upstream Vulkan) and
# vs the mirror's own Vulkan build, which a pin bump alone would give.
set -uo pipefail
W="$RUNNER_TEMP/sd14-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"; mkdir -p "$W"/{zips,b,m,out}; export TMPDIR="$W/tmp"
OUT="$W/out"
sec() { echo; echo "=================== $* ==================="; }
cp "$W"/dist/*.zip "$W/zips/pr_rocm.zip" 2>/dev/null || echo "NO PR BUNDLE"
get() { curl -fsSL --retry 3 -o "$W/zips/$1.zip" "$2" && echo "got $1 $(du -h "$W/zips/$1.zip" | cut -f1)" || echo "FAILED $1"; }
get up_rocm https://github.com/leejet/stable-diffusion.cpp/releases/download/master-813-bfbef5b/sd-master-bfbef5b-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.14.0.zip
get up_vulkan https://github.com/leejet/stable-diffusion.cpp/releases/download/master-813-bfbef5b/sd-master-bfbef5b-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip
get mirror_vulkan https://github.com/unslothai/stable-diffusion.cpp/releases/download/master-813-bfbef5b-u6321a69/sd-master-813-bfbef5b-u6321a69-bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip
declare -A CLI
for b in pr_rocm up_rocm up_vulkan mirror_vulkan; do
  [ -f "$W/zips/$b.zip" ] || continue
  # zipfile.extractall like Studio (symlinks aside), then chmod like Studio's _make_executable
  python3 -I -c 'import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$W/zips/$b.zip" "$W/b/$b"
  c="$(find "$W/b/$b" -name sd-cli -type f | head -1)"; chmod +x "$c" "$(dirname "$c")/sd-server" 2>/dev/null
  CLI[$b]="$c"; echo "$b: $c"
done
# upstream archives ship lib symlinks; zipfile flattens them, Studio recreates them. Do the same with unzip.
for b in up_rocm up_vulkan mirror_vulkan; do [ -n "${CLI[$b]:-}" ] && { rm -rf "$W/b/$b"; unzip -q "$W/zips/$b.zip" -d "$W/b/$b"; CLI[$b]="$(find "$W/b/$b" -name sd-cli -type f | head -1)"; chmod +x "${CLI[$b]}"; }; done

sec "host ROCm userspace"
ls -d /opt/rocm* 2>&1; cat /opt/rocm/.info/version 2>/dev/null
python3 -c '
import ctypes
for s in ("libamdhip64.so.7","libhipblas.so.3","librocblas.so.5","libgomp.so.1"):
    try: ctypes.CDLL(s); print("host loader resolves", s)
    except OSError as e: print("host loader MISSING", s)'

for b in pr_rocm up_rocm up_vulkan mirror_vulkan; do
  c="${CLI[$b]:-}"; [ -n "$c" ] || continue; d="$(dirname "$c")"
  sec "--list-devices $b (host, LD_LIBRARY_PATH=bindir as Studio's runtime_env)"
  LD_LIBRARY_PATH="$d" ldd "$c" | grep -E 'not found|amdhip64|rocblas|hsa-runtime|vulkan' || true
  LD_LIBRARY_PATH="$d" timeout 300 "$c" --list-devices; echo "exit=$?"
done

KFD_GID="$(stat -c %g /dev/kfd)"; REN_GID="$(stat -c %g "$(ls /dev/dri/renderD* | head -1)")"
dock() { local dir="$1"; shift
  docker run --rm --device /dev/kfd --device /dev/dri --group-add "$KFD_GID" --group-add "$REN_GID" \
    --security-opt seccomp=unconfined -e LD_LIBRARY_PATH=/b -v "$dir":/b -v "$W/m":/m:ro -v "$OUT":/o \
    ubuntu:24.04 sh -c "apt-get update -qq >/dev/null && apt-get install -y -qq libgomp1 >/dev/null; $*"; }
for b in pr_rocm up_rocm; do
  c="${CLI[$b]:-}"; [ -n "$c" ] || continue
  sec "$b in bare ubuntu:24.04 + libgomp1 (no ROCm userspace)"
  dock "$(dirname "$c")" 'ldd /b/sd-cli | grep -E "not found|amdhip|rocblas"; /b/sd-cli --list-devices; echo exit=$?'
done

sec models
curl -fsSL --retry 3 -o "$W/m/sd15.gguf" https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF/resolve/main/stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf; ls -la "$W/m"
ARGS='-m /m/sd15.gguf -p "a photograph of a red fox in the snow, highly detailed" -W 512 -H 512 --steps 20 --seed 42'
gen() { local name="$1" mode="$2" b="$3"; local c="${CLI[$b]:-}"; [ -n "$c" ] || { echo "skip $name"; return; }; local d; d="$(dirname "$c")"
  sec "gen $name [$mode]"; local t0=$SECONDS rc
  case "$mode" in
    host)   ( cd /; eval LD_LIBRARY_PATH="$d" timeout 1200 "$c" ${ARGS//\/m\//$W/m/} -o "$OUT/$name.png" ) > "$OUT/$name.log" 2>&1; rc=$? ;;
    shadow) ( cd /; eval LD_LIBRARY_PATH="$d:/opt/rocm/lib" timeout 1200 "$c" ${ARGS//\/m\//$W/m/} -o "$OUT/$name.png" ) > "$OUT/$name.log" 2>&1; rc=$? ;;
    bare)   dock "$d" "timeout 1200 /b/sd-cli $ARGS -o /o/$name.png" > "$OUT/$name.log" 2>&1; rc=$? ;;
  esac
  echo "rc=$rc wall=$((SECONDS - t0))s"
  grep -iE 'ggml_cuda_init|found [0-9]+ ROCm|Device 0|vulkan0|using .* backend|error|fail|abort|sampling completed|decode_first_stage completed|generate_image completed|completed, taking' "$OUT/$name.log" | tail -10
}
for pass in cold warm; do
  gen "pr_rocm_host_$pass" host pr_rocm
  gen "mirror_vulkan_host_$pass" host mirror_vulkan
  gen "up_rocm_host_$pass" host up_rocm
  gen "up_vulkan_host_$pass" host up_vulkan
done
gen pr_rocm_bare bare pr_rocm
gen up_rocm_bare bare up_rocm
[ -d /opt/rocm/lib ] && gen pr_rocm_shadow shadow pr_rocm

sec "image comparison"
python3 -m venv "$W/iv" && "$W/iv/bin/pip" -q install pillow numpy && "$W/iv/bin/python" - "$OUT" <<'PY'
import sys, glob, os, numpy as np
from PIL import Image
out = sys.argv[1]; fs = sorted(glob.glob(out + "/*.png"))
ref = next((f for f in fs if "mirror_vulkan_host_cold" in f), fs[0] if fs else None)
if ref:
    r = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float32)
    for f in fs:
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
        print(f"{os.path.basename(f):34s} mean|d| vs {os.path.basename(ref)} = {np.abs(a - r).mean():.2f}  std={a.std():.1f}")
PY
