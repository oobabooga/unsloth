# 'Install CUDA toolkit' step of unsloth-cuda-linux-setup; inputs.arch and inputs.cuda come from $ARCH_INPUT and $CUDA_INPUT
set -eux
case "${ARCH_INPUT}" in
  x64)   PLAT=linux-x86_64 ;;
  arm64) PLAT=linux-sbsa ;;
  *) echo "unsupported arch ${ARCH_INPUT}" >&2; exit 1 ;;
esac
# 12.8.0 maps to the 12.8.1 archives on purpose: Jimver's apt metapackage served the
# newest patch, so every Linux CUDA 12 bundle was built by nvcc V12.8.93 with libcublas
# 12.8.4.1 (run 34168016006). Shipped bits must not change with the install method.
case "${CUDA_INPUT}" in
  12.8.0) COMPONENTS="cuda_cudart:12.8.90 cuda_nvcc:12.8.93 cuda_nvrtc:12.8.93 libcublas:12.8.4.1 cuda_nvtx:12.8.90 cuda_profiler_api:12.8.90 cuda_cccl:12.8.90" ;;
  13.3)   COMPONENTS="cuda_crt:13.3.33 cuda_cudart:13.3.29 cuda_nvcc:13.3.33 cuda_nvrtc:13.3.33 libcublas:13.5.1.27 libnvvm:13.3.33 cuda_nvtx:13.3.29 cuda_profiler_api:13.3.27 cccl:13.3.3.3.1" ;;
  *) echo "no redist component table for CUDA ${CUDA_INPUT}" >&2; exit 1 ;;
esac
PREFIX="$HOME/cuda-${CUDA_INPUT}"
mkdir -p "$PREFIX"
BASE="https://developer.download.nvidia.com/compute/cuda/redist"
for cv in $COMPONENTS; do
  comp="${cv%:*}"; ver="${cv#*:}"
  name="${comp}-${PLAT}-${ver}-archive"
  curl -fsSL --retry 3 -o comp.tar.xz "$BASE/${comp}/${PLAT}/${name}.tar.xz"
  tar -xf comp.tar.xz
  cp -a "${name}/." "$PREFIX"/
  rm -rf comp.tar.xz "${name}"
done
# Redist ships libs under lib/; some CMake CUDA discovery expects lib64.
ln -sfn lib "$PREFIX/lib64"
{
  echo "CUDA_PATH=$PREFIX"
  echo "CUDA_HOME=$PREFIX"
  echo "LD_LIBRARY_PATH=$PREFIX/lib:${LD_LIBRARY_PATH:-}"
} >> "$GITHUB_ENV"
echo "$PREFIX/bin" >> "$GITHUB_PATH"
