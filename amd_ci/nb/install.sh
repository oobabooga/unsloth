#!/usr/bin/env bash
set -eo pipefail
python -m pip install -qU uv --root-user-action=ignore

ROCM_TAG="$({ command -v amd-smi >/dev/null 2>&1 && amd-smi version 2>/dev/null | awk -F'ROCm version: ' 'NF>1{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}'; } || { [ -r /opt/rocm/.info/version ] && awk -F. '{print "rocm"$1"."$2; exit}' /opt/rocm/.info/version; } || { command -v hipconfig >/dev/null 2>&1 && hipconfig --version 2>/dev/null | awk -F': *' '/HIP version/{split($2,a,"."); print "rocm"a[1]"."a[2]; ok=1; exit} END{exit !ok}'; } || { command -v dpkg-query >/dev/null 2>&1 && ver="$(dpkg-query -W -f='${Version}\n' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F'[.-]' '{print "rocm"$1"."$2; exit}' <<<"$ver"; } || { command -v rpm >/dev/null 2>&1 && ver="$(rpm -q --qf '%{VERSION}\n' rocm-core 2>/dev/null)" && [ -n "$ver" ] && awk -F'[.-]' '{print "rocm"$1"."$2; exit}' <<<"$ver"; })"
[ -n "$ROCM_TAG" ] || { echo "Could not detect ROCm. Install ROCm first or set ROCM_TAG manually."; exit 1; }
case "$ROCM_TAG" in
  rocm6.[0-4]|rocm7.[02]) T="$ROCM_TAG" ;;
  rocm6.*) T="rocm6.4" ;;
  *) T="rocm7.1" ;;
esac
pip install bitsandbytes
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${T}"
uv pip install -U --force-reinstall \
    torch torchvision torchaudio triton-rocm \
    --index-url "$PYTORCH_INDEX_URL"
uv pip install cut-cross-entropy torchao --no-deps
uv pip install -U --no-deps "unsloth[amd]==2026.9.14" "unsloth_zoo[amd]==2026.9.9"
uv pip install --no-deps -r "$(python -c 'import pathlib,site;print(next(p for r in [*site.getsitepackages(),site.getusersitepackages()] if (p:=pathlib.Path(r,"studio/backend/requirements/no-torch-runtime.txt")).exists()))')" torchao
uv pip install --no-deps -U "tokenizers>=0.22.0,<0.24.0"  # AMD-CI: was <=0.23.0
# AMD-CI harness: modules the --no-deps installs never pull (a Jupyter image has them)
uv pip install requests pandas pyarrow 'dill<0.4.1' xxhash 'multiprocess<0.70.17' tabulate scikit-learn scipy docstring-parser typeguard msgspec pydantic 'structlog>=24.1.0' importlib-metadata
uv pip install --no-deps diffusers
