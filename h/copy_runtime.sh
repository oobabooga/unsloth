# Verbatim 'Copy the CUDA runtime into the bundle' step; matrix.cuda comes from $CUDA_INPUT
set -euo pipefail
major="${CUDA_INPUT}"
major="${major%%.*}"
for lib in libcudart libcublas libcublasLt; do
  src="$CUDA_PATH/lib/$lib.so.$major"
  [ -e "$src" ] || { echo "the toolkit has no $src" >&2; exit 1; }
  cp -L "$src" "src/build/bin/$lib.so.$major"
done
ls -l src/build/bin/libcu*
