#!/usr/bin/env bash
# ab_one.sh <title> <hf repo> <hf revision> <file path in repo> <slug> [extra llama-server args...]
set +e
TITLE="$1"; REPO="$2"; REV="$3"; FILE="$4"; SLUG="$5"; shift 5
EXTRA=("$@")

W="${W:-$RUNNER_TEMP/uma}"
BIN="$W/bin"
MODELS="$W/models/$SLUG"
OUT="$W/out/$SLUG"
mkdir -p "$MODELS" "$OUT"

BASENAME="$(basename "$FILE")"
TARGET="$MODELS/$BASENAME"

if [ ! -s "$TARGET" ]; then
  echo "downloading $REPO/$FILE"
  curl -fL --retry 3 --retry-delay 5 -o "$TARGET" \
    "https://huggingface.co/$REPO/resolve/$REV/$FILE" || {
      echo "## $TITLE" >> "$GITHUB_STEP_SUMMARY"
      echo >> "$GITHUB_STEP_SUMMARY"
      echo "download failed" >> "$GITHUB_STEP_SUMMARY"
      exit 0
    }
fi
ls -l "$TARGET"

ARGS=(--bin-dir "$BIN" --gguf "$TARGET" --ngl 999 --ctx-size 8192
      --n-predict 64 --repeats 2 --load-timeout 2400 --request-timeout 1200)
for e in "${EXTRA[@]}"; do ARGS+=(--extra "$e"); done

python3 tools/llama_ab.py "${ARGS[@]}" \
  --label "UMA absent" --env "-GGML_CUDA_ENABLE_UNIFIED_MEMORY" \
  --out "$OUT/base.json"

python3 tools/llama_ab.py "${ARGS[@]}" \
  --label "UMA=1" --env "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1" \
  --out "$OUT/head.json"

python3 tools/compare.py "$TITLE" "$OUT/base.json" "$OUT/head.json" \
  | tee "$OUT/VERDICT.md" >> "$GITHUB_STEP_SUMMARY"
echo >> "$GITHUB_STEP_SUMMARY"
