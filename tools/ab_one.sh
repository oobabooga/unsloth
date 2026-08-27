#!/usr/bin/env bash
# ab_one.sh <title> <slug> <hf repo> <hf revision> <file in repo> [extra llama-server args...]
#
# Downloads the GGUF once and runs llama-server twice over it, changing only
# GGML_CUDA_ENABLE_UNIFIED_MEMORY. Every option is passed in --opt=value form:
# argparse reads a dash-leading value in the two-token form as another option,
# which is how the first attempt at this silently compared nothing.
set +e
TITLE="$1"; SLUG="$2"; REPO="$3"; REV="$4"; FILE="$5"; shift 5

W="${W:-$RUNNER_TEMP/uma}"
BIN="$W/bin"
OUT="$W/out/$SLUG"
MODELS="$W/models"
mkdir -p "$OUT" "$MODELS"

CTX="${CTX:-8192}"
NPREDICT="${NPREDICT:-64}"
REPEATS="${REPEATS:-2}"

BASENAME="$(basename "$FILE")"
TARGET="$MODELS/$BASENAME"

report() {
  { echo "### $TITLE"; echo; echo "$1"; echo; } >> "$GITHUB_STEP_SUMMARY"
}

if [ ! -s "$TARGET" ]; then
  echo "downloading $REPO/$FILE"
  curl -fL --retry 5 --retry-delay 10 -o "$TARGET" \
    "https://huggingface.co/$REPO/resolve/$REV/$FILE" || {
      report "model download failed"
      exit 0
    }
fi
ls -l "$TARGET"
free -h

ARGS=("--bin-dir=$BIN" "--gguf=$TARGET" "--ngl=999" "--ctx-size=$CTX"
      "--n-predict=$NPREDICT" "--repeats=$REPEATS"
      "--load-timeout=1800" "--request-timeout=1200")
for e in "$@"; do ARGS+=("--extra=$e"); done

python3 tools/llama_ab.py "${ARGS[@]}" \
  "--label=UMA absent" "--env=unset:GGML_CUDA_ENABLE_UNIFIED_MEMORY" \
  "--out=$OUT/base.json"
free -h

python3 tools/llama_ab.py "${ARGS[@]}" \
  "--label=UMA=1" "--env=GGML_CUDA_ENABLE_UNIFIED_MEMORY=1" \
  "--out=$OUT/head.json"
free -h

python3 tools/compare.py "$TITLE" "$OUT/base.json" "$OUT/head.json" > "$OUT/VERDICT.md" 2>&1
cat "$OUT/VERDICT.md"
cat "$OUT/VERDICT.md" >> "$GITHUB_STEP_SUMMARY"
echo >> "$GITHUB_STEP_SUMMARY"
