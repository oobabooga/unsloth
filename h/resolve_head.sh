# The resolve step of unsloth-prebuilt.yml at whisper.cpp commit 5dd16a95, up to the pairing decision,
# with the workflow expressions replaced by the values of a scheduled run.
set -euo pipefail
UT='latest'
SUFFIX=''
AGE_H=''
[ -n "$AGE_H" ] || AGE_H=6

# The real ABI key: the git tree id of ggml/ in the llama release we
# compile against, published in its manifest. Empty for releases cut
# before llama started recording it.
llama_ggml_tree() {
  gh release download "$1" --repo unslothai/llama.cpp \
    --pattern llama-prebuilt-manifest.json --output - 2>/dev/null \
    | jq -r '.ggml_tree // empty' 2>/dev/null || true
}

# The other half of the pairing. A slim bundle loads the libraries named in
# its requires_ggml_sonames from the llama install, and the installer skips a
# bundle when one of them is not there. llama can rename a runtime library
# without touching ggml (libomp140.<arch>.dll became libomp.dll), so the tree
# comparison alone would call such a release paired forever. Prints one line
# per library that the published manifest ($2) names and the llama release
# ($1) does not ship. An asset that cannot be read is not evidence of a
# mismatch and prints nothing.
llama_missing_sonames() {
  python3 - "$1" "$2" <<'PY'
import io, json, subprocess, sys, tarfile, zipfile
tag, manifest_path = sys.argv[1], sys.argv[2]
assets = {
    "linux": "app-{tag}-linux-{arch}-cpu.tar.gz",
    "windows": "app-{tag}-windows-{arch}-cpu.zip",
    "macos": "llama-{tag}-bin-macos-{arch}.tar.gz",
}
with open(manifest_path, encoding="utf-8") as fh:
    artifacts = json.load(fh).get("artifacts") or []
for art in artifacts:
    names = art.get("requires_ggml_sonames") or []
    pattern = assets.get(art.get("os"))
    if art.get("install_kind") != "slim" or not names or not pattern:
        continue
    asset = pattern.format(tag=tag, arch=art.get("arch"))
    got = subprocess.run(
        ["gh", "release", "download", tag, "--repo", "unslothai/llama.cpp",
         "--pattern", asset, "--output", "-"],
        capture_output=True)
    try:
        if got.returncode != 0 or not got.stdout:
            raise OSError("download failed")
        data = io.BytesIO(got.stdout)
        if asset.endswith(".zip"):
            members = zipfile.ZipFile(data).namelist()
        else:
            members = tarfile.open(fileobj=data, mode="r:gz").getnames()
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"note: could not read {asset} ({exc}); not comparing its libraries",
              file=sys.stderr)
        continue
    # Windows resolves DLL names without regard to case.
    fold = str.lower if art.get("os") == "windows" else str
    shipped = {fold(m.rstrip("/").rsplit("/", 1)[-1]) for m in members}
    for name in names:
        if fold(name) not in shipped:
            print(f"{art.get('os')}-{art.get('arch')}: {name}")
PY
}

# Fallback only. The "-mix-" suffix looks like a ggml commit but is a
# hash of llama's pinned PR set, so it stays constant while the base
# tag and ggml move: 53618c5 alone covers b9909 through b10001. Used
# when a manifest predates ggml_tree, where the alternative is a
# rebuild loop.
ggml_commit() {
  case "$1" in
    *-mix-?*) printf '%s' "${1##*-mix-}" ;;
    *)        printf '%s' "$1" ;;
  esac
}

# Resolve "latest" (the default, and what every scheduled run uses) to
# the newest published upstream release that has been public for >=
# AGE_H hours -- a supply-chain aging window so a yanked or malicious
# upstream tag is not shipped the instant it appears. GitHub does not
# guarantee list order, so pick the max by published_at explicitly. An
# explicit vX.Y.Z tag (manual dispatch) skips the filter.
# Shape is part of the SELECTION, not a later check: upstream's build-number
# releases (b4938) are not prereleases, so filtering on prerelease alone picks
# one and the line below then refuses it, exiting 1 before any build.
if [ "$UT" = "latest" ]; then
  CUTOFF="$(date -u -d "-${AGE_H} hours" +%s)"
  UT="$(gh api 'repos/ggml-org/whisper.cpp/releases?per_page=100' \
    --jq "[.[] | select(.draft==false and .prerelease==false) | select(.tag_name|test(\"^v[0-9]+\\\\.[0-9]+(\\\\.[0-9]+)?$\")) | select((.published_at|fromdateiso8601) <= ${CUTOFF})] | max_by(.published_at|fromdateiso8601) | .tag_name")"
  if [ -z "$UT" ] || [ "$UT" = "null" ]; then
    echo "refusing: no vMAJOR.MINOR[.PATCH] ggml-org/whisper.cpp release older than ${AGE_H}h in the last 100 releases" >&2
    exit 1
  fi
  echo "selected upstream ${UT} (aged >= ${AGE_H}h)"
fi
# Now only reachable for an explicit workflow_dispatch tag, where refusing is right.
printf '%s' "$UT" | grep -qE '^v[0-9]+\.[0-9]+(\.[0-9]+)?$' \
  || { echo "refusing non-release upstream tag '$UT' (want vMAJOR.MINOR[.PATCH])" >&2; exit 1; }

# A blank packaging suffix means the NEWEST published ordinal for this
# upstream tag, not ".1". Only the newest ordinal carries the current
# slim pairing; ".1" predates the slim model and has no
# requires_llama_tag, so the staleness comparison below bails and the
# roll never fires however far llama drifts.
#
# Must run after UT is a concrete tag: resolving it against UT="latest"
# matches no releases and silently falls back to ".1", which is the
# very bug this is here to prevent.
if [ -z "$SUFFIX" ]; then
  NEWEST_ORD="$(gh api "repos/${GITHUB_REPOSITORY}/releases?per_page=100" \
    --jq "[.[] | select(.draft==false) | .tag_name
          | select(startswith(\"${UT}-unsloth.\"))
          | ltrimstr(\"${UT}-unsloth.\")
          | select(test(\"^[0-9]+$\")) | tonumber] | max // empty")"
  [ -n "$NEWEST_ORD" ] || NEWEST_ORD=1
  SUFFIX="unsloth.${NEWEST_ORD}"
  echo "no packaging suffix given; using newest published ordinal ${SUFFIX}"
fi
printf '%s' "$SUFFIX" | grep -qE '^[A-Za-z0-9._-]+$' \
  || { echo "refusing packaging suffix '$SUFFIX'" >&2; exit 1; }
TAG="${UT}-${SUFFIX}"
REPO="$GITHUB_REPOSITORY"

# Slim bundles are compiled against the ggml tree of a PAIRED
# unslothai/llama.cpp release (the llama install provides every ggml
# library at runtime). A blank llama_tag -- including every scheduled
# run -- resolves to the newest PUBLISHED llama release, same style
# as the upstream-tag resolution above.
LT=''
if [ -z "$LT" ]; then
  LT="$(gh api 'repos/unslothai/llama.cpp/releases?per_page=30' \
    --jq '[.[] | select(.draft==false and .prerelease==false)] | max_by(.published_at|fromdateiso8601) | .tag_name')"
  if [ -z "$LT" ] || [ "$LT" = "null" ]; then
    echo "refusing: no published unslothai/llama.cpp release found for slim pairing" >&2
    exit 1
  fi
  echo "selected paired llama release ${LT}"
fi
printf '%s' "$LT" | grep -qE '^[A-Za-z0-9._-]+$' \
  || { echo "refusing llama tag '$LT'" >&2; exit 1; }

# Empty for llama releases cut before ggml_tree was recorded; the
# comparison below falls back to the suffix in that case.
LT_TREE="$(llama_ggml_tree "$LT")"
printf '%s' "$LT_TREE" | grep -qE '^[0-9a-f]{40}$' || LT_TREE=""
echo "paired llama ${LT} ggml tree: ${LT_TREE:-unknown}"

# Dereference the tag to an exact commit (annotated tags resolve to the
# tag object first, then to the commit it points at).
REF_JSON="$(gh api "repos/ggml-org/whisper.cpp/git/ref/tags/${UT}")"
OBJ_TYPE="$(jq -r '.object.type' <<<"$REF_JSON")"
OBJ_SHA="$(jq -r '.object.sha' <<<"$REF_JSON")"
if [ "$OBJ_TYPE" = "tag" ]; then
  COMMIT="$(gh api "repos/ggml-org/whisper.cpp/git/tags/${OBJ_SHA}" --jq '.object.sha')"
else
  COMMIT="$OBJ_SHA"
fi
printf '%s' "$COMMIT" | grep -qE '^[0-9a-f]{40}$' \
  || { echo "could not resolve ${UT} to a commit" >&2; exit 1; }
echo "resolved ${UT} -> ${COMMIT}"

EXISTS=false
PAIRED=false
if [ "$(gh release view "$TAG" --repo "$REPO" --json isDraft --jq .isDraft 2>/dev/null || true)" = "false" ]; then
  EXISTS=true
fi

# Slim bundles only load against a llama install whose ggml ABI they
# were built for, so a published packaging tag goes stale the moment
# unslothai/llama.cpp ships a new ggml commit -- the installer's
# llama_runtime_pairs() then rejects it and Studio setup reports
# "no compatible prebuilt", falling back to Transformers dictation.
# The packaging tag encodes only the upstream whisper version, so
# `exists` alone would make every scheduled run a no-op for as long
# as upstream stays on one tag, however far llama drifts. Compare the
# published release's own pairing against the llama tag resolved
# above and, when it no longer pairs, roll the packaging suffix to the
# next free ordinal so the schedule republishes.
#
# Only schedules auto-roll. A manual dispatch names its packaging
# suffix deliberately (rebuilding one tag, pinning an older llama),
# and silently publishing a different tag than asked for would be a
# surprise. Immutability is preserved either way: an already-published
# tag is never rebuilt, only superseded.
if [ "$EXISTS" = "true" ] && { [ "schedule" = "schedule" ] || [ "schedule" = "repository_dispatch" ]; }; then
  PUBLISHED_MANIFEST="$(gh release download "$TAG" --repo "$REPO" \
    --pattern whisper-prebuilt-manifest.json --output - 2>/dev/null || true)"
  PUBLISHED_LT="$(jq -r 'first(.artifacts[]? | select(.install_kind == "slim") | .requires_llama_tag) // empty' <<<"$PUBLISHED_MANIFEST" 2>/dev/null || true)"
  PUBLISHED_TREE="$(jq -r 'first(.artifacts[]? | select(.install_kind == "slim") | .requires_ggml_tree) // empty' <<<"$PUBLISHED_MANIFEST" 2>/dev/null || true)"
  if [ -z "$PUBLISHED_LT" ]; then
    # Unreadable manifest, or a pre-slim release with no pairing to go
    # stale. Neither is evidence that a rebuild is needed, so stay a
    # no-op rather than republish on a guess.
    echo "note: ${TAG} publishes no slim pairing to compare; leaving exists=true"
  elif [ -n "$PUBLISHED_TREE" ] && [ -n "$LT_TREE" ]; then
    # Both sides know their ggml tree: compare it, then the libraries the
    # published bundles load from the llama install.
    if [ "$PUBLISHED_TREE" = "$LT_TREE" ]; then
      printf '%s' "$PUBLISHED_MANIFEST" > "$RUNNER_TEMP/published-manifest.json"
      MISSING="$(llama_missing_sonames "$LT" "$RUNNER_TEMP/published-manifest.json")"
      if [ -z "$MISSING" ]; then
        echo "${TAG} pairs with ${LT} (ggml tree ${LT_TREE}); nothing to do"
        PAIRED=true
      else
        echo "${TAG} has the ggml tree of ${LT} but names libraries that release does not ship:"
        printf '%s\n' "$MISSING"
      fi
    fi
  elif [ "$(ggml_commit "$PUBLISHED_LT")" = "$(ggml_commit "$LT")" ]; then
    # One side predates ggml_tree. Approximate, and say so.
    echo "::warning::${TAG} compared against ${LT} on the -mix- suffix, which does not track ggml; treating as paired"
    PAIRED=true
  fi
  if [ "$PAIRED" != "true" ] && [ -n "$PUBLISHED_LT" ]; then
    PREFIX="${SUFFIX%.*}"
    ORDINAL="${SUFFIX##*.}"
    if [ "$PREFIX" = "$SUFFIX" ] || ! printf '%s' "$ORDINAL" | grep -qE '^[0-9]+$'; then
      echo "note: packaging suffix '${SUFFIX}' is not <prefix>.<N>; cannot roll it automatically" >&2
    else
      # Highest published ordinal + 1, not $ORDINAL + 1: a manual
      # dispatch may already have taken the next few, and reusing a
      # published tag would break immutability.
      NEXT="$(gh api "repos/${REPO}/releases?per_page=100" \
        --jq "[.[] | select(.draft==false) | .tag_name
              | select(type == \"string\" and startswith(\"${UT}-${PREFIX}.\"))
              | ltrimstr(\"${UT}-${PREFIX}.\")
              | select(test(\"^[0-9]+$\")) | tonumber] | (max // 0) + 1")"
      echo "${TAG} (built against llama ${PUBLISHED_LT}) does not pair with ${LT};" \
           "rebuilding as ${UT}-${PREFIX}.${NEXT}"
      SUFFIX="${PREFIX}.${NEXT}"
      TAG="${UT}-${SUFFIX}"
      EXISTS=false
    fi
  fi
fi


echo "RESULT TAG=$TAG EXISTS=$EXISTS PAIRED=$PAIRED LT=$LT"
