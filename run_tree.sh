#!/bin/bash
# usage: run_tree.sh <name> <ref> <kinds>
#   name: label for the outputs; ref: "main" or a fetched ref/SHA merged onto MAIN_SHA
#   kinds: any of "be" (backend suite), "repo" (tests/studio + tests/python), "inst" (tests/studio/install), "fe" (frontend)
# Needs W, MAIN_SHA in the environment; writes $W/out/<name>-<suite>.{log,ids}.
set -uo pipefail
NAME=$1; REF=$2; KINDS=$3
T="$W/trees/$NAME"; OUT="$W/out"; mkdir -p "$OUT" "$W/trees"
cd "$W/src"
git worktree add -q --detach "$T" "$MAIN_SHA"
if [ "$REF" = patch ]; then
  # The combined tree of every sub-PR: main plus the union diff shipped with the harness.
  (cd "$T" && git apply --index "$GITHUB_WORKSPACE/harness/stack.patch") || { echo "$NAME: PATCH DID NOT APPLY" | tee "$OUT/$NAME-merge.txt"; exit 0; }
elif [ "$REF" != main ]; then
  (cd "$T" && git -c user.name=ci -c user.email=ci@example.invalid merge -q --no-edit "$REF") || { echo "$NAME: MERGE CONFLICT with main" | tee "$OUT/$NAME-merge.txt"; exit 0; }
fi
echo "== $NAME: $(cd "$T" && git log -1 --format='%h %s' | cut -c1-90)"
ids() { grep -aE '^(FAILED|ERROR) ' "$1" | sed -E 's/^(FAILED|ERROR) //; s/ - .*//' | cut -c1-200 | sort -u; }
IGN="--ignore=tests/test_studio_api.py --ignore=tests/test_streaming_stripper.py --ignore=tests/test_llama_cpp_wait_for_vram_settle.py --ignore=tests/test_tool_xml_strip.py --ignore=tests/test_diffusion_checkpoint_resume.py --ignore=tests/test_tool_output_streaming.py --ignore=tests/test_web_fetch_extraction.py --ignore=tests/test_tool_call_parser_strict.py --ignore=tests/test_tunnel_safe_long_post.py --ignore=tests/test_scan_loras_off_event_loop.py --ignore=tests/test_anthropic_messages.py --ignore=tests/test_profile_stats.py --ignore=tests/test_media_auto_switch.py --ignore=tests/test_pr5624_regressions.py --ignore=tests/test_web_tools_overlap.py"
export HOME="$W/home-$NAME"; mkdir -p "$HOME"
for k in $KINDS; do
  BT="$W/bt-$NAME-$k"; mkdir -p "$BT"
  case $k in
    be)
      (cd "$T/studio/backend" && timeout 5400 "$W/venv/bin/python" -m pytest tests/ -q -rfE --tb=short -n 16 --dist loadgroup --timeout=330 --basetemp="$BT" -p no:cacheprovider $IGN -k 'not llama_cpp_load_progress_live' > "$OUT/$NAME-be.log" 2>&1; echo "rc=$?" >> "$OUT/$NAME-be.log")
      ids "$OUT/$NAME-be.log" > "$OUT/$NAME-be.ids"; echo "   be: $(grep -aE ' passed| failed' "$OUT/$NAME-be.log" | tail -1 | cut -c1-120) $(tail -1 "$OUT/$NAME-be.log")" ;;
    repo)
      (cd "$T" && PYTHONPATH="$T/studio:$T" UNSLOTH_COMPILE_DISABLE=1 timeout 5400 "$W/venv/bin/python" -m pytest tests/studio tests/python --ignore=tests/studio/install --ignore=tests/studio/studiobench --ignore=tests/studio/load_freeze --ignore=tests/studio/test_hardware_dispatch_matrix.py --ignore=tests/studio/test_is_mlx_dispatch_gate.py --ignore=tests/studio/test_xpu_spoof_pipeline.py --ignore=tests/studio/test_mlx_context_platform_matrix.py -q -rfE --tb=short -n 16 --dist loadgroup --timeout=330 --basetemp="$BT" -p no:cacheprovider -m 'not server and not e2e' > "$OUT/$NAME-repo.log" 2>&1; echo "rc=$?" >> "$OUT/$NAME-repo.log")
      ids "$OUT/$NAME-repo.log" > "$OUT/$NAME-repo.ids"; echo "   repo: $(grep -aE ' passed| failed' "$OUT/$NAME-repo.log" | tail -1 | cut -c1-120) $(tail -1 "$OUT/$NAME-repo.log")" ;;
    inst)
      (cd "$T" && PYTHONPATH="$T/studio:$T" timeout 5400 "$W/venv/bin/python" -m pytest tests/studio/install -q -rfE --tb=short -n 8 --timeout=330 --basetemp="$BT" -p no:cacheprovider -m 'not server and not e2e' > "$OUT/$NAME-inst.log" 2>&1; echo "rc=$?" >> "$OUT/$NAME-inst.log")
      ids "$OUT/$NAME-inst.log" > "$OUT/$NAME-inst.ids"; echo "   inst: $(grep -aE ' passed| failed' "$OUT/$NAME-inst.log" | tail -1 | cut -c1-120) $(tail -1 "$OUT/$NAME-inst.log")" ;;
    fe)
      # A real copy per tree: Vite refuses files reached through a symlink outside the project root.
      cp -a "$W/node_modules" "$T/studio/frontend/node_modules"
      (cd "$T/studio/frontend" && export PATH="$W/node/bin:$PATH" && { npm run -s i18n:check:strict > "$OUT/$NAME-i18n.log" 2>&1; echo "rc=$?" >> "$OUT/$NAME-i18n.log"; npm run -s typecheck > "$OUT/$NAME-tc.log" 2>&1; echo "rc=$?" >> "$OUT/$NAME-tc.log"; timeout 2400 npm test -s > "$OUT/$NAME-fe.log" 2>&1; echo "rc=$?" >> "$OUT/$NAME-fe.log"; })
      grep -aE '^not ok [0-9]+ - ' "$OUT/$NAME-fe.log" | sed 's/^not ok [0-9]* - //' | sort -u > "$OUT/$NAME-fe.ids"
      echo "   fe: i18n $(tail -1 "$OUT/$NAME-i18n.log") | typecheck $(tail -1 "$OUT/$NAME-tc.log") | $(grep -aE '^# (tests|pass|fail) ' "$OUT/$NAME-fe.log" | tr '\n' ' ')"
      rm -rf "$T/studio/frontend/node_modules" ;;
  esac
  rm -rf "$BT"
done
# Keep the logs small: only the failure sections travel in the artifact.
for f in "$OUT/$NAME"-*.log; do
  case "$f" in *-fe.log) grep -aE -A25 '^not ok [0-9]+ - ' "$f" | head -400 > "$f.trim";; *) tail -150 "$f" > "$f.trim";; esac
  mv "$f.trim" "$f"
done
cd "$W/src" && git worktree remove --force "$T"
