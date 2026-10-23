#!/usr/bin/env bash
# usage: run_lists.sh <branch> <src-dir> <out-dir> ; runs the per-branch test lists, writes junit + summary
set -uo pipefail
BR=$1; SRC=$2; OUT=$3; mkdir -p "$OUT"
EX_EFFCFG="studio/backend/tests/test_openai_auto_switch.py studio/backend/tests/test_llama_compat_routes.py unsloth_cli/tests/test_studio_run_parallel_flag.py unsloth_cli/tests/test_inference_chat.py"
EX_MEM="studio/backend/tests/test_gpu_arch_gate_7624.py studio/backend/tests/test_gpu_arch_gate_os_matrix_7624.py studio/backend/tests/test_rocm_oom_guard.py"
EX_MI="studio/backend/tests/test_imatrix_gguf_filtering.py studio/backend/tests/test_mtp_drafter_companion.py studio/backend/tests/test_refactor_guard.py tests/studio/test_model_picker_contracts.py"
EX_TF="studio/backend/tests/test_presence_penalty.py studio/backend/tests/test_generation_timing.py"
EX_MG="studio/backend/tests/test_managed_engines.py studio/backend/tests/test_managed_engine_reasoning.py studio/backend/tests/test_npu_chat_route.py"
EX_SF="studio/backend/tests/test_llama_cpp_start_failure_classification.py"
EX_STT="studio/backend/tests/test_stt_mtmd_sidecar.py studio/backend/tests/test_stt_ggml_request_path.py"
EX_IMG="studio/backend/tests/test_sd_cpp_backend.py studio/backend/tests/test_sd_cpp_engine.py"
case "$BR" in
  base) FILES="$EX_EFFCFG $EX_TF $EX_MG studio/backend/tests/test_research_json_fallback.py studio/backend/tests/test_mlx_context_compaction.py";;
  effcfg) FILES="$EX_EFFCFG studio/backend/tests/test_load_applies_saved_settings.py studio/backend/tests/test_model_override_field_parity.py";;
  chat-capabilities) FILES="$EX_TF $EX_MG studio/backend/tests/test_transformers_sampling_controls.py studio/backend/tests/test_chat_controls.py studio/backend/tests/test_research_json_fallback.py studio/backend/tests/test_mlx_context_compaction.py";;
  mem-capacity) FILES="$EX_MEM studio/backend/tests/test_rocm_apu_one_classifier.py";;
  model-identity) FILES="$EX_MI studio/backend/tests/test_gguf_names_single_owner.py studio/backend/tests/test_openai_auto_switch.py";;
  tf-sampling-controls) FILES="$EX_TF studio/backend/tests/test_transformers_sampling_controls.py";;
  managed-engine-param) FILES="$EX_MG";;
  start-failure-messages) FILES="$EX_SF studio/backend/tests/test_llama_start_failure_names_template_and_build.py";;
  stt-sidecar-logs) FILES="$EX_STT studio/backend/tests/test_stt_sidecar_start_failure_log.py";;
  image-engine-failure) FILES="$EX_IMG studio/backend/tests/test_sd_cpp_probe_failure_recorded.py";;
esac
BACKEND=""; CLI=""; REPO=""
for f in $FILES; do
  [ -e "$SRC/$f" ] || continue
  case "$f" in
    studio/backend/*) BACKEND="$BACKEND ${f#studio/backend/}";;
    unsloth_cli/*) CLI="$CLI $f";;
    *) REPO="$REPO $f";;
  esac
done
rc=0
if [ -n "$BACKEND" ]; then (cd "$SRC/studio/backend" && python -m pytest -q -p no:cacheprovider --timeout=600 -rfE $BACKEND --junitxml="$OUT/backend.xml") > "$OUT/backend.log" 2>&1 || rc=1; fi
if [ -n "$CLI" ]; then (cd "$SRC" && python -m pytest -q -p no:cacheprovider --timeout=600 -rfE $CLI --junitxml="$OUT/cli.xml") > "$OUT/cli.log" 2>&1 || rc=1; fi
if [ -n "$REPO" ]; then (cd "$SRC" && python -m pytest -q -p no:cacheprovider --timeout=600 -rfE $REPO --junitxml="$OUT/repo.xml") > "$OUT/repo.log" 2>&1 || rc=1; fi
for l in "$OUT"/*.log; do echo "== $(basename $l): $(grep -E '(passed|failed|error).* in [0-9.]+s|^# (pass|fail)' $l | tr '\n' ' ')"; grep -E '^(FAILED|ERROR) ' $l | sed 's/ - .*//' | head -40; done
exit 0
