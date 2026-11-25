#!/usr/bin/env bash
# AITER checkouts for the a12 probe (items 7 conv3d, 6 rdna_attn). Never fails the job: every step is bounded and
# its outcome lands in $A12_WORK/aiter_setup.json (written by python from env vars, never parsed from stdout).
#
#   AMD_CI_PY=<studio venv python> A12_WORK=<dir> bash aiter_setup.sh
#
# $A12_WORK/aiter       ROCm/aiter main + the gfx1151 conv3d configs of PR #6973 (Triton only, no HIP build:
#                       the scripts import aiter.ops.triton.conv.conv3d through stub packages)
# $A12_WORK/aiter_rdna  ROCm/aiter PR #8605 head (rdna_attn), JIT-built against a hipcc that matches the venv's ROCm
# $A12_WORK/pydeps      pybind11 + ninja (pip --target; the Studio venv is not modified for these)
# The venv gains rocm-sdk-devel only if no toolchain is present (same version as its rocm-sdk-core; torch untouched).
set +e
: "${AMD_CI_PY:?}" "${A12_WORK:?}"
mkdir -p "$A12_WORK"
LOG="$A12_WORK/aiter_setup.log"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export S_CONV_OK=0 S_RDNA_SRC_OK=0 S_HIPCC="" S_ROCM_HOME="" S_DEVEL_INSTALL="" S_BUILD_RC="" S_ERRS=""
note() { echo "[aiter_setup] $*" | tee -a "$LOG"; S_ERRS="$S_ERRS$*; "; }
T0=$(date +%s)
unset GH_TOKEN GITHUB_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN

# 1. aiter main + PR 6973 configs (overlay only the gfx1151 conv3d config tree: the PR branch is older than main)
if timeout 600 git clone -q --depth 1 https://github.com/ROCm/aiter "$A12_WORK/aiter" >>"$LOG" 2>&1; then
  export S_MAIN_SHA=$(git -C "$A12_WORK/aiter" rev-parse HEAD)
  if timeout 300 git -C "$A12_WORK/aiter" fetch -q --depth 1 origin pull/6973/head:pr6973 >>"$LOG" 2>&1; then
    export S_PR6973_SHA=$(git -C "$A12_WORK/aiter" rev-parse pr6973)
    git -C "$A12_WORK/aiter" checkout pr6973 -- aiter/ops/triton/configs/gfx1151/triton/conv >>"$LOG" 2>&1 \
      || note "pr6973 config overlay failed"
  else
    note "fetch pull/6973 failed"
  fi
  [ -f "$A12_WORK/aiter/aiter/ops/triton/conv/conv3d.py" ] && S_CONV_OK=1
  export S_GFX1151_CONV3D_CFGS=$(ls "$A12_WORK/aiter/aiter/ops/triton/configs/gfx1151/triton/conv" 2>/dev/null | grep -c conv3d)
else
  note "clone aiter failed"
fi

# 2. PR 8605 head in its own checkout
mkdir -p "$A12_WORK/aiter_rdna"
( cd "$A12_WORK/aiter_rdna" && git init -q && git remote add origin https://github.com/ROCm/aiter \
  && timeout 600 git fetch -q --depth 1 origin pull/8605/head && git checkout -q FETCH_HEAD ) >>"$LOG" 2>&1
if [ -f "$A12_WORK/aiter_rdna/aiter/ops/rdna_attn.py" ]; then
  S_RDNA_SRC_OK=1; export S_PR8605_SHA=$(git -C "$A12_WORK/aiter_rdna" rev-parse HEAD)
else
  note "PR 8605 checkout failed"
fi

# 3. build deps outside the venv
timeout 600 "$AMD_CI_PY" -m pip install -q --target "$A12_WORK/pydeps" pybind11 ninja >>"$LOG" 2>&1 \
  || timeout 600 uv pip install -q --target "$A12_WORK/pydeps" pybind11 ninja >>"$LOG" 2>&1 \
  || note "pybind11/ninja install failed"
export PATH="$A12_WORK/pydeps/bin:$PATH" PYTHONPATH="$A12_WORK/pydeps${PYTHONPATH:+:$PYTHONPATH}"

# 4. a hipcc matching the venv's ROCm: _rocm_sdk_devel in the venv, else install it, else /opt/rocm (mismatch risk)
find_devel() {
  "$AMD_CI_PY" - <<'EOF' 2>>"$LOG"
import importlib.util, os
s = importlib.util.find_spec("_rocm_sdk_devel")
if s and s.submodule_search_locations:
    root = list(s.submodule_search_locations)[0]
    for c in ("bin/hipcc", "lib/llvm/bin/clang"):
        if os.path.exists(os.path.join(root, c)):
            print(root)
            break
EOF
}
ROOT=$(find_devel)
if [ -z "$ROOT" ]; then
  VER=$("$AMD_CI_PY" -c "import importlib.metadata as m; print(m.version('rocm-sdk-core'))" 2>>"$LOG")
  export S_SDK_CORE_VER="$VER"
  if [ -n "$VER" ]; then
    IDX="--extra-index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --extra-index-url https://repo.amd.com/rocm/whl/gfx1151/ --extra-index-url https://rocm.nightlies.amd.com/whl-multi-arch/"
    for inst in "uv pip install --python $AMD_CI_PY" "$AMD_CI_PY -m pip install"; do
      if timeout 1500 $inst --no-deps $IDX "rocm-sdk-devel==$VER" >>"$LOG" 2>&1; then
        S_DEVEL_INSTALL="ok:$inst"; break
      fi
    done
    [ -n "$S_DEVEL_INSTALL" ] || { S_DEVEL_INSTALL="failed"; note "rocm-sdk-devel==$VER install failed"; }
    timeout 900 "$AMD_CI_PY" -m rocm_sdk init >>"$LOG" 2>&1 \
      || timeout 900 "$(dirname "$AMD_CI_PY")/rocm-sdk" init >>"$LOG" 2>&1 || note "rocm-sdk init failed"
    ROOT=$(find_devel)
  else
    note "no rocm-sdk-core in the venv"
  fi
fi
if [ -n "$ROOT" ]; then
  S_ROCM_HOME="$ROOT"
elif [ -x /opt/rocm/bin/hipcc ]; then
  S_ROCM_HOME=/opt/rocm; note "using /opt/rocm (may not match torch's HIP runtime)"
fi
[ -n "$S_ROCM_HOME" ] && [ -x "$S_ROCM_HOME/bin/hipcc" ] && S_HIPCC="$S_ROCM_HOME/bin/hipcc"
export S_ROCM_HOME S_HIPCC S_DEVEL_INSTALL
export S_ROCM_VERSION_FILE=$(cat "$S_ROCM_HOME/.info/version" 2>/dev/null | head -1)

# 5. warm JIT build of module_rdna_attn + the d128 bf16 unit (one tiny encoder call); the timed race reuses it
if [ "$S_RDNA_SRC_OK" = 1 ] && [ -n "$S_HIPCC" ]; then
  TB=$(date +%s)
  ROCM_HOME="$S_ROCM_HOME" ROCM_PATH="$S_ROCM_HOME" GPU_ARCHS=gfx1151 PATH="$S_ROCM_HOME/bin:$PATH" \
    timeout 1800 "$AMD_CI_PY" "$HERE/aiter_rdna.py" --aiter-dir "$A12_WORK/aiter_rdna" --arm rdna --shape warm \
    --out "$A12_WORK/aiter_rdna_warm.json" >>"$LOG" 2>&1
  S_BUILD_RC=$?; export S_BUILD_S=$(( $(date +%s) - TB ))
else
  S_BUILD_RC="skipped"; note "rdna build skipped (no source or no hipcc)"
fi
export S_CONV_OK S_RDNA_SRC_OK S_BUILD_RC S_ERRS S_TOTAL_S=$(( $(date +%s) - T0 ))

"$AMD_CI_PY" - "$A12_WORK/aiter_setup.json" <<'EOF'
import json, os, sys
e = os.environ
warm = os.path.join(e["A12_WORK"], "aiter_rdna_warm.json")
try:
    warm_obs = json.load(open(warm, encoding = "utf-8"))
except Exception:  # noqa: BLE001
    warm_obs = None
obs = {"conv_ok": e.get("S_CONV_OK") == "1", "aiter_main_sha": e.get("S_MAIN_SHA"),
       "pr6973_sha": e.get("S_PR6973_SHA"), "gfx1151_conv3d_cfg_dirs": e.get("S_GFX1151_CONV3D_CFGS"),
       "rdna_src_ok": e.get("S_RDNA_SRC_OK") == "1", "pr8605_sha": e.get("S_PR8605_SHA"),
       "rocm_home": e.get("S_ROCM_HOME") or None, "hipcc": e.get("S_HIPCC") or None,
       "rocm_version_file": e.get("S_ROCM_VERSION_FILE"), "sdk_core_version": e.get("S_SDK_CORE_VER"),
       "devel_install": e.get("S_DEVEL_INSTALL"), "rdna_build_rc": e.get("S_BUILD_RC"),
       "rdna_build_s": e.get("S_BUILD_S"), "rdna_warm": warm_obs, "errors": e.get("S_ERRS"),
       "total_s": e.get("S_TOTAL_S"), "pydeps": os.path.join(e["A12_WORK"], "pydeps")}
open(sys.argv[1], "w", encoding = "utf-8").write(json.dumps(obs, indent = 1))
EOF
exit 0
