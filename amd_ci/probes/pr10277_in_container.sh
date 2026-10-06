#!/usr/bin/env bash
# Runs INSIDE rocm/dev-ubuntu-22.04:<6.x>. Real install.sh --local, then measure.
# Writes /out/result.json, /out/install.log, /out/measure.log. Never exits early:
# every outcome is data for the criteria module.
set +e
export DEBIAN_FRONTEND=noninteractive
R=/out/result_env.json
: > /out/container.log

{
  apt-get update -qq
  apt-get install -y -qq git curl ca-certificates build-essential cmake pciutils python3 >/dev/null
} >> /out/container.log 2>&1
git config --global --add safe.directory '*'

ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null | head -1)
GFX_SEEN=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | sort -u | tr '\n' ' ')
KFD_GFX=$(cat /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | awk '/gfx_target_version/ && $2 != 0 {print $2}' | sort -u | tr '\n' ' ')

cp -a /src /w_src
cd /w_src || exit 0
export UNSLOTH_STUDIO_HOME=/w_studio UNSLOTH_SKIP_AUTOSTART=1
timeout "${INSTALL_TIMEOUT:-5400}" bash install.sh --local > /out/install.log 2>&1
INSTALL_RC=$?

V=""
for c in "$UNSLOTH_STUDIO_HOME"/unsloth_studio "$UNSLOTH_STUDIO_HOME"/.venv_t5_*; do
  [ -x "$c/bin/python" ] && { V="$c"; break; }
done
INDEX_LINE=$(grep -oE 'download\.pytorch\.org/whl/[a-z0-9.]+|repo\.amd\.com[^ ]*|torch_index_family=[^ ]+' /out/install.log | sort -u | tr '\n' ' ')

python3 - "$R" <<PY
import json, sys
json.dump({"container_rocm_version": "$ROCM_VER", "rocminfo_gfx": "$GFX_SEEN".split(),
           "kfd_gfx_target_versions": "$KFD_GFX".split(), "install_rc": $INSTALL_RC,
           "venv": "$V", "install_log_index_mentions": "$INDEX_LINE".split()},
          open(sys.argv[1], "w"), indent=2)
PY

if [ -n "$V" ]; then
  "$V/bin/python" /probe/pr10277_measure.py --env "$R" --out /out/result.json > /out/measure.log 2>&1
else
  cp "$R" /out/result.json
fi
tail -40 /out/install.log > /out/install_tail.log
chown -R "${HOST_UID:-0}:${HOST_GID:-0}" /out 2>/dev/null
exit 0
