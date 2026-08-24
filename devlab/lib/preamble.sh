#!/usr/bin/env bash
# Isolation preamble for every DevLab job. Source this first, from a step whose
# only job is setup.
#
# The rules encoded here were each learned by losing a run to them:
#
#   set +e        GitHub runs step shells as `bash -e`, so the first non-zero
#                 exit aborts the step. In a job whose subject is "does this
#                 command fail", that throws away the measurement.
#   sudo shim     The host is persistent and shared. Nothing here needs root,
#                 and a shim makes that a checkable property rather than a
#                 claim. Every attempt is logged.
#   $RUNNER_TEMP  The only tree the runner reclaims between jobs. $HOME and /tmp
#                 are the small root overlay; /mnt/scratch is not writable.
#
# Usage from a workflow step:
#   source devlab/lib/preamble.sh my-job-name
# It exports DEVLAB_WORK and appends WORK/TMPDIR/PATH to $GITHUB_ENV.

set +e
set -uo pipefail

_devlab_name="${1:-devlab}"
DEVLAB_WORK="${RUNNER_TEMP:-/tmp}/${_devlab_name}"
mkdir -p "$DEVLAB_WORK"/{bin,out,src,tmp}

cat > "$DEVLAB_WORK/bin/sudo" <<'SH'
#!/bin/sh
echo "refused privilege escalation: $*" >&2
echo "$(date -Is) $*" >> "${DEVLAB_WORK:-/tmp}/out/escalation-attempts.log" 2>/dev/null
exit 1
SH
chmod +x "$DEVLAB_WORK/bin/sudo"

export DEVLAB_WORK
export TMPDIR="$DEVLAB_WORK/tmp"
export PATH="$DEVLAB_WORK/bin:$PATH"

if [ -n "${GITHUB_ENV:-}" ]; then
  {
    echo "DEVLAB_WORK=$DEVLAB_WORK"
    echo "TMPDIR=$DEVLAB_WORK/tmp"
    echo "PATH=$DEVLAB_WORK/bin:$PATH"
  } >> "$GITHUB_ENV"
fi

echo "devlab work root: $DEVLAB_WORK"
df -h "${RUNNER_TEMP:-/tmp}" | tail -1
