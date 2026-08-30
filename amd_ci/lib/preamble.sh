#!/usr/bin/env bash
# Isolation preamble for every AMD CI job. Source this first, from a step whose
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
#   source amd_ci/lib/preamble.sh my-job-name
# It exports AMD_CI_WORK and appends WORK/TMPDIR/PATH to $GITHUB_ENV.

set +e
set -uo pipefail

_amd_ci_name="${1:-amd_ci}"
AMD_CI_WORK="${RUNNER_TEMP:-/tmp}/${_amd_ci_name}"
mkdir -p "$AMD_CI_WORK"/{bin,out,src,tmp}

cat > "$AMD_CI_WORK/bin/sudo" <<'SH'
#!/bin/sh
echo "refused privilege escalation: $*" >&2
echo "$(date -Is) $*" >> "${AMD_CI_WORK:-/tmp}/out/escalation-attempts.log" 2>/dev/null
exit 1
SH
chmod +x "$AMD_CI_WORK/bin/sudo"

export AMD_CI_WORK
export TMPDIR="$AMD_CI_WORK/tmp"
export PATH="$AMD_CI_WORK/bin:$PATH"

if [ -n "${GITHUB_ENV:-}" ]; then
  {
    echo "AMD_CI_WORK=$AMD_CI_WORK"
    echo "TMPDIR=$AMD_CI_WORK/tmp"
    echo "PATH=$AMD_CI_WORK/bin:$PATH"
  } >> "$GITHUB_ENV"
fi

echo "amd_ci work root: $AMD_CI_WORK"
df -h "${RUNNER_TEMP:-/tmp}" | tail -1
