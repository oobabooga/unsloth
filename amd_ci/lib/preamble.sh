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
#
# LINUX ONLY. The Windows AMD runners have no bash at all and their counterpart
# is lib/preamble.ps1, run under `shell: powershell`. See the guard below: this
# file refuses rather than half-working, because a preamble that quietly puts the
# work root somewhere the job cannot see is the exact silent failure the toolkit
# exists to prevent.

set +e
set -uo pipefail

# Refuse the two shapes that would "succeed" on a Windows AMD runner.
#
#   Git Bash / MSYS   $RUNNER_TEMP arrives as C:\... and the POSIX paths below
#                     land in the MSYS root, not where the job looks.
#   WSL               one of the four boxes has a WindowsApps bash.exe that is
#                     the WSL stub. Inside WSL the filesystem, the GPU and
#                     $GITHUB_ENV are all a different machine's.
_amd_ci_uname="$(uname -s 2>/dev/null || echo unknown)"
case "$_amd_ci_uname" in
  MINGW*|MSYS*|CYGWIN*)
    echo "FATAL: preamble.sh was sourced under $_amd_ci_uname (a Windows bash)." >&2
    echo "       The AMD Windows runners take lib/preamble.ps1 with 'shell: powershell'." >&2
    echo "       Refusing rather than writing the work root somewhere the job cannot see." >&2
    exit 1
    ;;
esac
if [ -n "${GITHUB_ENV:-}" ] && grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
  echo "FATAL: this is WSL, reached from a Windows job (bash.exe is the WSL stub on" >&2
  echo "       one of the AMD Windows runners). WSL sees a different filesystem and a" >&2
  echo "       different GPU than the job that launched it, so nothing measured here" >&2
  echo "       would describe the runner. Use lib/preamble.ps1 with 'shell: powershell'." >&2
  exit 1
fi

_amd_ci_name="${1:-amd_ci}"
if [ -z "${RUNNER_TEMP:-}" ] && [ -n "${GITHUB_ENV:-}" ]; then
  echo "WARNING: RUNNER_TEMP is unset inside a GitHub job; falling back to /tmp, which" >&2
  echo "         is the small root overlay and is NOT reclaimed between jobs." >&2
fi
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
