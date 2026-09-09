<#
Isolation preamble for the WINDOWS AMD CI jobs. The counterpart of lib/preamble.sh,
which refuses to run here on purpose; this file is what `shell: powershell` steps call.

It exists because the Windows template referenced it and the toolkit never shipped it,
so every Windows job died on its first step with CommandNotFoundException.

What it guarantees, matching the bash version:

  $ErrorActionPreference   "Continue", so a non-zero exit inside a step does not abort
                           the step. In a job whose subject is "does this command fail",
                           aborting throws away the measurement.
  $env:RUNNER_TEMP         the only tree the runner reclaims between jobs.
  AMD_CI_WORK              exported AND appended to $GITHUB_ENV, with bin/out/src/tmp,
                           so later steps in the same job see the same work root.

There is deliberately NO sudo shim: Windows has no sudo, and a stub named `sudo.cmd`
would assert a protection that nothing on this host can bypass anyway. The escalation
question is a Linux one.

Usage from a workflow step (note the leading dot is NOT needed; call it directly):
  .\amd_ci\lib\preamble.ps1 -Name "my-job-name"
#>

param([string]$Name = "amd_ci")

$ErrorActionPreference = "Continue"

# RUNNER_OS has been observed EMPTY on these jobs, so do not branch on it. Check for
# the one thing that actually matters: a reclaimable temp root.
if ([string]::IsNullOrEmpty($env:RUNNER_TEMP)) {
    Write-Host "WARNING: RUNNER_TEMP is unset inside a GitHub job; falling back to TEMP,"
    Write-Host "         which is NOT reclaimed between jobs on these persistent boxes."
    $root = $env:TEMP
} else {
    $root = $env:RUNNER_TEMP
}

$work = Join-Path $root $Name
foreach ($sub in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $sub) | Out-Null
}

$env:AMD_CI_WORK = $work
$env:TMPDIR = Join-Path $work "tmp"
$env:TEMP = Join-Path $work "tmp"
$env:TMP = Join-Path $work "tmp"

if (-not [string]::IsNullOrEmpty($env:GITHUB_ENV)) {
    Add-Content -Path $env:GITHUB_ENV -Value "AMD_CI_WORK=$work"
    Add-Content -Path $env:GITHUB_ENV -Value "TMPDIR=$work\tmp"
    Add-Content -Path $env:GITHUB_ENV -Value "TEMP=$work\tmp"
    Add-Content -Path $env:GITHUB_ENV -Value "TMP=$work\tmp"
}

Write-Host "amd_ci work root: $work"
Write-Host "machine: $env:COMPUTERNAME  RUNNER_OS='$env:RUNNER_OS' (empty is normal here)"
$drive = (Get-Item $work).PSDrive
if ($null -ne $drive) {
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    Write-Host "free on $($drive.Name): $freeGB GB"
}
exit 0
