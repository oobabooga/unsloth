# Isolation preamble for Windows AMD CI jobs: the counterpart of preamble.sh for
# `shell: powershell` (Windows PowerShell 5.1; there is no pwsh and no bash).
#
# Usage from a workflow step:
#   .\amd_ci\lib\preamble.ps1 -Name "my-job-name"
# Creates the work root under $env:RUNNER_TEMP (the only tree the runner reclaims
# between jobs) and appends AMD_CI_WORK / TMP / TEMP to $env:GITHUB_ENV so later
# steps see it.
param([string]$Name = "amd_ci")

$ErrorActionPreference = "Continue"

if (-not $env:RUNNER_TEMP) {
    Write-Host "FATAL: RUNNER_TEMP is empty. Refusing to put the work root somewhere the job cannot see."
    exit 1
}

$work = Join-Path $env:RUNNER_TEMP $Name
foreach ($sub in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $sub) | Out-Null
}
$tmp = Join-Path $work "tmp"

$env:AMD_CI_WORK = $work
$env:TMP = $tmp
$env:TEMP = $tmp

if ($env:GITHUB_ENV) {
    Add-Content -Path $env:GITHUB_ENV -Value "AMD_CI_WORK=$work"
    Add-Content -Path $env:GITHUB_ENV -Value "TMP=$tmp"
    Add-Content -Path $env:GITHUB_ENV -Value "TEMP=$tmp"
}

Write-Host "amd_ci work root: $work"
Write-Host ("RUNNER_OS=[{0}] COMPUTERNAME={1}" -f $env:RUNNER_OS, $env:COMPUTERNAME)
exit 0
