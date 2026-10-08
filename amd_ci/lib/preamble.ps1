# Isolation preamble for the WINDOWS AMD CI jobs, the counterpart of preamble.sh.
# Run first, under `shell: powershell` (5.1 Desktop; there is no pwsh or bash):
#   .\amd_ci\lib\preamble.ps1 -Name my-job-name
# Creates $RUNNER_TEMP\<Name>\{bin,out,src,tmp} and appends AMD_CI_WORK, TMP and
# TEMP to $GITHUB_ENV. $RUNNER_TEMP is the only tree the runner reclaims per job.
param([string]$Name = "amd_ci")

$ErrorActionPreference = "Continue"
$root = $env:RUNNER_TEMP
if (-not $root) {
    if ($env:GITHUB_ENV) {
        Write-Host "FATAL: RUNNER_TEMP is unset inside a GitHub job; refusing to pick a work root the runner never reclaims."
        exit 1
    }
    $root = $env:TEMP
}
$work = Join-Path $root $Name
foreach ($d in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $d) | Out-Null
}
if ($env:GITHUB_ENV) {
    $tmp = Join-Path $work "tmp"
    Add-Content -Path $env:GITHUB_ENV -Value "AMD_CI_WORK=$work"
    Add-Content -Path $env:GITHUB_ENV -Value "TMP=$tmp"
    Add-Content -Path $env:GITHUB_ENV -Value "TEMP=$tmp"
}
Write-Host "amd_ci work root: $work"
Write-Host "RUNNER_OS: '$env:RUNNER_OS'  machine: $env:COMPUTERNAME  user: $env:USERNAME"
exit 0
