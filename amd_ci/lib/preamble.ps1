# Isolation preamble for Windows AMD CI jobs (counterpart of preamble.sh, which is Linux only).
# Run from a `shell: powershell` step: .\amd_ci\lib\preamble.ps1 -Name my-job
# Exports AMD_CI_WORK (under $env:RUNNER_TEMP, the only tree the runner reclaims) and
# appends AMD_CI_WORK / TMP / TEMP to $env:GITHUB_ENV for later steps.
param([string]$Name = "amd_ci")
$ErrorActionPreference = "Continue"
if (-not $env:RUNNER_TEMP) {
    Write-Host "FATAL: RUNNER_TEMP is empty; refusing to pick a work root the runner never reclaims."
    exit 1
}
$work = Join-Path $env:RUNNER_TEMP $Name
foreach ($d in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $d) | Out-Null
}
$tmp = Join-Path $work "tmp"
if ($env:GITHUB_ENV) {
    Add-Content -Path $env:GITHUB_ENV -Encoding utf8 -Value "AMD_CI_WORK=$work"
    Add-Content -Path $env:GITHUB_ENV -Encoding utf8 -Value "TMP=$tmp"
    Add-Content -Path $env:GITHUB_ENV -Encoding utf8 -Value "TEMP=$tmp"
}
Write-Host "amd_ci work root: $work"
Write-Host "RUNNER_OS: '$env:RUNNER_OS'  host: $env:COMPUTERNAME"
exit 0
