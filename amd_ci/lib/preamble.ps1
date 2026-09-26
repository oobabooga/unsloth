# Isolation preamble for AMD CI jobs on the WINDOWS runners; the counterpart of preamble.sh.
# Run first, under `shell: powershell` (5.1 Desktop; there is no pwsh and no bash here):
#   .\amd_ci\lib\preamble.ps1 -Name my-job-name
# Exports AMD_CI_WORK under $env:RUNNER_TEMP (the only tree the runner reclaims between jobs)
# and appends AMD_CI_WORK / TMP / TEMP to $env:GITHUB_ENV with Add-Content: Out-File in 5.1
# writes a BOM that corrupts the first variable (lint W104).
param([string]$Name = "amd_ci")

$ErrorActionPreference = "Continue"
Write-Host "RUNNER_OS='$($env:RUNNER_OS)' machine=$env:COMPUTERNAME PowerShell=$($PSVersionTable.PSVersion)"
$root = $env:RUNNER_TEMP
if (-not $root) {
    Write-Host "FATAL: RUNNER_TEMP is empty; refusing to pick a work root the runner never reclaims."
    exit 1
}
$work = Join-Path $root $Name
foreach ($d in @("out", "tmp", "src")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $d) | Out-Null
}
$tmp = Join-Path $work "tmp"
$env:AMD_CI_WORK = $work
if ($env:GITHUB_ENV) {
    Add-Content -Path $env:GITHUB_ENV -Value "AMD_CI_WORK=$work"
    Add-Content -Path $env:GITHUB_ENV -Value "TMP=$tmp"
    Add-Content -Path $env:GITHUB_ENV -Value "TEMP=$tmp"
}
Write-Host "amd_ci work root: $work"
Get-PSDrive -Name ($work.Substring(0, 1)) | Select-Object Name, Free | Format-Table -AutoSize | Out-String | Write-Host
exit 0
