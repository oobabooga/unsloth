# Isolation preamble for every AMD CI job on the WINDOWS half of the pool.
# The PowerShell 5.1 counterpart of lib/preamble.sh; run it from a step whose only
# job is setup, under `shell: powershell` (there is no pwsh on these boxes).
#
#   $env:RUNNER_TEMP   the only tree the runner reclaims between jobs. Everything
#                      the job writes goes under it, never under the checkout.
#   GITHUB_ENV         written with Add-Content, never Out-File: PowerShell 5.1's
#                      Out-File emits a UTF-16/BOM file that the runner cannot parse
#                      (lint rule W104).
#   RUNNER_OS          observed EMPTY on these runners; nothing here branches on it.
#
# Usage from a workflow step:
#   .\amd_ci\lib\preamble.ps1 -Name "pr1234-windows"
# It exports AMD_CI_WORK and appends AMD_CI_WORK / TMP / TEMP to $env:GITHUB_ENV.

param(
    [string]$Name = "amd_ci"
)

$ErrorActionPreference = "Continue"

$runnerTemp = $env:RUNNER_TEMP
if ([string]::IsNullOrEmpty($runnerTemp)) {
    $runnerTemp = [System.IO.Path]::GetTempPath().TrimEnd('\')
    Write-Host "WARNING: RUNNER_TEMP is unset; falling back to $runnerTemp, which is NOT reclaimed between jobs."
}

$work = Join-Path $runnerTemp $Name
foreach ($sub in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $sub) | Out-Null
}

$env:AMD_CI_WORK = $work
$env:TMP = Join-Path $work "tmp"
$env:TEMP = $env:TMP

if (-not [string]::IsNullOrEmpty($env:GITHUB_ENV)) {
    Add-Content -Path $env:GITHUB_ENV -Value "AMD_CI_WORK=$work"
    Add-Content -Path $env:GITHUB_ENV -Value "TMP=$($env:TMP)"
    Add-Content -Path $env:GITHUB_ENV -Value "TEMP=$($env:TEMP)"
}

Write-Host "amd_ci work root: $work"
Write-Host "machine: $env:COMPUTERNAME  RUNNER_OS='$env:RUNNER_OS'  PowerShell $($PSVersionTable.PSVersion)"
try {
    $drive = Get-PSDrive -Name ($work.Substring(0, 1)) -ErrorAction Stop
    Write-Host ("free on {0}: {1:N1} GB" -f $drive.Root, ($drive.Free / 1GB))
} catch {
    Write-Host "free space: unknown ($($_.Exception.Message))"
}
exit 0
