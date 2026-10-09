# Isolation preamble for every AMD CI job on the WINDOWS runners; the counterpart of
# lib/preamble.sh. Run it first, from a step whose only job is setup:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\amd_ci\lib\preamble.ps1 -Name my-job
#
# It creates the work root under $env:RUNNER_TEMP (the only tree the runner reclaims
# between jobs) and appends AMD_CI_WORK / TMP / TEMP to $env:GITHUB_ENV, so every later
# step sees them. Add-Content, not Out-File: PowerShell 5.1's Out-File writes a BOM that
# corrupts the first GITHUB_ENV line (lint W104).
#
# There is no sudo shim: nothing on Windows escalates through a PATH lookup, and the
# jobs run as the runner's service account.

param([string]$Name = "amd_ci")

$ErrorActionPreference = "Continue"

$root = $env:RUNNER_TEMP
if (-not $root) {
    if ($env:GITHUB_ENV) {
        Write-Host "FATAL: RUNNER_TEMP is empty inside a GitHub job. Refusing to pick a work root"
        Write-Host "       the runner never reclaims."
        exit 1
    }
    $root = [System.IO.Path]::GetTempPath()
}

$work = Join-Path $root $Name
foreach ($sub in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $sub) | Out-Null
}
$tmp = Join-Path $work "tmp"

if ($env:GITHUB_ENV) {
    Add-Content -Path $env:GITHUB_ENV -Value "AMD_CI_WORK=$work"
    Add-Content -Path $env:GITHUB_ENV -Value "TMP=$tmp"
    Add-Content -Path $env:GITHUB_ENV -Value "TEMP=$tmp"
}

Write-Host "amd_ci work root: $work"
Write-Host "RUNNER_OS='$($env:RUNNER_OS)' machine=$($env:COMPUTERNAME)"
try {
    $drive = (Get-Item -LiteralPath $root).PSDrive
    Write-Host ("free on {0}: {1:N1} GiB" -f $drive.Name, ($drive.Free / 1GB))
} catch {
    Write-Host "free space: unavailable ($($_.Exception.Message))"
}
exit 0
