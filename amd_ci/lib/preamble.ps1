# Isolation preamble for every AMD CI job that runs on a WINDOWS runner.
# The counterpart of lib/preamble.sh, which is Linux only and refuses to run here.
#
# Usage from a workflow step:
#   - name: Preamble
#     shell: powershell
#     run: .\amd_ci\lib\preamble.ps1 -Name "pr1234-windows"
#
# It sets AMD_CI_WORK and appends AMD_CI_WORK / TMP / TEMP / PATH to $env:GITHUB_ENV.
#
# Measured on the four Windows AMD runners (X2-LA01-S05-D01, X2-LA01-S04-D02,
# DESKTOP-O81NVEC, DESKTOP-52OM6HM), and each fact below is a trap that cost a run:
#
#   shell: powershell   is the ONLY correct value. `pwsh` is not installed
#                       (PowerShell 5.1 Desktop only, no PowerShell 7), and
#                       `bash` is not installed either. Both fail with
#                       "command not found" before a single line of the step runs.
#   $env:RUNNER_OS      is EMPTY on these jobs. Never branch on it.
#   Add-Content         writes UTF-8 with no BOM on PowerShell 5.1. The form
#                       GitHub's docs suggest, `Out-File -Encoding utf8`, writes a
#                       BOM on 5.1 (there is no utf8NoBOM before PowerShell 6),
#                       which can corrupt the first name written to GITHUB_ENV.
#   $ErrorActionPreference
#                       GitHub sets it to 'Stop' around a step's script, which is
#                       the analogue of the `bash -e` that preamble.sh disables
#                       with `set +e`. Setting it here would NOT reach the calling
#                       step, so a step whose subject is "does this command fail"
#                       must set `$ErrorActionPreference = 'Continue'` itself.
#   docker              is NOT installed on any of them. A containerised workload
#                       has to stay on the Linux runners.

param(
    [string] $Name = "amd_ci"
)

$ErrorActionPreference = "Continue"

Write-Host "PowerShell $($PSVersionTable.PSVersion) $($PSVersionTable.PSEdition) edition"
Write-Host "machine: $env:COMPUTERNAME"
# Stated every run so nobody rediscovers it by writing a branch that never fires.
# Note the machine-name prefix X2-LA01-* is SHARED with the Linux pool, so the
# name does not tell you which OS you are on either.
Write-Host "RUNNER_OS is '$env:RUNNER_OS' (empty on these runners; do not branch on it)"

if ([string]::IsNullOrEmpty($env:RUNNER_TEMP)) {
    if (-not [string]::IsNullOrEmpty($env:GITHUB_ENV)) {
        Write-Host "WARNING: RUNNER_TEMP is unset inside a GitHub job; falling back to TEMP,"
        Write-Host "         which is NOT reclaimed between jobs."
    }
    $root = $env:TEMP
} else {
    $root = $env:RUNNER_TEMP
}
if ([string]::IsNullOrEmpty($root)) {
    Write-Host "FATAL: neither RUNNER_TEMP nor TEMP is set; refusing to guess a work root."
    exit 1
}

$AmdCiWork = Join-Path $root $Name
foreach ($sub in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $AmdCiWork $sub) | Out-Null
}

# Parity with the sudo shim in preamble.sh. There is no sudo on Windows, but the
# property being preserved is the same one: nothing this toolkit runs needs to
# elevate, and an attempt should be recorded rather than assumed absent.
$shim = Join-Path $AmdCiWork "bin\sudo.cmd"
$log = Join-Path $AmdCiWork "out\escalation-attempts.log"
@(
    "@echo off",
    "echo refused privilege escalation: %* 1>&2",
    "echo %DATE% %TIME% %* >> `"$log`"",
    "exit /b 1"
) | Set-Content -Path $shim -Encoding ASCII

$env:AMD_CI_WORK = $AmdCiWork
$env:TMP = Join-Path $AmdCiWork "tmp"
$env:TEMP = $env:TMP
$env:PATH = (Join-Path $AmdCiWork "bin") + ";" + $env:PATH

if (-not [string]::IsNullOrEmpty($env:GITHUB_ENV)) {
    # Add-Content, not Out-File: no BOM on PowerShell 5.1. See the header.
    Add-Content -Path $env:GITHUB_ENV -Value "AMD_CI_WORK=$env:AMD_CI_WORK"
    Add-Content -Path $env:GITHUB_ENV -Value "TMP=$env:TMP"
    Add-Content -Path $env:GITHUB_ENV -Value "TEMP=$env:TEMP"
    Add-Content -Path $env:GITHUB_ENV -Value "PATH=$env:PATH"
}

Write-Host "amd_ci work root: $env:AMD_CI_WORK"
# Guarded: a path with no drive qualifier makes Split-Path emit an error, and a
# preamble whose last output is a red stack trace teaches people to ignore it.
$qual = Split-Path -Qualifier $AmdCiWork -ErrorAction SilentlyContinue
if (-not [string]::IsNullOrEmpty($qual)) {
    $drive = $qual.TrimEnd(":")
    $vol = Get-PSDrive -Name $drive -ErrorAction SilentlyContinue
    if ($vol) {
        $freeGb = [math]::Round($vol.Free / 1GB, 1)
        $usedGb = [math]::Round($vol.Used / 1GB, 1)
        Write-Host "drive ${drive}: $freeGb GiB free, $usedGb GiB used"
    }
}

# Docker is absent here by measurement. Say so once per job so a workflow that
# drifted into needing it reads as a stated limit and not as a mystery.
if ($null -eq (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "note: no docker CLI on this runner (expected on the Windows AMD boxes);"
    Write-Host "      containerised work belongs on the Linux runners."
}

exit 0
