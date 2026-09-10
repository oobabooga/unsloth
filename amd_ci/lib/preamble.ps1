# Isolation preamble for every AMD CI job on the WINDOWS half of the pool.
# The counterpart to lib/preamble.sh, which is Linux only and refuses to run here.
#
# Run it first, from a step whose only job is setup:
#   - name: Preamble
#     run: .\amd_ci\lib\preamble.ps1 -Name "pr1234-windows"
#
# What it encodes, and why each line is here rather than inline in the workflow:
#
#   ErrorActionPreference   PowerShell's default is Continue for non-terminating
#                           errors but the runner still fails a step on a native
#                           non-zero exit. A job whose subject is "does this
#                           command fail" has to read $LASTEXITCODE itself, so
#                           the preamble sets Continue and every later step
#                           checks the code explicitly.
#   $env:RUNNER_TEMP        The only tree the runner reclaims between jobs. It is
#                           also the variable a bash-style ${RUNNER_TEMP} inside a
#                           PowerShell block silently expands to "" (E104), which
#                           is why the work root is resolved once, here.
#   no sudo shim            There is no sudo on Windows. The equivalent property
#                           is that nothing here runs elevated, and the job has no
#                           way to elevate without an interactive prompt.
#
# Sets AMD_CI_WORK, TEMP and TMP for this step and appends all three to
# $GITHUB_ENV for the rest of the job. Add-Content, never Out-File: PowerShell
# 5.1's Out-File writes a UTF-16 BOM that the runner reads as part of the name
# (W104), and there is no PowerShell 7 on these boxes to avoid it.

param(
    [string]$Name = "amd_ci"
)

$ErrorActionPreference = "Continue"

# RUNNER_OS has been observed EMPTY on these runners, so nothing below branches on
# it; it is echoed because a blank value is itself worth seeing in the log.
Write-Host "RUNNER_OS='$env:RUNNER_OS'  machine=$env:COMPUTERNAME"

if ([string]::IsNullOrEmpty($env:RUNNER_TEMP)) {
    Write-Host "WARNING: RUNNER_TEMP is unset inside a GitHub job; falling back to the"
    Write-Host "         user TEMP, which is NOT reclaimed between jobs."
    $root = $env:TEMP
} else {
    $root = $env:RUNNER_TEMP
}
if ([string]::IsNullOrEmpty($root)) {
    Write-Host "FATAL: neither RUNNER_TEMP nor TEMP is set, so there is no work root."
    Write-Host "Not continuing: a job that writes where it cannot read back is the exact"
    Write-Host "silent failure this toolkit exists to prevent."
    exit 1
}

$work = Join-Path $root $Name
foreach ($sub in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $sub) | Out-Null
}

$env:AMD_CI_WORK = $work
$env:TEMP = Join-Path $work "tmp"
$env:TMP = $env:TEMP

Add-Content -Path $env:GITHUB_ENV -Value "AMD_CI_WORK=$work"
Add-Content -Path $env:GITHUB_ENV -Value "TEMP=$env:TEMP"
Add-Content -Path $env:GITHUB_ENV -Value "TMP=$env:TMP"

Write-Host "amd_ci work root: $work"
Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($work).TrimEnd(':', '\')) |
    Select-Object Name, @{n = "UsedGB"; e = { [math]::Round($_.Used / 1GB, 1) } },
                  @{n = "FreeGB"; e = { [math]::Round($_.Free / 1GB, 1) } } |
    Format-Table | Out-String | Write-Host
