# Isolation preamble for every AMD CI job on the WINDOWS half of the pool.
# The counterpart of lib/preamble.sh, for `shell: powershell` (5.1 Desktop is
# the only PowerShell on these boxes; there is no pwsh and no bash).
#
#   $env:RUNNER_TEMP  the only tree the runner reclaims between jobs; everything
#                     the job writes lives under it.
#   sudo shim         nothing here needs elevation; a shim on PATH turns "we did
#                     not escalate" into a logged, checkable property.
#   Add-Content       PowerShell 5.1's Out-File writes a UTF-16/BOM file, which
#                     GITHUB_ENV cannot parse (lint W104); Add-Content is ASCII
#                     for the ASCII paths written here.
#
# Usage from a workflow step:
#   .\amd_ci\lib\preamble.ps1 -Name "pr1234-windows"
# It sets AMD_CI_WORK, TMP and TEMP for the job and prepends AMD_CI_WORK\bin to PATH.
param(
    [string]$Name = "amd_ci"
)
$ErrorActionPreference = "Continue"

if (-not ([System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT)) {
    Write-Host "FATAL: preamble.ps1 is for the Windows AMD runners; on Linux source lib/preamble.sh."
    exit 1
}

$runnerTemp = $env:RUNNER_TEMP
if (-not $runnerTemp) {
    Write-Host "WARNING: RUNNER_TEMP is unset; falling back to the user TEMP, which is NOT reclaimed between jobs."
    $runnerTemp = $env:TEMP
}
# Short on purpose: Studio installs below this root and Windows path limits are
# reached long before anything else fails.
$work = Join-Path $runnerTemp $Name
foreach ($sub in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $sub) | Out-Null
}

$shim = Join-Path $work "bin\sudo.cmd"
@(
    "@echo off",
    "echo refused privilege escalation: %* 1>&2",
    "echo %DATE% %TIME% %*>> `"$work\out\escalation-attempts.log`"",
    "exit /b 1"
) | Set-Content -LiteralPath $shim -Encoding Ascii

$env:AMD_CI_WORK = $work
$env:TMP = Join-Path $work "tmp"
$env:TEMP = $env:TMP
$env:PATH = (Join-Path $work "bin") + ";" + $env:PATH

if ($env:GITHUB_ENV) {
    Add-Content -Path $env:GITHUB_ENV -Value "AMD_CI_WORK=$work"
    Add-Content -Path $env:GITHUB_ENV -Value "TMP=$($env:TMP)"
    Add-Content -Path $env:GITHUB_ENV -Value "TEMP=$($env:TEMP)"
    Add-Content -Path $env:GITHUB_ENV -Value "PATH=$($env:PATH)"
}

Write-Host "amd_ci work root: $work"
Write-Host "machine: $env:COMPUTERNAME  runner os (suspect, see W103): '$env:RUNNER_OS'"
try {
    $drive = (Get-Item -LiteralPath $work).PSDrive
    Write-Host ("free on {0}: {1:N1} GB" -f $drive.Name, ($drive.Free / 1GB))
} catch {
    Write-Host "free space: unknown ($($_.Exception.Message))"
}
