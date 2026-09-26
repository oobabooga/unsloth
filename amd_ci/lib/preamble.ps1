# Isolation preamble for AMD CI jobs on the WINDOWS runners (`shell: powershell`,
# PowerShell 5.1 Desktop; there is no pwsh and no bash). Counterpart of preamble.sh.
#
#   $env:RUNNER_TEMP  the only tree the runner reclaims between jobs; the work root lives under it.
#   sudo shim         nothing here needs elevation; a sudo.cmd on PATH that refuses and logs
#                     makes that checkable.
#   GITHUB_ENV        written with Add-Content -Encoding ascii: PowerShell 5.1 Out-File writes a
#                     BOM, which corrupts the first variable name (W104).
#
# Usage from a workflow step:  .\amd_ci\lib\preamble.ps1 -Name "my-job"
# Exports AMD_CI_WORK, TEMP / TMP / TMPDIR, PIP_CACHE_DIR and PATH for later steps.
param([string]$Name = "amd_ci")

$ErrorActionPreference = "Continue"

$temp = $env:RUNNER_TEMP
if ([string]::IsNullOrEmpty($temp)) {
    if (-not [string]::IsNullOrEmpty($env:GITHUB_ENV)) {
        Write-Host "FATAL: RUNNER_TEMP is empty inside a GitHub job; refusing to guess a work root."
        exit 1
    }
    $temp = [System.IO.Path]::GetTempPath()
}

$work = Join-Path $temp $Name
foreach ($d in @("bin", "out", "src", "tmp", "pip-cache")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $d) | Out-Null
}

$log = Join-Path $work "out\escalation-attempts.log"
$shim = "@echo off`r`necho refused privilege escalation: %* 1>&2`r`necho %DATE% %TIME% %* >> `"$log`"`r`nexit /b 1`r`n"
Set-Content -Path (Join-Path $work "bin\sudo.cmd") -Value $shim -Encoding ascii
Set-Content -Path (Join-Path $work "bin\runas.cmd") -Value $shim -Encoding ascii

$tmp = Join-Path $work "tmp"
$bin = Join-Path $work "bin"
$env:AMD_CI_WORK = $work
$env:TEMP = $tmp
$env:TMP = $tmp
$env:TMPDIR = $tmp
$env:PIP_CACHE_DIR = Join-Path $work "pip-cache"
$env:PATH = "$bin;$env:PATH"

if (-not [string]::IsNullOrEmpty($env:GITHUB_ENV)) {
    Add-Content -Path $env:GITHUB_ENV -Encoding ascii -Value @(
        "AMD_CI_WORK=$work",
        "TEMP=$tmp",
        "TMP=$tmp",
        "TMPDIR=$tmp",
        "PIP_CACHE_DIR=$($env:PIP_CACHE_DIR)"
    )
    if (-not [string]::IsNullOrEmpty($env:GITHUB_PATH)) {
        Add-Content -Path $env:GITHUB_PATH -Encoding ascii -Value $bin
    }
}

Write-Host "amd_ci work root: $work"
Write-Host "machine: $env:COMPUTERNAME  RUNNER_OS='$env:RUNNER_OS'  PowerShell $($PSVersionTable.PSVersion)"
try {
    $drive = (Get-Item $temp).PSDrive
    Write-Host ("free on {0}: {1:N1} GB" -f $drive.Name, ($drive.Free / 1GB))
} catch {
    Write-Host "free space: unavailable ($_)"
}
exit 0
