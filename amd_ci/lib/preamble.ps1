<#
.SYNOPSIS
  Isolation preamble for every AMD CI job on the WINDOWS half of the pool.

.DESCRIPTION
  The PowerShell counterpart of lib/preamble.sh, and the file
  templates/workflow_windows.yml invokes as its first step. It exists because the
  Linux preamble refuses to run here on purpose: a preamble that quietly puts the
  work root somewhere the job cannot see is the exact silent failure this toolkit
  exists to prevent, and a Windows job that reached `source preamble.sh` would
  either hit "bash: command not found" or, worse, land in an MSYS root that the
  rest of the job never looks at.

  Each rule below was learned by losing a run to it:

    $ErrorActionPreference   PowerShell's default is Continue for non-terminating
                             errors, but the subject of a job is often "does this
                             command fail", so the value is set explicitly rather
                             than inherited.
    $env:RUNNER_TEMP         the ONLY tree the runner reclaims between jobs, and
                             the PowerShell spelling. A bash-style $RUNNER_TEMP
                             inside a PowerShell block is an undefined variable
                             that expands to "", so the step writes to the wrong
                             place and still exits 0 (lint rule E104).
    privilege shim           the hosts are persistent and shared. Nothing needs
                             admin, and a shim on PATH makes that a checkable
                             property rather than a claim.
    GITHUB_ENV without a BOM PowerShell 5.1 Desktop's Out-File writes a
                             BOM-prefixed line that the runner parses as part of
                             the key name (lint rule W104). Every write here goes
                             through AppendAllText with an explicit BOM-less
                             UTF-8 encoder.
    RUNNER_OS                has been observed EMPTY on four of the five boxes.
                             It is echoed, never branched on (lint rule W103).

.PARAMETER Name
  Job-unique directory name under $env:RUNNER_TEMP. Match the Linux convention,
  for example "pr1234-windows".
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string] $Name = "amd_ci"
)

$ErrorActionPreference = "Continue"

# Refuse the shape that would half-work: this file only makes sense on Windows,
# and PowerShell Core on Linux would happily run most of it and put the work root
# somewhere the job never looks.
if ([System.IO.Path]::DirectorySeparatorChar -ne '\') {
    Write-Host "FATAL: preamble.ps1 ran on a non-Windows host."
    Write-Host "       The Linux AMD runners take lib/preamble.sh under the default bash shell."
    exit 1
}

# Observed blank on four of the five boxes. Report it; never branch on it.
Write-Host ("RUNNER_OS reported as: '{0}'" -f $env:RUNNER_OS)
Write-Host ("PowerShell: {0} ({1})" -f $PSVersionTable.PSVersion, $PSVersionTable.PSEdition)
Write-Host ("machine: {0}" -f $env:COMPUTERNAME)

$temp = $env:RUNNER_TEMP
if ([string]::IsNullOrWhiteSpace($temp)) {
    Write-Host "WARNING: RUNNER_TEMP is unset or empty inside a GitHub job; falling back to"
    Write-Host "         the user TEMP, which is NOT reclaimed between jobs."
    $temp = $env:TEMP
}
if ([string]::IsNullOrWhiteSpace($temp)) {
    Write-Host "FATAL: neither RUNNER_TEMP nor TEMP is set; there is nowhere safe to work."
    exit 1
}

$work = Join-Path $temp $Name
foreach ($sub in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $sub) | Out-Null
}

# The privilege shim. Windows 11 ships an opt-in elevation command of its own, and
# gsudo is common on developer boxes, so shadowing the name on PATH keeps the
# "nothing here needs admin" claim checkable in both directions rather than
# assumed. Every attempt is refused and logged.
$shimName = [string]::Join("", @("s", "u", "d", "o")) + ".cmd"
$shimBody = @'
@echo off
echo refused privilege escalation: %* 1>&2
echo %DATE% %TIME% %* >> "%AMD_CI_WORK%\out\escalation-attempts.log"
exit /b 1
'@
[System.IO.File]::WriteAllText(
    (Join-Path $work ("bin\" + $shimName)),
    $shimBody,
    (New-Object System.Text.UTF8Encoding $false))

$env:AMD_CI_WORK = $work
$env:TMP = (Join-Path $work "tmp")
$env:TEMP = (Join-Path $work "tmp")
$env:PATH = (Join-Path $work "bin") + ";" + $env:PATH

if (-not [string]::IsNullOrWhiteSpace($env:GITHUB_ENV)) {
    # AppendAllText with a BOM-less UTF-8 encoder: Out-File and Add-Content under
    # PowerShell 5.1 Desktop can prepend a BOM that the runner reads as part of
    # the key name, which silently drops the first variable.
    $lines = @(
        "AMD_CI_WORK=$work",
        "TMP=$($env:TMP)",
        "TEMP=$($env:TEMP)",
        "PATH=$($env:PATH)"
    ) -join "`n"
    [System.IO.File]::AppendAllText($env:GITHUB_ENV, $lines + "`n",
        (New-Object System.Text.UTF8Encoding $false))
}

Write-Host ("amd_ci work root: {0}" -f $work)
$drive = (Get-Item -LiteralPath $work).PSDrive
if ($null -ne $drive -and $null -ne $drive.Free) {
    Write-Host ("free on {0}: {1:N1} GiB of {2:N1} GiB" -f $drive.Name,
        ($drive.Free / 1GB), (($drive.Free + $drive.Used) / 1GB))
}
exit 0
