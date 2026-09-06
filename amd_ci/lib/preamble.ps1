# Isolation preamble for every AMD CI job on the WINDOWS runners. The counterpart
# of lib/preamble.sh, and the Windows half of the pool has no bash at all, so it
# is a rewrite rather than a wrapper.
#
# Usage from a workflow step (`shell: powershell`, never `pwsh` and never `bash`):
#   .\amd_ci\lib\preamble.ps1 -Name my-job-name
# It sets AMD_CI_WORK and appends WORK/TMP/PATH to $env:GITHUB_ENV.
#
# The rules encoded here, each the reason a run was lost:
#
#   $ErrorActionPreference   PowerShell's default is Continue for non-terminating
#                            errors, which is what a measuring job wants; it is
#                            set explicitly so a caller's Stop cannot leak in and
#                            abort the step on the first unwelcome exit code.
#   sudo shim                the host is persistent and shared. Nothing needs
#                            admin, and a shim makes that checkable rather than
#                            asserted. Every attempt is logged.
#   $env:RUNNER_TEMP         the only tree the runner reclaims between jobs. Note
#                            the `$env:` prefix: a bash-style `$RUNNER_TEMP` in a
#                            PowerShell block is an undefined variable that
#                            expands to "", so the work root silently becomes the
#                            filesystem root and the step still exits 0 (E104).
#   Add-Content, not Out-File  Out-File under PowerShell 5.1 writes a UTF-16 BOM,
#                            and the Actions runner reads GITHUB_ENV as UTF-8, so
#                            the first assignment in the file is corrupted (W104).
#
# WINDOWS ONLY. It refuses under PowerShell 7 rather than half-working: the four
# boxes carry 5.1 Desktop only, so a run that reached this under pwsh is a run
# whose environment is not the one the results would be about.

param(
    [string] $Name = "amd_ci"
)

$ErrorActionPreference = "Continue"

if ($PSVersionTable.PSEdition -eq "Core") {
    Write-Host "FATAL: running under PowerShell Core (pwsh)."
    Write-Host "       The AMD Windows runners have PowerShell 5.1 Desktop only, so this"
    Write-Host "       is not the environment the job's results would describe."
    Write-Host "       Use 'shell: powershell'."
    exit 1
}

# Echoed rather than branched on: $env:RUNNER_OS has been observed EMPTY on these
# boxes under some runner versions and correct on others, so it is evidence for a
# reader and never a condition here (W103).
Write-Host "RUNNER_OS as seen by this shell: '$env:RUNNER_OS'"
Write-Host "machine: $env:COMPUTERNAME"

$temp = $env:RUNNER_TEMP
if ([string]::IsNullOrWhiteSpace($temp)) {
    if (-not [string]::IsNullOrWhiteSpace($env:GITHUB_ENV)) {
        Write-Host "WARNING: RUNNER_TEMP is unset inside a GitHub job; falling back to TEMP,"
        Write-Host "         which is NOT reclaimed between jobs."
    }
    $temp = $env:TEMP
}
if ([string]::IsNullOrWhiteSpace($temp)) {
    Write-Host "FATAL: neither RUNNER_TEMP nor TEMP is set, so there is nowhere to put the"
    Write-Host "       work root. Refusing rather than writing to the filesystem root."
    exit 1
}

$work = Join-Path $temp $Name
foreach ($sub in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $sub) | Out-Null
}

# The shim is a .cmd so it is found by CreateProcess as well as by PowerShell:
# a .ps1 on PATH is not executable by a bare `sudo` from a child process.
$shim = Join-Path $work "bin\sudo.cmd"
@(
    '@echo off',
    'echo refused privilege escalation: %* 1>&2',
    'echo %DATE% %TIME% %* >> "%AMD_CI_WORK%\out\escalation-attempts.log"',
    'exit /b 1'
) | Set-Content -LiteralPath $shim -Encoding ascii

$env:AMD_CI_WORK = $work
$env:TMP = Join-Path $work "tmp"
$env:TEMP = $env:TMP
$env:PATH = (Join-Path $work "bin") + ";" + $env:PATH

if (-not [string]::IsNullOrWhiteSpace($env:GITHUB_ENV)) {
    Add-Content -LiteralPath $env:GITHUB_ENV -Encoding ascii -Value @(
        "AMD_CI_WORK=$work",
        "TMP=$env:TMP",
        "TEMP=$env:TEMP",
        "PATH=$env:PATH"
    )
}

Write-Host "amd_ci work root: $work"
$drive = (Get-Item -LiteralPath $work).PSDrive
if ($null -ne $drive) {
    $freeGb = [math]::Round($drive.Free / 1GB, 1)
    $usedGb = [math]::Round($drive.Used / 1GB, 1)
    Write-Host "drive $($drive.Name): $freeGb GB free, $usedGb GB used"
}
