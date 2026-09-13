# Isolation preamble for every AMD CI job on the WINDOWS half of the pool.
# The counterpart of lib/preamble.sh, run from a step whose only job is setup:
#
#   - name: Preamble
#     run: .\amd_ci\lib\preamble.ps1 -Name "my-job-name"
#
# It sets AMD_CI_WORK, TMPDIR and PATH for the rest of the job, via $GITHUB_ENV.
#
# What differs from the bash preamble, and why:
#
#   no `set +e`        PowerShell does not abort a step on a non-zero native
#                      exit code, so there is nothing to switch off. What DOES
#                      abort it is a terminating error, so callers set
#                      $ErrorActionPreference = "Continue" in their own steps and
#                      check $LASTEXITCODE. This file sets "Stop" for itself,
#                      because a preamble that half-runs is worse than one that
#                      fails: the job would carry on writing to a work root that
#                      does not exist.
#   no sudo shim       There is no sudo. The Windows equivalent of the property
#                      the shim makes checkable -- "this job did not need
#                      admin" -- is that the runner service is not elevated, so
#                      the preamble reports the elevation state rather than
#                      shimming anything.
#   $env:RUNNER_TEMP   Same role as on Linux: the only tree reclaimed between
#                      jobs. Written as $env:RUNNER_TEMP, never $RUNNER_TEMP,
#                      which PowerShell expands to the empty string -- lint rule
#                      E104 exists because that silently wrote to the wrong place
#                      and still exited 0.
#
# PowerShell 5.1 Desktop only. There is no pwsh on these boxes, so nothing here
# may use a 6+ construct (no ?., no ??, no -Encoding utf8NoBOM).

[CmdletBinding()]
param(
    [string] $Name = "amd_ci"
)

$ErrorActionPreference = "Stop"

# Refuse the shape that would "succeed" while describing another machine. The
# bash preamble guards the mirror image of this; here the risk is a job that
# reached PowerShell but is not actually on the Windows runner we think.
if (-not $env:RUNNER_TEMP) {
    if ($env:GITHUB_ENV) {
        Write-Host "FATAL: RUNNER_TEMP is unset inside a GitHub job."
        Write-Host "       On these runners that means the step is not running where it"
        Write-Host "       thinks it is. Refusing rather than writing the work root to a"
        Write-Host "       path the rest of the job will not look in."
        exit 1
    }
    # Outside a job (a local syntax check), fall back so the file is runnable.
    $env:RUNNER_TEMP = $env:TEMP
}

$work = Join-Path $env:RUNNER_TEMP $Name
foreach ($sub in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $sub) | Out-Null
}

$env:AMD_CI_WORK = $work
$env:TMPDIR = Join-Path $work "tmp"
# TEMP/TMP too: Windows tooling reads these, not TMPDIR, so setting only the
# POSIX name leaves pip and python writing to the machine-wide temp that is NOT
# reclaimed between jobs.
$env:TEMP = $env:TMPDIR
$env:TMP = $env:TMPDIR
$env:PATH = (Join-Path $work "bin") + ";" + $env:PATH

if ($env:GITHUB_ENV) {
    # Add-Content, never Out-File: on PowerShell 5.1 `-Encoding utf8` means utf8
    # WITH a BOM, and the BOM corrupts the first variable name written. Lint rule
    # W104 is this mistake.
    Add-Content -Path $env:GITHUB_ENV -Value "AMD_CI_WORK=$work"
    Add-Content -Path $env:GITHUB_ENV -Value "TMPDIR=$($env:TMPDIR)"
    Add-Content -Path $env:GITHUB_ENV -Value "TEMP=$($env:TMPDIR)"
    Add-Content -Path $env:GITHUB_ENV -Value "TMP=$($env:TMPDIR)"
    Add-Content -Path $env:GITHUB_ENV -Value "PATH=$($env:PATH)"
}

Write-Host "amd_ci work root: $work"

# Echoed rather than branched on: RUNNER_OS has been observed EMPTY on these
# boxes, and the machine-name prefix is shared with the Linux pool, so neither
# identifies the OS. Recording both is what makes a confusing run diagnosable.
Write-Host "RUNNER_OS='$($env:RUNNER_OS)'  computer='$($env:COMPUTERNAME)'"
Write-Host "powershell: $($PSVersionTable.PSVersion)  edition: $($PSVersionTable.PSEdition)"

try {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    $elevated = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    Write-Host "elevated: $elevated  user: $($identity.Name)"
} catch {
    Write-Host "elevated: unknown ($($_.Exception.Message))"
}

try {
    $drive = (Get-Item -LiteralPath $env:RUNNER_TEMP).PSDrive
    $freeGb = [math]::Round($drive.Free / 1GB, 1)
    $usedGb = [math]::Round($drive.Used / 1GB, 1)
    Write-Host "disk $($drive.Name): free ${freeGb} GiB, used ${usedGb} GiB"
} catch {
    Write-Host "disk: unknown ($($_.Exception.Message))"
}
