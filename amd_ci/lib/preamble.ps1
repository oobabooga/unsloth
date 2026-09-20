# Isolation preamble for the WINDOWS half of the AMD CI pool. The counterpart of
# lib/preamble.sh, which refuses to run here.
#
# Run from a step whose only job is setup, under `shell: powershell`:
#
#   .\amd_ci\lib\preamble.ps1 -Name "pr1234-windows"
#
# It sets AMD_CI_WORK and TMP/TEMP, and appends them to $env:GITHUB_ENV so every
# later step in the job sees them.
#
# The rules encoded here, each learned by losing a run to them:
#
#   $ErrorActionPreference   "Continue", matching preamble.sh's `set +e`. A job
#                            whose subject is "does this command fail" must not
#                            abort on the first non-zero exit.
#   $env:RUNNER_TEMP         the only tree the runner reclaims between jobs, and
#                            the PowerShell spelling. A bash-style $RUNNER_TEMP
#                            inside a PowerShell block expands to "" and the job
#                            writes to the wrong place while still exiting 0.
#   $env:RUNNER_OS           observed EMPTY on these boxes, so it is echoed
#                            rather than branched on.
#   UTF8 on every write      Set-Content defaults to the ANSI code page here, and
#                            $GITHUB_ENV read back as cp1252 is mojibake.
#
# PowerShell 5.1 Desktop only: there is no pwsh on any of the four boxes, so
# nothing below may use 7.x syntax (no ternaries, no ??, no -NoNewLine on
# Add-Content in a way 5.1 rejects).

param(
  [Parameter(Mandatory = $false)]
  [string] $Name = "amd_ci"
)

$ErrorActionPreference = "Continue"

Write-Host "RUNNER_OS='$env:RUNNER_OS' (empty is expected on these boxes)"
Write-Host "machine: $env:COMPUTERNAME"
Write-Host ("PowerShell: " + $PSVersionTable.PSVersion.ToString())

$root = $env:RUNNER_TEMP
if ([string]::IsNullOrWhiteSpace($root)) {
  Write-Host "WARNING: RUNNER_TEMP is unset; falling back to the user TEMP, which is"
  Write-Host "         NOT reclaimed between jobs."
  $root = $env:TEMP
}
if ([string]::IsNullOrWhiteSpace($root)) {
  Write-Host "FATAL: neither RUNNER_TEMP nor TEMP is set, so there is nowhere the job"
  Write-Host "       can write that the runner will reclaim. Refusing rather than"
  Write-Host "       writing somewhere the job cannot see."
  exit 1
}

$work = Join-Path $root $Name
foreach ($sub in @("", "bin", "out", "src", "tmp")) {
  $path = if ($sub -eq "") { $work } else { Join-Path $work $sub }
  if (-not (Test-Path -LiteralPath $path)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
  }
}

$env:AMD_CI_WORK = $work
$env:TMP = Join-Path $work "tmp"
$env:TEMP = $env:TMP

if (-not [string]::IsNullOrWhiteSpace($env:GITHUB_ENV)) {
  $lines = @(
    "AMD_CI_WORK=$work",
    "TMP=$env:TMP",
    "TEMP=$env:TEMP"
  )
  # Named encoding on every write: the default here is the ANSI code page, and a
  # path read back as cp1252 is a path the next step cannot open.
  Add-Content -Path $env:GITHUB_ENV -Value $lines -Encoding UTF8
}

Write-Host "amd_ci work root: $work"
$drive = (Get-Item -LiteralPath $work).PSDrive
if ($null -ne $drive) {
  $freeGb = [math]::Round($drive.Free / 1GB, 1)
  Write-Host ("free on " + $drive.Name + ": " + $freeGb + " GiB")
}
