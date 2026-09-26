# Isolation preamble for the Windows AMD CI jobs, the counterpart to preamble.sh.
# The workflow template has always called it; it was never shipped, so every
# Windows job died on its first step with CommandNotFoundException.
#
#   $env:RUNNER_TEMP  the only tree the runner reclaims between jobs.
#   no root           nothing here needs it, and these boxes are shared and
#                     persistent.
#   $ErrorActionPreference = "Continue"
#                     a job whose subject is "does this command fail" must not
#                     lose the measurement to the first non-zero exit.
param([string]$Name = "amd_ci")

$ErrorActionPreference = "Continue"

$root = $env:RUNNER_TEMP
if ([string]::IsNullOrWhiteSpace($root)) {
    Write-Host "WARNING: RUNNER_TEMP is unset inside a GitHub job; falling back to TEMP,"
    Write-Host "         which is not reclaimed between jobs."
    $root = $env:TEMP
}
$work = Join-Path $root $Name
foreach ($sub in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $sub) | Out-Null
}

$env:AMD_CI_WORK = $work
$env:TMPDIR = Join-Path $work "tmp"
$env:TEMP = Join-Path $work "tmp"
$env:TMP = Join-Path $work "tmp"
if ($env:GITHUB_ENV) {
    Add-Content -Path $env:GITHUB_ENV -Value "AMD_CI_WORK=$work"
    Add-Content -Path $env:GITHUB_ENV -Value "TMPDIR=$(Join-Path $work 'tmp')"
    Add-Content -Path $env:GITHUB_ENV -Value "TEMP=$(Join-Path $work 'tmp')"
    Add-Content -Path $env:GITHUB_ENV -Value "TMP=$(Join-Path $work 'tmp')"
}

# Echoed rather than branched on: RUNNER_OS has been observed empty on these boxes.
Write-Host "amd_ci work root: $work"
Write-Host "RUNNER_OS='$env:RUNNER_OS' machine='$env:COMPUTERNAME'"
Get-PSDrive -Name ($work.Substring(0, 1)) |
    ForEach-Object { Write-Host ("free: {0:N1} GB" -f ($_.Free / 1GB)) }
