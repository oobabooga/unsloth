# Windows counterpart of preamble.sh, for steps with `shell: powershell`.
# Exports AMD_CI_WORK (under $env:RUNNER_TEMP, the only tree reclaimed per job)
# and TEMP/TMP/TMPDIR into $env:GITHUB_ENV.
param([string]$Name = "amd_ci")
$ErrorActionPreference = "Continue"
Write-Host "RUNNER_OS='$env:RUNNER_OS' RUNNER_NAME='$env:RUNNER_NAME'"
if ([string]::IsNullOrEmpty($env:RUNNER_TEMP)) {
  Write-Host "FATAL: RUNNER_TEMP is empty; refusing to pick a work root the job cannot see."
  exit 1
}
$work = Join-Path $env:RUNNER_TEMP $Name
foreach ($d in @("bin", "out", "src", "tmp")) {
  New-Item -ItemType Directory -Force -Path (Join-Path $work $d) | Out-Null
}
$tmp = Join-Path $work "tmp"
if (-not [string]::IsNullOrEmpty($env:GITHUB_ENV)) {
  Add-Content -Encoding ascii -Path $env:GITHUB_ENV -Value "AMD_CI_WORK=$work"
  Add-Content -Encoding ascii -Path $env:GITHUB_ENV -Value "TEMP=$tmp"
  Add-Content -Encoding ascii -Path $env:GITHUB_ENV -Value "TMP=$tmp"
  Add-Content -Encoding ascii -Path $env:GITHUB_ENV -Value "TMPDIR=$tmp"
}
Write-Host "amd_ci work root: $work"
Get-PSDrive -PSProvider FileSystem | Where-Object { $env:RUNNER_TEMP.StartsWith($_.Root) } | Format-Table Name, Used, Free -AutoSize
exit 0
