param([string]$Phase, [string]$ExitCode)
$ErrorActionPreference = "Continue"
$venv = "$env:UNSLOTH_STUDIO_HOME\unsloth_studio"
$py = "$venv\Scripts\python.exe"
$cli = "$venv\Scripts\unsloth.exe"
$log = "$env:WORK\out\$Phase.log"
$manifest = Join-Path $venv "unsloth_install_manifest.json"

Write-Host "=== $Phase (leg $env:LEG) ==="
Write-Host "exit code: $ExitCode"
if (Test-Path -LiteralPath $log) {
    Select-String -LiteralPath $log -Pattern "pairing Triton for Windows|could not pair|does not pair with torch|studio install incomplete|forcing dependency pass|could not restore the install manifest|Triton replacement failed" |
        ForEach-Object { Write-Host ("log: " + $_.Line.Trim()) }
}
$hasManifest = Test-Path -LiteralPath $manifest -PathType Leaf
$hasPending = Test-Path -LiteralPath "$manifest.triton-pending" -PathType Leaf
Write-Host "manifest present: $hasManifest"
Write-Host "triton-pending present: $hasPending"
$versions = & $py -c "import importlib.metadata as m; print('torch', m.version('torch')); print('triton-windows', ','.join(d.version for d in m.distributions(name='triton-windows')))" 2>&1 | Out-String
Write-Host $versions.Trim()
$capRaw = & $cli studio desktop-capabilities --json 2>$null | Out-String
$cap = $null
try { $cap = $capRaw | ConvertFrom-Json } catch { }
if ($cap) {
    Write-Host "desktop-capabilities: studio_install_ok=$($cap.studio_install_ok) studio_install_reason=$($cap.studio_install_reason) llama_runtime_ok=$($cap.llama_runtime_ok)"
} else {
    Write-Host "desktop-capabilities did not answer JSON: $capRaw"
}
$ok = [bool]($cap -and $cap.studio_install_ok -eq $true)
$tw = if ($versions -match 'triton-windows (\S*)') { $Matches[1] } else { "" }

# What each leg is expected to show. main: the trigger leaves exit 0 with no manifest (Desktop: not ready).
# fix: the trigger leaves exit 0 with the manifest (Desktop: ready). Phase 3 removes the trigger: both legs pair 3.8.
$expectOk = if ($Phase -eq "3-update-local") { $true } else { $env:LEG -eq "fix" }
$match = ($ExitCode -eq "0") -and ($ok -eq $expectOk)
$line = "| $env:LEG | $Phase | exit $ExitCode | triton-windows $tw | manifest $hasManifest | studio_install_ok $($cap.studio_install_ok) ($($cap.studio_install_reason)) | expected ok=$expectOk | $(if ($match) { 'MATCH' } else { 'MISMATCH' }) |"
Add-Content -Path "$env:WORK\out\verdict.txt" -Value $line
Write-Host $line
exit 0
