# Prints the evidence from run_install.ps1 and asserts the leg's expectation.
param(
    [Parameter(Mandatory = $true)][string]$OutDir,
    [Parameter(Mandatory = $true)][string]$Expect   # 'success' | 'repro'
)
$envTxt = Get-Content -LiteralPath (Join-Path $OutDir 'env.txt') -Raw -ErrorAction SilentlyContinue
$log = Get-Content -LiteralPath (Join-Path $OutDir 'install.log') -ErrorAction SilentlyContinue
$pkgs = Get-Content -LiteralPath (Join-Path $OutDir 'pkgs.txt') -Raw -ErrorAction SilentlyContinue
$left = Get-Content -LiteralPath (Join-Path $OutDir 'leftover.txt') -Raw -ErrorAction SilentlyContinue
$rc = (Get-Content -LiteralPath (Join-Path $OutDir 'done.txt') -Raw -ErrorAction SilentlyContinue)
if ($null -ne $rc) { $rc = $rc.Trim() }
$interesting = @($log | Select-String -Pattern 'install unsloth|File not found|8\.3 short|without freezing|Failed to install|\[ERROR\]|Setup Complete|torch ' | ForEach-Object { $_.Line.Trim() })
$summary = @("### $env:LEG", '```', $envTxt, "installer exit: $rc", $pkgs, $left, '```', '```') + $interesting + @('```')
$extra = @()
foreach ($f in 'elapsed.txt', 'cli.txt', 'hung-descendants.txt', 'hung-threads.txt') {
    $fp = Join-Path $OutDir $f
    if (Test-Path -LiteralPath $fp) { $extra += "--- $f ---"; $extra += @(Get-Content -LiteralPath $fp) }
}
$err = Join-Path $OutDir 'install.err.log'
if (Test-Path -LiteralPath $err) { $extra += '--- install.err.log (last 15) ---'; $extra += @(Get-Content -LiteralPath $err | Select-Object -Last 15) }
$summary += @('```') + $extra + @('```')
$summary | ForEach-Object { Write-Host $_ }
if ($env:GITHUB_STEP_SUMMARY) { $summary | Out-File -Append -FilePath $env:GITHUB_STEP_SUMMARY -Encoding utf8 }
Write-Host '--- install.log tail ---'
$log | Select-Object -Last 40 | ForEach-Object { Write-Host $_ }

$fnf = @($log | Select-String -Pattern 'File not found').Count
$warn83 = @($log | Select-String -Pattern 'no 8\.3 short name').Count
if ($Expect -eq 'repro') {
    if ($rc -ne '0' -and $fnf -gt 0) { Write-Host 'EXPECTED: main reproduces the failure'; exit 0 }
    Write-Host "::error::control did not reproduce (rc=$rc, File-not-found lines=$fnf)"; exit 1
}
$ok = ($rc -eq '0') -and ($fnf -eq 0) -and ($warn83 -eq 0) -and ($pkgs -match 'unsloth \d') -and ($pkgs -match 'torch \d') -and ($left -match 'leftover=0')
if ($ok) { Write-Host 'PASS'; exit 0 }
Write-Host "::error::install leg failed (rc=$rc, File-not-found=$fnf, warn83=$warn83)"; exit 1
