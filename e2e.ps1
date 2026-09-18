# End to end on a simulated Dynamic Code Security host: the native helper is forced off in a copy
# of the installer, then a fresh install, a re-run blocked by a live venv interpreter, and a clean
# re-run are driven through the real install.ps1.
param([Parameter(Mandatory = $true)][string]$Label, [Parameter(Mandatory = $true)][string]$Shell)
$ErrorActionPreference = "Stop"
$fails = 0
function Check($name, $cond) { if ($cond) { Write-Host "  PASS  [$Label] $name" } else { Write-Host "  FAIL  [$Label] $name"; $script:fails++ } }
$src = Get-Content -Raw -LiteralPath install.ps1
$hook = 'function Test-StudioCanDefineNativeTypes {'
if (-not $src.Contains($hook)) { throw "hook not found" }
$src = $src.Replace($hook, $hook + ' return $false')
$trace = Join-Path $env:RUNNER_TEMP "rung.log"
$hasRung = $src.Contains('function Get-StudioPythonProcessImageTable {')
if ($hasRung) {
    $src = $src.Replace('function Get-StudioPythonProcessImageTable {', 'function Get-StudioPythonProcessImageTable { Add-Content -LiteralPath ''' + $trace + ''' -Value "called"')
    $src = $src.Replace("        if (`$table.Count -eq 0) { return `$null }`n        return `$table", "        Add-Content -LiteralPath '" + $trace + "' -Value (""rows="" + `$table.Count)`n        if (`$table.Count -eq 0) { return `$null }`n        return `$table")
}
Set-Content -LiteralPath install.e2e.ps1 -Value $src -Encoding UTF8
function Invoke-Install($log) {
    $child = '$ErrorActionPreference = ''Stop''; $ProgressPreference = ''SilentlyContinue''; & ./install.e2e.ps1 --local --no-torch; exit $LASTEXITCODE'
    $exe = if ($Shell -eq "ps51") { "powershell" } else { "pwsh" }
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $exe -NoProfile -ExecutionPolicy Bypass -Command $child *> $log
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    return $code
}
New-Item -ItemType Directory -Force -Path logs | Out-Null
Remove-Item -LiteralPath $trace -ErrorAction SilentlyContinue

# 1. Fresh install with the native helper off.
$c1 = Invoke-Install "logs/1-fresh.log"
$l1 = Get-Content -Raw "logs/1-fresh.log"
Write-Host "  fresh install exit=$c1"
Check "fresh install succeeds" ($c1 -eq 0)
Check "the native helper really was off (the degraded-scan warning printed)" ($l1 -match 'without the native helper')
if ($hasRung) {
    $t1 = @(Get-Content -LiteralPath $trace -ErrorAction SilentlyContinue)
    Write-Host "  rung trace after fresh install: $($t1 -join ' ')"
    Check "the new rung ran exactly once during the scan" (@($t1 | Where-Object { $_ -eq 'called' }).Count -eq 1)
    Check "and answered with a real table" (@($t1 | Where-Object { $_ -match '^rows=(\d+)$' -and [int]$Matches[1] -gt 20 }).Count -eq 1)
}
Check "no Tauri error marker in a plain install" ($l1 -notmatch '\[TAURI:ERROR\]')

# 2. A live interpreter from the managed venv must block a re-run.
$venvPy = Join-Path $env:USERPROFILE ".unsloth\studio\unsloth_studio\Scripts\python.exe"
Check "the managed venv interpreter exists" (Test-Path -LiteralPath $venvPy)
$live = Start-Process -FilePath $venvPy -ArgumentList '-c "import time;time.sleep(900)"' -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 3
Check "harness: the live venv interpreter is running" (-not $live.HasExited)
Remove-Item -LiteralPath $trace -ErrorAction SilentlyContinue
$c2 = Invoke-Install "logs/2-blocked.log"
$l2 = Get-Content -Raw "logs/2-blocked.log"
Write-Host "  re-run with a live venv interpreter exit=$c2"
Check "the re-run refuses to touch the live venv" ($c2 -ne 0 -and $l2 -match 'Unsloth Studio is using the managed Python environment')
Check "and names the live PID" ($l2 -match "PID $($live.Id)")
Stop-Process -Id $live.Id -Force -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "ParentProcessId=$($live.Id)" -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3

# 3. With nothing running, the update proceeds.
$c3 = Invoke-Install "logs/3-rerun.log"
Write-Host "  clean re-run exit=$c3"
Check "a clean re-run (update) succeeds" ($c3 -eq 0)
& (Join-Path $env:USERPROFILE ".unsloth\studio\unsloth_studio\Scripts\python.exe") -c "import unsloth_cli, studio; print('import ok')"
Check "the updated venv still imports the CLI" ($LASTEXITCODE -eq 0)
if ($fails) { Write-Host "$fails FAILED [$Label]"; exit 1 }
Write-Host "all passed [$Label]"
