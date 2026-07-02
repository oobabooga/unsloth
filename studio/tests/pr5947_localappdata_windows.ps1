#Requires -Version 5.1
<#
    Live Windows test for the LOCALAPPDATA systemprofile guard added to
    studio/setup.ps1 by unslothai/unsloth#5947 (+ follow-up hardening).

    It does NOT run the whole installer. It extracts the actual guard block
    out of setup.ps1 and exercises it against real path shapes on a real
    Windows machine, under both Windows PowerShell 5.1 and PowerShell 7, and
    reproduces the genuine SYSTEM-token scenario via a scheduled task.

    Fails (exit 1) if any assertion fails.
#>

$ErrorActionPreference = 'Stop'
$script:fails = 0
$script:total = 0

function Check {
    param([string]$Name, [bool]$Cond, [string]$Detail = '')
    $script:total++
    if ($Cond) {
        Write-Host "[PASS] $Name" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $Name -- $Detail" -ForegroundColor Red
        $script:fails++
    }
}

# ── Extract the guard block from setup.ps1 ───────────────────────────────
$setupPath = Join-Path $PSScriptRoot '..\setup.ps1'
if (-not (Test-Path $setupPath)) { throw "setup.ps1 not found at $setupPath" }
$lines = Get-Content -LiteralPath $setupPath

$startMatch = $lines | Select-String -SimpleMatch 'Guard: LOCALAPPDATA must not point to systemprofile' | Select-Object -First 1
$endMatch   = $lines | Select-String -SimpleMatch 'Detect if running from pip install' | Select-Object -First 1
if (-not $startMatch -or -not $endMatch) { throw "Could not locate guard block markers in setup.ps1" }

$startIdx = $startMatch.LineNumber - 1           # 0-based, at the banner comment
$endIdx   = $endMatch.LineNumber - 2             # last guard line, before the marker
$guardBlock = ($lines[$startIdx..$endIdx] -join "`r`n")

Write-Host "=== Extracted guard block ($($endIdx - $startIdx + 1) lines) ===" -ForegroundColor Cyan
Write-Host $guardBlock
Write-Host "=== end guard block ===`n" -ForegroundColor Cyan

$SYSPROFILE = 'C:\Windows\*\config\systemprofile*'

# ── Run one case in a child of $shellExe with a prep line ────────────────
function Invoke-GuardCase {
    param([string]$ShellExe, [string]$Prep)
    $body = $Prep + "`r`n" + $guardBlock + "`r`n" + 'Write-Output ("RESULT=" + $env:LOCALAPPDATA)'
    $ps1 = Join-Path ([System.IO.Path]::GetTempPath()) ("guardcase_" + [guid]::NewGuid().ToString('N') + ".ps1")
    Set-Content -LiteralPath $ps1 -Value $body -Encoding UTF8
    try {
        $out = & $ShellExe -NoProfile -ExecutionPolicy Bypass -File $ps1 2>&1
        $code = $LASTEXITCODE
    } finally {
        Remove-Item $ps1 -Force -ErrorAction SilentlyContinue
    }
    $resLine = $out | Where-Object { $_ -match '^RESULT=' } | Select-Object -First 1
    $result = if ($resLine) { ($resLine -replace '^RESULT=', '') } else { $null }
    [pscustomobject]@{ Code = $code; Result = $result; Raw = ($out -join "`n") }
}

# ── SYSTEM-token repro via scheduled task ────────────────────────────────
function Test-SystemToken {
    param([string]$ShellExe)
    $work = Join-Path $env:RUNNER_TEMP ("guardsys_" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $work -Force | Out-Null
    $guardOnly  = Join-Path $work 'guard_only.ps1'
    $resultFile = Join-Path $work 'result.txt'
    $runner     = Join-Path $work 'system_runner.ps1'
    # guard_only.ps1: the raw guard, no env override -> runs against SYSTEM's real env
    Set-Content -LiteralPath $guardOnly -Value $guardBlock -Encoding UTF8
    $tmpl = @'
$ErrorActionPreference = 'Continue'
$p = Start-Process '__SHELL__' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','__GUARD__') -Wait -PassThru -WindowStyle Hidden
Set-Content -LiteralPath '__RESULT__' -Encoding ascii -Value @(
  "EXITCODE=$($p.ExitCode)",
  "WHOAMI=$(whoami)",
  "LOCALAPPDATA=$env:LOCALAPPDATA",
  "USERPROFILE=$env:USERPROFILE"
)
'@
    $runnerBody = $tmpl.Replace('__SHELL__', $ShellExe).Replace('__GUARD__', $guardOnly).Replace('__RESULT__', $resultFile)
    Set-Content -LiteralPath $runner -Value $runnerBody -Encoding UTF8

    $tn = "UnslothGuardSys_" + [guid]::NewGuid().ToString('N').Substring(0, 8)
    & schtasks /create /tn $tn /tr "$ShellExe -NoProfile -ExecutionPolicy Bypass -File `"$runner`"" /sc once /st 23:59 /ru SYSTEM /rl HIGHEST /f | Out-Null
    & schtasks /run /tn $tn | Out-Null
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline -and -not (Test-Path $resultFile)) { Start-Sleep -Milliseconds 500 }
    Start-Sleep -Seconds 1
    & schtasks /delete /tn $tn /f | Out-Null

    if (-not (Test-Path $resultFile)) { return $null }
    $data = @{}
    foreach ($ln in Get-Content $resultFile) {
        if ($ln -match '^([^=]+)=(.*)$') { $data[$matches[1]] = $matches[2] }
    }
    $data
}

# ── Cases exercised under both shells ────────────────────────────────────
$shells = @('powershell', 'pwsh')

foreach ($sh in $shells) {
    if (-not (Get-Command $sh -ErrorAction SilentlyContinue)) {
        Write-Host "[SKIP] $sh not available on this runner" -ForegroundColor Yellow
        continue
    }
    Write-Host "`n########## Shell: $sh ##########" -ForegroundColor Magenta

    # 1. Normal user path -> untouched no-op
    $normal = 'C:\Users\testuser\AppData\Local'
    $r = Invoke-GuardCase $sh "`$env:LOCALAPPDATA = '$normal'"
    Check "$sh / normal path is a no-op (exit 0)" ($r.Code -eq 0) "code=$($r.Code) raw=$($r.Raw)"
    Check "$sh / normal path preserved unchanged" ($r.Result -eq $normal) "got '$($r.Result)'"

    # 2. system32 systemprofile -> corrected away from systemprofile
    $sp = 'C:\Windows\system32\config\systemprofile\AppData\Local'
    $r = Invoke-GuardCase $sh "`$env:LOCALAPPDATA = '$sp'"
    Check "$sh / systemprofile input exits 0 (resolved)" ($r.Code -eq 0) "code=$($r.Code) raw=$($r.Raw)"
    Check "$sh / systemprofile input corrected to real path" (($r.Result) -and ($r.Result -notlike $SYSPROFILE)) "got '$($r.Result)'"

    # 3. SysWOW64 variant -> pattern still fires, corrected
    $wow = 'C:\Windows\SysWOW64\config\systemprofile\AppData\Local'
    $r = Invoke-GuardCase $sh "`$env:LOCALAPPDATA = '$wow'"
    Check "$sh / SysWOW64 systemprofile corrected" (($r.Code -eq 0) -and ($r.Result) -and ($r.Result -notlike $SYSPROFILE)) "code=$($r.Code) got '$($r.Result)'"

    # 4. Empty LOCALAPPDATA -> resolved to a real path
    $r = Invoke-GuardCase $sh "Remove-Item Env:LOCALAPPDATA -ErrorAction SilentlyContinue"
    Check "$sh / empty input resolved (exit 0, non-empty, not systemprofile)" (($r.Code -eq 0) -and ($r.Result) -and ($r.Result -notlike $SYSPROFILE)) "code=$($r.Code) got '$($r.Result)'"

    # 5. UNC-redirected AppData -> preserved (regression guard for redirected AppData)
    $unc = '\\netshare\redirect\AppData\Local'
    $r = Invoke-GuardCase $sh "`$env:LOCALAPPDATA = '$unc'"
    Check "$sh / UNC redirect preserved unchanged" (($r.Code -eq 0) -and ($r.Result -eq $unc)) "code=$($r.Code) got '$($r.Result)'"
}

# 6. Genuine SYSTEM token -> guard must exit 1 (both fallbacks resolve to systemprofile)
Write-Host "`n########## SYSTEM-token repro (scheduled task) ##########" -ForegroundColor Magenta
$sys = Test-SystemToken 'powershell'
if ($null -eq $sys) {
    Check "SYSTEM-token repro produced a result" $false "scheduled task did not write a result file within timeout"
} else {
    Write-Host ("SYSTEM run: whoami='{0}' exitcode='{1}' LOCALAPPDATA='{2}'" -f $sys['WHOAMI'], $sys['EXITCODE'], $sys['LOCALAPPDATA'])
    Check "SYSTEM task actually ran as SYSTEM" ($sys['WHOAMI'] -match 'system') "whoami='$($sys['WHOAMI'])'"
    Check "guard exits 1 under a real SYSTEM token" ($sys['EXITCODE'] -eq '1') "exitcode='$($sys['EXITCODE'])' LOCALAPPDATA='$($sys['LOCALAPPDATA'])'"
}

# ── Summary ──────────────────────────────────────────────────────────────
Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host ("RESULT: {0}/{1} checks passed" -f ($script:total - $script:fails), $script:total) -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
if ($script:fails -gt 0) { exit 1 } else { exit 0 }
