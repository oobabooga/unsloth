# Drives the real New-StudioShortcuts from install.ps1 on a real Windows desktop, with a shell
# notification listener running, in one of two modes:
#   fallback: the native-type gate says no (a Dynamic Code Security host), so the catch runs the
#             Python child. Expect the child to be called with both links and to answer true, and
#             the listener to see SHCNE_UPDATEITEM for both links.
#   primary:  the type already exists, as on an ordinary host. Expect the child never to be called.
param([string]$InstallPs1, [string]$Mode, [string]$VenvPython, [string]$Work, [string]$HostPython)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Work | Out-Null
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($InstallPs1, [ref]$tokens, [ref]$errors)
if ($errors) { throw "parse errors" }
# Every function definition in the file; definitions have no side effects.
foreach ($fn in $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    if ($fn.Name -in @("Install-UnslothStudio")) { continue }
    Invoke-Expression $fn.Extent.Text
}
function substep { param($m, $c) Write-Host "  [substep] $m" }
function Write-StudioLine { param($m) Write-Host "  [line] $m" }
function Get-ManagedLlamaCppDir { return $null }
$script:GateCalls = 0
if ($Mode -eq "fallback") {
    function Test-StudioCanDefineNativeTypes { $script:GateCalls++; return $false }
} else {
    Add-Type -Namespace "" -Name UnslothShellIconRefresh -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("shell32.dll")]
public static extern void SHChangeNotify(int wEventId, uint uFlags, string dwItem1, System.IntPtr dwItem2);
"@ -ErrorAction SilentlyContinue
    if (-not ("UnslothShellIconRefresh" -as [type])) { throw "could not predefine the type" }
    function Test-StudioCanDefineNativeTypes { $script:GateCalls++; return $true }
}
${function:Real-IconRefresh} = ${function:Invoke-StudioPythonShellIconRefresh}
$script:IconCalls = @()
function Invoke-StudioPythonShellIconRefresh {
    param([string[]]$Paths = @(), [string]$Exe = "")
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Real-IconRefresh -Paths $Paths -Exe $Exe
    $script:IconCalls += , @{ Paths = ($Paths -join " | "); Exe = $Exe; Result = $r; Ms = $sw.ElapsedMilliseconds }
    return $r
}

$StudioHome = Join-Path $Work "home"; $StudioDataDir = Join-Path $Work "data"
New-Item -ItemType Directory -Force -Path $StudioHome, $StudioDataDir | Out-Null
$StudioRedirectMode = $null
$script:UnslothCliTrampoline = "C:\fake\unsloth.exe"

$desktop = [Environment]::GetFolderPath("Desktop")
$startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$links = @((Join-Path $desktop "Unsloth Studio.lnk"), (Join-Path $startMenu "Unsloth Studio.lnk"))
foreach ($l in $links) { Remove-Item -LiteralPath $l -Force -ErrorAction SilentlyContinue }
$log = Join-Path $Work "listener.log"; $stop = Join-Path $Work "listener.stop"
Remove-Item -LiteralPath $log, $stop -Force -ErrorAction SilentlyContinue
$listener = Start-Process -FilePath $HostPython -ArgumentList @(
    "`"$PSScriptRoot\listener.py`"", "`"$log`"", "`"$stop`"", "300", "`"$desktop`"", "`"$startMenu`"") -PassThru -WindowStyle Hidden
for ($i = 0; $i -lt 100 -and -not ((Test-Path $log) -and ((Get-Content $log -Raw) -match "READY")); $i++) { Start-Sleep -Milliseconds 100 }
Get-Content $log | ForEach-Object { Write-Host "  listener: $_" }

$result = @{}
foreach ($pass in @("first", "again")) {
    $script:IconCalls = @()
    $t0 = Get-Date
    New-StudioShortcuts -ManagedPythonPath $VenvPython
    $result[$pass] = @{ Calls = $script:IconCalls; Seconds = ((Get-Date) - $t0).TotalSeconds }
    Start-Sleep -Seconds 3
}
New-Item -ItemType File -Path $stop -Force | Out-Null
$listener.WaitForExit(20000) | Out-Null
$events = @(Get-Content $log | Where-Object { $_ -match "`tEVENT`t" })

$failures = 0
function Check($name, $cond) { if ($cond) { Write-Host "  PASS  $name" } else { Write-Host "  FAIL  $name"; $script:failures++ } }
Write-Host "mode=$Mode  PS=$($PSVersionTable.PSVersion)  gate calls=$script:GateCalls"
foreach ($pass in @("first", "again")) {
    Write-Host "pass=$pass took $([math]::Round($result[$pass].Seconds,2)) s"
    foreach ($c in $result[$pass].Calls) { Write-Host "  child call: result=$($c.Result) ms=$($c.Ms) exe=$($c.Exe) paths=$($c.Paths)" }
}
$events | ForEach-Object { Write-Host "  event: $_" }
foreach ($l in $links) { Check "the shortcut exists: $l" (Test-Path -LiteralPath $l) }
$updates = @($events | Where-Object { $_ -match "`t0x00002000`t" } | ForEach-Object { ($_ -split "`t")[3].ToLowerInvariant() })
if ($Mode -eq "fallback") {
    foreach ($pass in @("first", "again")) {
        $c = $result[$pass].Calls
        Check "$pass pass: the child was called once" ($c.Count -eq 1)
        Check "$pass pass: the child answered true" ($c.Count -eq 1 -and $c[0].Result -eq $true)
        Check "$pass pass: it got both links" ($c.Count -eq 1 -and $c[0].Paths -eq ($links -join " | "))
        Check "$pass pass: the child got the managed interpreter" ($c.Count -eq 1 -and $c[0].Exe -eq (Resolve-Path -LiteralPath $VenvPython).Path)
        Check "$pass pass: under 10 s" ($c.Count -eq 1 -and $c[0].Ms -lt 10000)
    }
} else {
    Check "the child was never called" ($result["first"].Calls.Count -eq 0 -and $result["again"].Calls.Count -eq 0)
}
foreach ($l in $links) {
    $n = @($updates | Where-Object { $_ -eq $l.ToLowerInvariant() }).Count
    Check "the listener saw SHCNE_UPDATEITEM for $l on both passes ($n)" ($n -ge 2)
}
if ($failures) { exit 1 }
