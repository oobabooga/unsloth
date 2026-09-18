param([Parameter(Mandatory = $true)][string]$Repo, [Parameter(Mandatory = $true)][string]$Label)
$ErrorActionPreference = "Stop"
$fails = 0
function Check($name, $cond) { if ($cond) { Write-Host "  PASS  [$Label] $name" } else { Write-Host "  FAIL  [$Label] $name"; $script:fails++ } }
Write-Host "== $Label  PS $($PSVersionTable.PSVersion) $($PSVersionTable.PSEdition)  repo=$Repo"
$tok = $null; $err = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $Repo "install.ps1"), [ref]$tok, [ref]$err)
$names = @("Get-StudioEarlyPython", "Invoke-StudioEarlyPython", "Invoke-StudioEarlyPythonScript",
    "Get-StudioPythonProcessImageTable", "Get-StudioProcessImagePath")
$present = @{}
foreach ($n in $names) {
    $f = $ast.FindAll({ param($x) $x -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $x.Name -eq $n }, $true)
    if ($f.Count) { Invoke-Expression $f[0].Extent.Text; $present[$n] = $true } else { Write-Host "  (absent: $n)" }
}
function Write-StudioLine { param([string]$Line, [string]$ForegroundColor = "") }
function Initialize-StudioProcessImageNativeType { return $false }
function Reset-State {
    $script:StudioEarlyPythonProbed = $false; $script:StudioEarlyPython = $null
    $script:StudioEarlyPythonProbedWithoutVenv = $false
    $script:StudioPythonProcessImageTable = $null; $script:StudioPythonProcessImageProbed = $false
    $script:StudioProcessImageTable = $null; $script:StudioProcessImageWarned = $true
}
Reset-State
$py = Get-StudioEarlyPython
Write-Host "  interpreter: $py"
$arch = & $py -c "import struct;print(struct.calcsize('P')*8)"
Write-Host "  python bits: $arch"

# A managed-looking venv with a live interpreter inside it.
$venv = Join-Path $env:RUNNER_TEMP ("fakevenv-" + $Label)
if (-not (Test-Path $venv)) { & $py -m venv $venv | Out-Null }
$venvPy = Join-Path $venv "Scripts\python.exe"
$live = Start-Process -FilePath $venvPy -ArgumentList '-c "import time;time.sleep(600)"' -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2
Check "harness: the live venv interpreter is running" (-not $live.HasExited)
try {
    if ($present["Get-StudioPythonProcessImageTable"]) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $table = Get-StudioPythonProcessImageTable
        $sw.Stop()
        Write-Host "  table rows: $($table.Count) in $($sw.ElapsedMilliseconds) ms"
        Check "the real ctypes probe returns a table" ($table -and $table.Count -gt 20)
        Check "it names this shell's own image" ($table[$PID] -and ($table[$PID] -ieq (Get-Process -Id $PID).Path))
        Check "it names the live venv interpreter" ($table[$live.Id] -ieq $venvPy)
        # Parity with Get-Process where Get-Process can answer.
        $agree = 0; $disagree = @(); $gpBlind = 0; $gpBlindCovered = 0
        foreach ($p in Get-Process) {
            $gp = $null; try { $gp = $p.Path } catch {}
            $mine = $table[[int]$p.Id]
            if ($gp) {
                if ($mine) { if ($mine -ieq $gp) { $agree++ } else { $disagree += "$($p.Id) gp=$gp ct=$mine" } }
            } else {
                $gpBlind++
                if ($mine) { $gpBlindCovered++ }
            }
        }
        Write-Host "  agree=$agree disagree=$($disagree.Count) getprocess_blind=$gpBlind covered_by_ctypes=$gpBlindCovered"
        $disagree | Select-Object -First 5 | ForEach-Object { Write-Host "    $_" }
        Check "no disagreement with Get-Process where both answer" ($disagree.Count -eq 0)
        # Handle leak: run the exact probe with a handle count before and after.
        $captured = $null
        $saved = ${function:Invoke-StudioEarlyPythonScript}
        function Invoke-StudioEarlyPythonScript { param($Exe, $Script, $ScriptArgs, $TimeoutMs) $script:captured = $Script; return "" }
        $script:StudioPythonProcessImageProbed = $false
        $null = Get-StudioPythonProcessImageTable
        ${function:Invoke-StudioEarlyPythonScript} = $saved
        $pre = "import ctypes as _c,sys as _s" + [char]10 + "_k=_c.WinDLL('kernel32');_h=_c.c_ulong();_k.GetProcessHandleCount(_c.c_void_p(-1),_c.byref(_h));_s.stderr.write('PRE=%d\n'%_h.value)" + [char]10
        $post = [char]10 + "_k.GetProcessHandleCount(_c.c_void_p(-1),_c.byref(_h));_s.stderr.write('POST=%d\n'%_h.value)"
        $leakFile = Join-Path $env:RUNNER_TEMP "leak.py"
        Set-Content -LiteralPath $leakFile -Value ($pre + $script:captured + $post) -Encoding UTF8
        $errFile = Join-Path $env:RUNNER_TEMP "leak.err"
        $outFile = Join-Path $env:RUNNER_TEMP "leak.out"
        Start-Process -FilePath $py -ArgumentList "-I -S `"$leakFile`"" -Wait -NoNewWindow -RedirectStandardError $errFile -RedirectStandardOutput $outFile
        $errOut = @(Get-Content -LiteralPath $errFile)
        $preN = [int](($errOut | Where-Object { $_ -match '^PRE=' }) -replace 'PRE=', '')
        $postN = [int](($errOut | Where-Object { $_ -match '^POST=' }) -replace 'POST=', '')
        Write-Host "  handles pre=$preN post=$postN"
        Check "the probe does not leak one handle per process" (($postN - $preN) -lt 20 -and $postN -gt 0)
    }

    # The scenario the rung exists for: Get-Process cannot read .Path and WMI is broken.
    Reset-State
    function Get-Process { param($Id, $ErrorAction) return [pscustomobject]@{ Id = $Id; Path = $null } }
    function Get-CimInstance { param($ClassName, $ErrorAction) throw "WMI repository is broken" }
    $answer = Get-StudioProcessImagePath -ProcessId $live.Id
    Write-Host "  Get-StudioProcessImagePath (no .Path, no WMI) -> '$answer'"
    if ($present["Get-StudioPythonProcessImageTable"]) {
        Check "after: the live venv interpreter is still found" ($answer -ieq $venvPy)
    } else {
        Check "before: nothing finds it (the gap this PR closes)" ([string]::IsNullOrWhiteSpace($answer))
    }
    # And with WMI healthy, the answer is the same either way.
    Remove-Item Function:Get-CimInstance
    Reset-State
    $answer2 = Get-StudioProcessImagePath -ProcessId $live.Id
    Check "with WMI healthy the live venv interpreter is found" ($answer2 -ieq $venvPy)
    Remove-Item Function:Get-Process
} finally {
    Stop-Process -Id $live.Id -Force -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$($live.Id)" -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
if ($fails) { Write-Host "$fails FAILED"; exit 1 }
Write-Host "all passed [$Label]"
