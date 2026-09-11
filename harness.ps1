# Exercises New-UnslothTorchOverridesFile from main and from PR #10765 against a real uv.
# Runs under Windows PowerShell 5.1 and pwsh 7. Exit code = number of failed expectations.
param(
    [Parameter(Mandatory = $true)][string]$WorkRoot,
    [Parameter(Mandatory = $true)][string]$Short83,
    [Parameter(Mandatory = $true)][string]$Uv
)
$ErrorActionPreference = 'Stop'
$results = New-Object System.Collections.ArrayList
$script:bad = 0
function Rec($variant, $scenario, $check, $ok, $detail) {
    [void]$results.Add([pscustomobject]@{ Variant = $variant; Scenario = $scenario; Check = $check; Ok = [bool]$ok; Detail = "$detail" })
    if (-not $ok) { $script:bad++ }
}

$venvPy = Join-Path $WorkRoot 'venv\Scripts\python.exe'
$fakePy = Join-Path $WorkRoot 'fakepython.cmd'
Set-Content -LiteralPath $fakePy -Value @('@echo torch==2.11.0+cpu', '@echo torchvision==0.26.0+cpu', '@echo torchaudio==2.11.0+cpu') -Encoding ascii
$fso = New-Object -ComObject Scripting.FileSystemObject

foreach ($v in 'main', 'pr') {
    $path = Join-Path $WorkRoot "$v\install.ps1"
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
    if ($errors) { throw "parse errors in $path" }
    $fn = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'New-UnslothTorchOverridesFile' }, $true)
    if ($fn.Count -ne 1) { throw "expected one helper in $path, found $($fn.Count)" }
    Invoke-Expression ($fn[0].Extent.Text -replace 'function New-UnslothTorchOverridesFile', "function New-Overrides_$v")
}
$SkipTorch = $false
$script:TorchOverridesFile = $null
$script:substepCalls = New-Object System.Collections.ArrayList
function substep { param($Message, $Color) [void]$script:substepCalls.Add($Message) }

function Invoke-Helper($variant) {
    $script:substepCalls.Clear()
    $r = @(& "New-Overrides_$variant" -PythonExe $fakePy)
    if ($r.Count -eq 0) { return $null }
    return $r[-1]
}

function Invoke-UvDry($ovPath, $cwd) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    Push-Location -LiteralPath $cwd
    [Environment]::CurrentDirectory = $cwd
    try {
        $a = @('pip', 'install', '--dry-run', '--python', $venvPy)
        if ($ovPath) { $a += @('--overrides', $ovPath) }
        $a += @('requests')
        $out = @(& $Uv @a 2>&1 | ForEach-Object { "$_" })
        $rc = $LASTEXITCODE
    } finally {
        Pop-Location
        [Environment]::CurrentDirectory = (Get-Location).ProviderPath
        $ErrorActionPreference = $prev
    }
    return [pscustomobject]@{ Rc = $rc; Out = ($out -join "`n") }
}

function Get-PinFiles($dir) {
    @(Get-ChildItem -LiteralPath $dir -File -ErrorAction SilentlyContinue | Where-Object {
        (Get-Content -LiteralPath $_.FullName -TotalCount 1 -ErrorAction SilentlyContinue) -match '^torch=='
    })
}

function HasSpace($s) { return ($null -ne $s) -and ([string]$s).Contains(' ') }

$acute = [char]0x00E9
$plainTmp = Join-Path $WorkRoot 'plain-tmp'
$spacedTmp = 'C:\Users\John Doe\AppData\Local\Temp'
$acuteTmp = "C:\Users\Jos$acute Doe\AppData\Local\Temp"
$callerDir = 'C:\Users\John Doe\proj'
foreach ($d in @($plainTmp, $spacedTmp, $acuteTmp, $callerDir)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
Write-Host "short form of '$spacedTmp': $($fso.GetFolder($spacedTmp).ShortPath)"
Write-Host "short form of acute dir: $($fso.GetFolder($acuteTmp).ShortPath)"
Write-Host "PowerShell $($PSVersionTable.PSVersion) ($($PSVersionTable.PSEdition)), 8.3 $Short83, uv $(& $Uv --version)"

# ---- TMP-driven scenarios -------------------------------------------------------------
$tmpScenarios = @(
    @{ Name = 'plain-tmp'; Dir = $plainTmp; Spaced = $false },
    @{ Name = 'spaced-tmp'; Dir = $spacedTmp; Spaced = $true },
    @{ Name = 'spaced-nonascii-tmp'; Dir = $acuteTmp; Spaced = $true }
)
if ($Short83 -eq 'on') {
    $tmpScenarios += @{ Name = 'already-short-tmp'; Dir = $fso.GetFolder($spacedTmp).ShortPath; Spaced = $false }
}

foreach ($variant in 'main', 'pr') {
    foreach ($sc in $tmpScenarios) {
        $n = $sc.Name
        Get-PinFiles $sc.Dir | Remove-Item -Force
        Remove-Item Env:UV_OVERRIDE -ErrorAction SilentlyContinue
        $savedTmp = $env:TMP
        $env:TMP = $sc.Dir
        try { $p = Invoke-Helper $variant } finally { $env:TMP = $savedTmp }
        $warned = $script:substepCalls.Count
        Write-Host "[$variant/$n] returned: $p  (warnings: $warned)"
        $uvr = $null
        if ($p) { $uvr = Invoke-UvDry $p $WorkRoot }
        if ($uvr) { Write-Host "[$variant/$n] uv rc=$($uvr.Rc): $(($uvr.Out -split "`n" | Select-Object -Last 2) -join ' | ')" }

        if (-not $sc.Spaced) {
            Rec $variant $n 'returns a file' ($p -and (Test-Path -LiteralPath $p)) $p
            Rec $variant $n 'path unchanged: lives in TMP as given' ($p -and ((Split-Path -Parent $p) -eq $sc.Dir.TrimEnd('\'))) $p
            Rec $variant $n 'uv accepts --overrides' ($uvr -and $uvr.Rc -eq 0) $(if ($uvr) { $uvr.Rc })
            Rec $variant $n 'no warning' ($warned -eq 0) $warned
        } elseif ($variant -eq 'main') {
            Rec $variant $n 'control: main hands uv a spaced path' (HasSpace $p) $p
            Rec $variant $n 'control: uv fails with File not found' ($uvr -and $uvr.Rc -ne 0 -and $uvr.Out -match 'File not found') $(if ($uvr) { $uvr.Rc })
        } elseif ($Short83 -eq 'on') {
            $only = @(Get-PinFiles $sc.Dir)
            Rec $variant $n 'returns a path without a space' ($p -and -not (HasSpace $p)) $p
            Rec $variant $n 'short path is the single file written in TMP' (($only.Count -eq 1) -and ($fso.GetFile($only[0].FullName).ShortPath -eq $p)) "files=$($only.Count)"
            Rec $variant $n 'uv accepts --overrides' ($uvr -and $uvr.Rc -eq 0) $(if ($uvr) { $uvr.Rc })
            Rec $variant $n 'no warning' ($warned -eq 0) $warned
        } else {
            Rec $variant $n 'no 8.3: returns null' ($null -eq $p) $p
            Rec $variant $n 'no 8.3: warns' ($warned -eq 2) $warned
            Rec $variant $n 'no 8.3: leaves no pin file behind' (@(Get-PinFiles $sc.Dir).Count -eq 0) (@(Get-PinFiles $sc.Dir).Count)
        }
        # The installer's caller cleanup: Remove-Item on whatever came back.
        if ($p) {
            Remove-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue
            Rec $variant $n 'caller cleanup removes the file' (@(Get-PinFiles $sc.Dir).Count -eq 0) (@(Get-PinFiles $sc.Dir).Count)
        }
    }
}

# ---- caller UV_OVERRIDE, relative entry, spaced working directory ------------------------
[System.IO.File]::WriteAllText((Join-Path $callerDir 'over.txt'), "-r nested.txt`ncertifi==2024.2.2`ntorch==1.0`n")
[System.IO.File]::WriteAllText((Join-Path $callerDir 'nested.txt'), "idna==3.6`n")
foreach ($variant in 'main', 'pr') {
    $n = 'caller-override-spaced-cwd'
    Get-ChildItem -LiteralPath $callerDir -Filter 'unsloth-torch-overrides-*' | Remove-Item -Force
    Push-Location -LiteralPath $callerDir
    [Environment]::CurrentDirectory = $callerDir
    $env:UV_OVERRIDE = 'over.txt'
    try { $p = Invoke-Helper $variant } finally {
        Pop-Location
        [Environment]::CurrentDirectory = (Get-Location).ProviderPath
    }
    $warned = $script:substepCalls.Count
    Write-Host "[$variant/$n] returned: $p  (warnings: $warned)"
    if ($variant -eq 'main') {
        $uvr = Invoke-UvDry $p $callerDir
        Rec $variant $n 'control: merged file sits in the spaced caller dir' (HasSpace $p) $p
        Rec $variant $n 'control: uv fails with File not found' ($uvr.Rc -ne 0 -and $uvr.Out -match 'File not found') $uvr.Rc
    } elseif ($Short83 -eq 'on') {
        $uvr = Invoke-UvDry $p $callerDir
        Write-Host "[$variant/$n] uv (installer cwd, UV_OVERRIDE set) rc=$($uvr.Rc)"
        Write-Host $uvr.Out
        Rec $variant $n 'returns a path without a space' ($p -and -not (HasSpace $p)) $p
        Rec $variant $n 'uv applies the merged pins and the relative -r include' ($uvr.Rc -eq 0 -and $uvr.Out -match 'certifi==2024\.2\.2' -and $uvr.Out -match 'idna==3\.6') $uvr.Rc
        Remove-Item Env:UV_OVERRIDE
        $uvr2 = Invoke-UvDry $p $WorkRoot
        Rec $variant $n 'relative include resolves from the file dir, not the cwd' ($uvr2.Rc -eq 0 -and $uvr2.Out -match 'idna==3\.6') $uvr2.Rc
    } else {
        Rec $variant $n 'no 8.3: returns null' ($null -eq $p) $p
        Rec $variant $n 'no 8.3: leaves no merged file behind' (@(Get-ChildItem -LiteralPath $callerDir -Filter 'unsloth-torch-overrides-*').Count -eq 0) ''
        # The installer then runs uv without --overrides, so uv reads UV_OVERRIDE itself.
        $uvr = Invoke-UvDry $null $callerDir
        Rec $variant $n "no 8.3: fallback still honors the caller's UV_OVERRIDE" ($uvr.Rc -eq 0 -and $uvr.Out -match 'certifi==2024\.2\.2' -and $uvr.Out -match 'idna==3\.6') $uvr.Rc
    }
    if ($p) { Remove-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue }
    Remove-Item Env:UV_OVERRIDE -ErrorAction SilentlyContinue
}

Write-Host ''
$results | Format-Table -AutoSize -Wrap | Out-String -Width 250 | Write-Host
$md = @("### helper matrix: $($PSVersionTable.PSEdition) $($PSVersionTable.PSVersion), 8.3 $Short83", '', '| variant | scenario | check | ok | detail |', '| --- | --- | --- | --- | --- |')
foreach ($r in $results) { $md += "| $($r.Variant) | $($r.Scenario) | $($r.Check) | $(if ($r.Ok) { 'PASS' } else { 'FAIL' }) | $($r.Detail) |" }
if ($env:GITHUB_STEP_SUMMARY) { $md | Out-File -Append -FilePath $env:GITHUB_STEP_SUMMARY -Encoding utf8 }
Write-Host "failed expectations: $script:bad"
exit $script:bad
