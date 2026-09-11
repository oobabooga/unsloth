# AST and deterministic-routing checks: main vs PR #10765 install.ps1. Exit code = failed checks.
# Runs under Windows PowerShell 5.1, pwsh 7 on Windows, and pwsh 7 on Linux (content checks on
# Windows only, since the helper's spaced branch and 8.3 names exist only there).
param([Parameter(Mandatory = $true)][string]$WorkRoot)
$ErrorActionPreference = 'Stop'
$script:bad = 0
$rows = New-Object System.Collections.ArrayList
function Check($name, $ok, $detail = '') {
    [void]$rows.Add([pscustomobject]@{ Check = $name; Ok = [bool]$ok; Detail = "$detail" })
    if (-not $ok) { $script:bad++ }
}
$onWindows = -not ($IsLinux -or $IsMacOS)
$T = [System.Management.Automation.Language.FunctionDefinitionAst]
$HelperName = 'New-UnslothTorchOverridesFile'

function Parse($path) {
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
    return @{ Ast = $ast; Errors = @($errors); Text = [System.IO.File]::ReadAllText($path) }
}
$m = Parse (Join-Path (Join-Path $WorkRoot 'main') 'install.ps1')
$p = Parse (Join-Path (Join-Path $WorkRoot 'pr') 'install.ps1')
Check 'main parses without errors' ($m.Errors.Count -eq 0) $m.Errors.Count
Check 'PR parses without errors' ($p.Errors.Count -eq 0) $p.Errors.Count

$mf = @($m.Ast.FindAll({ param($n) $n -is $T }, $true))
$pf = @($p.Ast.FindAll({ param($n) $n -is $T }, $true))
$mNames = @($mf | ForEach-Object { $_.Name }) -join ','
$pNames = @($pf | ForEach-Object { $_.Name }) -join ','
Check 'same functions, same order' ($mNames -eq $pNames) "$($mf.Count) vs $($pf.Count)"

# Every function whose extent does not contain the helper is textually identical.
$mh = @($mf | Where-Object { $_.Name -eq $HelperName })
$ph = @($pf | Where-Object { $_.Name -eq $HelperName })
Check 'exactly one helper in each' (($mh.Count -eq 1) -and ($ph.Count -eq 1)) "$($mh.Count)/$($ph.Count)"
$differ = @()
for ($i = 0; $i -lt [Math]::Min($mf.Count, $pf.Count); $i++) {
    if ($mf[$i].Extent.Text -ne $pf[$i].Extent.Text) { $differ += $pf[$i].Name }
}
$containsHelper = @($pf | Where-Object { $_.Extent.StartOffset -le $ph[0].Extent.StartOffset -and $_.Extent.EndOffset -ge $ph[0].Extent.EndOffset } | ForEach-Object { $_.Name })
Check 'only the helper and the functions enclosing it differ' ((@($differ | Where-Object { $containsHelper -notcontains $_ })).Count -eq 0) ("differ: " + ($differ -join ','))

# Helper body: main's statements plus one IfStatement inserted right before the final `return $f`.
$ms = @($mh[0].Body.EndBlock.Statements)
$ps = @($ph[0].Body.EndBlock.Statements)
Check 'helper gains exactly one statement' ($ps.Count -eq $ms.Count + 1) "$($ms.Count) -> $($ps.Count)"
$same = $true
for ($i = 0; $i -lt $ms.Count - 1; $i++) { if ($ms[$i].Extent.Text -ne $ps[$i].Extent.Text) { $same = $false } }
Check 'all earlier helper statements unchanged' $same
Check 'final statement is `return $f` in both' (($ms[-1].Extent.Text -eq 'return $f') -and ($ps[-1].Extent.Text -eq 'return $f'))
$ins = $ps[-2]
Check 'inserted statement is an if on $f.Contains(" ")' (($ins -is [System.Management.Automation.Language.IfStatementAst]) -and ($ins.Clauses[0].Item1.Extent.Text -eq '$f.Contains(" ")') -and ($null -eq $ins.ElseClause))
$cmds = @($ins.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true) | ForEach-Object { $_.GetCommandName() } | Sort-Object -Unique)
Check 'inserted block runs no external program (commands: New-Object, Remove-Item, substep)' ((($cmds -join ',') -eq 'New-Object,Remove-Item,substep')) ($cmds -join ',')
$assigns = @($ins.FindAll({ param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] }, $true) | ForEach-Object { $_.Left.Extent.Text } | Sort-Object -Unique)
Check 'inserted block assigns only $f and $short (no $script:/$env:/global state)' ((($assigns -join ',') -eq '$f,$short')) ($assigns -join ',')
$rets = @($ins.FindAll({ param($n) $n -is [System.Management.Automation.Language.ReturnStatementAst] }, $true) | ForEach-Object { $_.Extent.Text })
Check 'inserted block exits only via `return $null`' ((($rets -join ',') -eq 'return $null')) ($rets -join ',')

# Call sites: exactly two, identical text, same order, in both.
$callsM = @($m.Ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] -and $n.GetCommandName() -eq $HelperName }, $true))
$callsP = @($p.Ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] -and $n.GetCommandName() -eq $HelperName }, $true))
Check 'two call sites in both, textually identical' (($callsM.Count -eq 2) -and ($callsP.Count -eq 2) -and ((@($callsM | ForEach-Object { $_.Parent.Extent.Text }) -join '|') -eq (@($callsP | ForEach-Object { $_.Parent.Extent.Text }) -join '|')))
# Each call sits in the elseif/else of an if whose first clause tests $SkipTorch.
$gated = $true
foreach ($c in $callsP) {
    $n = $c
    while ($n -and -not ($n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Clauses[0].Item1.Extent.Text -eq '$SkipTorch')) { $n = $n.Parent }
    if (-not $n) { $gated = $false; continue }
    if ($n.Clauses[0].Item2.Extent.StartOffset -le $c.Extent.StartOffset -and $n.Clauses[0].Item2.Extent.EndOffset -ge $c.Extent.EndOffset) { $gated = $false }
}
Check 'both call sites are outside the `if ($SkipTorch)` (no-torch) branch' $gated

# --shortcuts-only (what `unsloth studio update` and Desktop run) returns before the helper exists.
$so = @($p.Ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Clauses[0].Item1.Extent.Text -eq '$ShortcutsOnly' }, $true))
Check 'one `if ($ShortcutsOnly)` block' ($so.Count -eq 1) $so.Count
if ($so.Count -eq 1) {
    $tryAst = @($so[0].Clauses[0].Item2.Statements | Where-Object { $_ -is [System.Management.Automation.Language.TryStatementAst] })
    $last = if ($tryAst.Count) { @($tryAst[0].Body.Statements)[-1] } else { $null }
    Check '--shortcuts-only body ends in an unconditional return' (($tryAst.Count -eq 1) -and ($last -is [System.Management.Automation.Language.ReturnStatementAst]) -and (@($so[0].Clauses[0].Item2.Statements).Count -eq 1))
    Check '--shortcuts-only block precedes the helper definition and both call sites' (($so[0].Extent.EndOffset -lt $ph[0].Extent.StartOffset) -and ($so[0].Extent.EndOffset -lt $callsP[0].Extent.StartOffset))
    $soTextEqual = $false
    $soM = @($m.Ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and $n.Clauses[0].Item1.Extent.Text -eq '$ShortcutsOnly' }, $true))
    if ($soM.Count -eq 1) { $soTextEqual = ($soM[0].Extent.Text -eq $so[0].Extent.Text) }
    Check '--shortcuts-only block identical to main' $soTextEqual
}

# Byte level: PR text minus the inserted if-block is exactly main.
$mainNorm = $m.Text
# The inserted statement also carries its leading comment line and indentation; strip that span too.
$lineStart = $p.Text.LastIndexOf("`n", [Math]::Max(0, $ins.Extent.StartOffset - 1))
$commentStart = $p.Text.LastIndexOf("`n", [Math]::Max(0, $lineStart - 1))
$stripped = $p.Text.Remove($commentStart, ($ins.Extent.EndOffset - $commentStart))
Check 'PR install.ps1 == main + only that block (byte comparison)' ($stripped -eq $mainNorm)

# ---- deterministic content: main vs PR on the same inputs, across torch flavours ----
if ($onWindows) {
    Invoke-Expression ($mh[0].Extent.Text -replace "function $HelperName", 'function Ov-main')
    Invoke-Expression ($ph[0].Extent.Text -replace "function $HelperName", 'function Ov-pr')
    $SkipTorch = $false
    $script:TorchOverridesFile = $null
    function substep { param($Message, $Color) }
    $fso = New-Object -ComObject Scripting.FileSystemObject
    $flavours = @{
        cpu = @('torch==2.11.0+cpu', 'torchvision==0.26.0+cpu', 'torchaudio==2.11.0+cpu')
        cu130 = @('torch==2.11.0+cu130', 'torchvision==0.26.0+cu130', 'torchaudio==2.11.0+cu130')
        rocm = @('torch==2.11.0+rocm7.2', 'torchvision==0.26.0+rocm7.2')
        xpu = @('torch==2.11.0+xpu', 'torchvision==0.26.0+xpu', 'torchaudio==2.11.0+xpu')
        none = @()
    }
    $plain = Join-Path $WorkRoot 'plain-tmp'
    $spaced = 'C:\Users\Jane Roe\AppData\Local\Temp'
    New-Item -ItemType Directory -Force -Path $plain, $spaced | Out-Null
    foreach ($fl in @('cpu', 'cu130', 'rocm', 'xpu', 'none')) {
        $fake = Join-Path $WorkRoot "fake-$fl.cmd"
        $body = @('@echo off') + @($flavours[$fl] | ForEach-Object { "echo $_" })
        Set-Content -LiteralPath $fake -Value $body -Encoding ascii
        foreach ($dir in @($plain, $spaced)) {
            $saved = $env:TMP; $env:TMP = $dir
            Remove-Item Env:UV_OVERRIDE -ErrorAction SilentlyContinue
            try {
                $a = @(Ov-main -PythonExe $fake); $a = if ($a.Count) { $a[-1] } else { $null }
                $b = @(Ov-pr -PythonExe $fake); $b = if ($b.Count) { $b[-1] } else { $null }
            } finally { $env:TMP = $saved }
            $tag = "$fl / $(if ($dir -eq $plain) { 'plain' } else { 'spaced' })"
            if ($fl -eq 'none') {
                Check "$tag : both return null when no torch is installed" (($null -eq $a) -and ($null -eq $b))
                continue
            }
            $ba = [System.IO.File]::ReadAllBytes($a); $bb = [System.IO.File]::ReadAllBytes($b)
            Check "$tag : identical pin bytes" ([Convert]::ToBase64String($ba) -eq [Convert]::ToBase64String($bb)) "$($ba.Length) bytes"
            if ($dir -eq $plain) {
                Check "$tag : identical path shape (same dir, long form)" ((Split-Path -Parent $a) -eq (Split-Path -Parent $b)) $b
            } else {
                Check "$tag : PR path is the short form of a file in the same dir" ((-not $b.Contains(' ')) -and ($fso.GetFolder((Split-Path -Parent $a)).ShortPath -eq (Split-Path -Parent $b))) $b
            }
            Remove-Item -LiteralPath $a, $b -Force -ErrorAction SilentlyContinue
        }
    }
    $SkipTorch = $true
    Check 'no-torch: both return null' (($null -eq (Ov-main -PythonExe (Join-Path $WorkRoot 'fake-cpu.cmd'))) -and ($null -eq (Ov-pr -PythonExe (Join-Path $WorkRoot 'fake-cpu.cmd'))))
    $SkipTorch = $false
}

$rows | Format-Table -AutoSize -Wrap | Out-String -Width 250 | Write-Host
$md = @("### routing: $($PSVersionTable.PSEdition) $($PSVersionTable.PSVersion) on $(if ($onWindows) { 'Windows' } else { 'Linux' })", '', '| check | ok | detail |', '| --- | --- | --- |')
foreach ($r in $rows) { $md += "| $($r.Check) | $(if ($r.Ok) { 'PASS' } else { 'FAIL' }) | $($r.Detail) |" }
if ($env:GITHUB_STEP_SUMMARY) { $md | Out-File -Append -FilePath $env:GITHUB_STEP_SUMMARY -Encoding utf8 }
Write-Host "failed checks: $script:bad"
exit $script:bad
