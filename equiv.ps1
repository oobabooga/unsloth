param([string]$Base, [string]$Pr, [string]$Py, [string]$Work)
$ErrorActionPreference = "Stop"
function Get-FnText($file, $name) { $t=$null;$e=$null; $a=[System.Management.Automation.Language.Parser]::ParseFile($file,[ref]$t,[ref]$e); $a.FindAll({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true)[0].Extent.Text }
Invoke-Expression ((Get-FnText $Base 'Invoke-StudioEarlyPython') -replace 'function Invoke-StudioEarlyPython', 'function Invoke-BaseResolve')
Invoke-Expression (Get-FnText $Pr 'Invoke-StudioEarlyPythonScript')
Invoke-Expression ((Get-FnText $Pr 'Invoke-StudioEarlyPython') -replace 'function Invoke-StudioEarlyPython', 'function Invoke-PrResolve')
New-Item -ItemType Directory -Force -Path $Work | Out-Null
$sep = [System.IO.Path]::DirectorySeparatorChar
$cases = [ordered]@{}
$cases['plain dir'] = (New-Item -ItemType Directory -Force -Path (Join-Path $Work 'plain')).FullName
$cases['dir with spaces'] = (New-Item -ItemType Directory -Force -Path (Join-Path $Work 'a b c')).FullName
$cases['non-ASCII dir'] = (New-Item -ItemType Directory -Force -Path (Join-Path $Work ('uni-' + [char]0xE9 + [char]0x4E2D))).FullName
$cases['trailing separator'] = (Join-Path $Work 'plain') + $sep
$cases['file'] = (New-Item -ItemType File -Force -Path (Join-Path $Work 'f.txt')).FullName
$cases['missing path'] = Join-Path $Work 'does-not-exist'
$cases['relative dot'] = '.'
$cases['dotdot component'] = Join-Path (Join-Path $Work 'plain') '..'
try { $null = New-Item -ItemType SymbolicLink -Force -Path (Join-Path $Work 'link') -Target (Join-Path $Work 'plain'); $cases['symlink'] = Join-Path $Work 'link' } catch { Write-Host "  (symlink not creatable here: $($_.Exception.Message))" }
try { $null = New-Item -ItemType SymbolicLink -Force -Path (Join-Path $Work 'loop') -Target (Join-Path $Work 'loop'); $cases['symlink loop'] = Join-Path $Work 'loop' } catch {}
$fails = 0; $n = 0
foreach ($k in $cases.Keys) {
    $p = $cases[$k]
    $b = Invoke-BaseResolve -Exe $Py -Path $p; $r = Invoke-PrResolve -Exe $Py -Path $p
    $n++; $same = ($null -eq $b -and $null -eq $r) -or ($b -ceq $r)
    if (-not $same) { $fails++ }
    Write-Host ("  {0}  {1,-20} base='{2}' pr='{3}'" -f ($(if ($same) {'SAME'} else {'DIFF'})), $k, $b, $r)
}
# Interpreter-level failures: missing exe, non-zero exit, hang.
$b = Invoke-BaseResolve -Exe (Join-Path $Work 'no-such-python') -Path $Work; $r = Invoke-PrResolve -Exe (Join-Path $Work 'no-such-python') -Path $Work
$n++; if (-not ($null -eq $b -and $null -eq $r)) { $fails++ }; Write-Host "  missing interpreter: base='$b' pr='$r'"
$b = Invoke-BaseResolve -Exe $Py -Path $Work -TimeoutMs 1; $r = Invoke-PrResolve -Exe $Py -Path $Work -TimeoutMs 1
$n++; if (-not ($null -eq $b -and $null -eq $r)) { $fails++ }; Write-Host "  1 ms timeout: base='$b' pr='$r'"
Write-Host "equivalence: $($n - $fails)/$n identical"
if ($fails) { exit 1 }
