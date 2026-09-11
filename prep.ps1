# Job setup: scripts and installers into C:\Users\Public\e2e, 8.3 names on, optional account.
param([string]$Installers = '', [string]$Account = '')   # Installers: "name=ref;name=ref"
$ErrorActionPreference = 'Stop'
$e2e = 'C:\Users\Public\e2e'
New-Item -ItemType Directory -Force -Path $e2e | Out-Null
Copy-Item (Join-Path $PSScriptRoot '*.ps1') $e2e
foreach ($pair in ($Installers -split ';' | Where-Object { $_ })) {
    $n, $ref = $pair -split '=', 2
    & curl.exe -fsSL -o (Join-Path $e2e "$n.ps1") "https://raw.githubusercontent.com/oobabooga/unsloth/$ref/install.ps1"
    if ($LASTEXITCODE) {
        & curl.exe -fsSL -o (Join-Path $e2e "$n.ps1") "https://raw.githubusercontent.com/unslothai/unsloth/$ref/install.ps1"
        if ($LASTEXITCODE) { throw "could not fetch install.ps1 at $ref" }
    }
    Write-Host "$n.ps1 <- $ref ($((Get-Item (Join-Path $e2e "$n.ps1")).Length) bytes)"
}
& fsutil 8dot3name set 0 | Out-Null
& icacls $e2e /grant '*S-1-1-0:(OI)(CI)M' | Out-Null
if ($Account) {
    $pw = 'Aa1!' + [guid]::NewGuid().ToString('N')
    [System.IO.File]::WriteAllText((Join-Path $env:RUNNER_TEMP 'pw.txt'), $pw)
    New-LocalUser -Name $Account -Password (ConvertTo-SecureString $pw -AsPlainText -Force) -PasswordNeverExpires -AccountNeverExpires | Out-Null
    Add-LocalGroupMember -Group 'Administrators' -Member $Account
    Write-Host "created account '$Account'"
}
exit 0
