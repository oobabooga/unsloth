# Boots the installed Studio as the current user and checks health, Desktop-style login, an
# authenticated API call and the served web page. Records the package set and the desktop secret.
param(
    [Parameter(Mandatory = $true)][string]$OutFile,
    [int]$Port = 8899,
    [switch]$Provision
)
$ErrorActionPreference = 'Continue'
$r = [ordered]@{}
$root = Join-Path $env:USERPROFILE '.unsloth\studio'
$py = Join-Path $root 'unsloth_studio\Scripts\python.exe'
$cli = Join-Path $root 'unsloth_studio\Scripts\unsloth.exe'
$secretPath = Join-Path $root 'auth\.desktop_secret'
$r.user = "$(whoami)"
$r.root = $root
$r.venv = [bool](Test-Path -LiteralPath $py)

function Sha($s) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    return ([BitConverter]::ToString($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes([string]$s)))).Replace('-', '').Substring(0, 16)
}

$freeze = @(& $py -c "import importlib.metadata as m`nfor x in sorted({(d.metadata['Name'] or '').lower().replace('_','-') + '==' + d.version for d in m.distributions()}): print(x)" 2>&1 | ForEach-Object { "$_" })
[System.IO.File]::WriteAllLines("$OutFile.freeze.txt", [string[]]$freeze)
$r.packages = $freeze.Count
$r.freeze_sha = Sha ($freeze -join "`n")
$r.torch = (@($freeze | Where-Object { $_ -match '^torch==' }) -join ' ')
$r.unsloth = (@($freeze | Where-Object { $_ -match '^unsloth==' }) -join ' ')

if ($Provision) {
    $out = & $cli studio provision-desktop-auth 2>&1 | ForEach-Object { "$_" }
    $r.provision_rc = $LASTEXITCODE
    $r.provision_out = ($out -join ' ').Trim()
}
$r.secret_present = [bool](Test-Path -LiteralPath $secretPath)
$secret = $null
if ($r.secret_present) { $secret = ([System.IO.File]::ReadAllText($secretPath)).Trim(); $r.secret_sha = Sha $secret }

$env:UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = '1'
# A scheduled task starts in C:\Windows\system32, which Unsloth refuses to run from by hand.
Set-Location -LiteralPath $env:USERPROFILE
[Environment]::CurrentDirectory = $env:USERPROFILE
$p = Start-Process -FilePath $cli -ArgumentList @('studio', '-p', "$Port") -NoNewWindow -PassThru `
    -RedirectStandardOutput "$OutFile.studio.log" -RedirectStandardError "$OutFile.studio.err.log"
$null = $p.Handle
$base = "http://127.0.0.1:$Port"
$healthy = $false
$deadline = (Get-Date).AddSeconds(420)
while ((Get-Date) -lt $deadline) {
    try {
        $h = Invoke-RestMethod -Uri "$base/api/health" -TimeoutSec 5
        if ($h.status -eq 'healthy') { $healthy = $true; $r.health_service = $h.service; break }
    } catch {}
    if ($p.HasExited) { $r.studio_exited_early = $p.ExitCode; break }
    Start-Sleep -Seconds 3
}
$r.healthy = $healthy
if ($healthy -and $secret) {
    try {
        $tok = Invoke-RestMethod -Method Post -Uri "$base/api/auth/desktop-login" -ContentType 'application/json' -Body (@{ secret = $secret } | ConvertTo-Json) -TimeoutSec 30
        $r.login = [bool]$tok.access_token
        $st = Invoke-RestMethod -Uri "$base/api/inference/status" -Headers @{ Authorization = "Bearer $($tok.access_token)" } -TimeoutSec 30
        $r.authed_status_call = $true
    } catch { $r.login_error = $_.Exception.Message }
}
if ($healthy) {
    try {
        $page = Invoke-WebRequest -Uri "$base/" -UseBasicParsing -TimeoutSec 30
        $r.page_status = $page.StatusCode
        $r.page_has_root = [bool]($page.Content -match '<div id="root"')
    } catch { $r.page_error = $_.Exception.Message }
}
& taskkill.exe /PID $p.Id /T /F 2>&1 | Out-Null
Start-Sleep -Seconds 3
if (-not $healthy) {
    $tail = @(Get-Content -LiteralPath "$OutFile.studio.log", "$OutFile.studio.err.log" -ErrorAction SilentlyContinue | Select-Object -Last 15)
    $r.studio_log_tail = ($tail -join ' || ')
}
$lines = @($r.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" })
[System.IO.File]::WriteAllLines($OutFile, [string[]]$lines)
[System.IO.File]::WriteAllText("$OutFile.done", 'done')
