# Runs install.ps1 in a child Windows PowerShell 5.1 and records what a reviewer needs.
# Used both by the scheduled task that runs as the "John Doe" account and directly by runneradmin.
param(
    [Parameter(Mandatory = $true)][string]$Installer,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string]$InstallArgs = ''
)
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
@(
    "whoami=$(whoami)",
    "USERPROFILE=$env:USERPROFILE",
    "TEMP=$env:TEMP",
    "TMP=$env:TMP",
    "GetTempPath=$([System.IO.Path]::GetTempPath())",
    "LOCALAPPDATA=$env:LOCALAPPDATA",
    "short(USERPROFILE)=$((New-Object -ComObject Scripting.FileSystemObject).GetFolder($env:USERPROFILE).ShortPath)"
) | Out-File -FilePath (Join-Path $OutDir 'env.txt') -Encoding utf8

$env:UNSLOTH_SKIP_AUTOSTART = '1'
$env:UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = '1'
$argv = @('-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $Installer)
if ($InstallArgs) { $argv += @($InstallArgs -split ' ') }
& powershell.exe @argv *>&1 | ForEach-Object { "$_" } | Out-File -FilePath (Join-Path $OutDir 'install.log') -Encoding utf8
$rc = $LASTEXITCODE

$py = Join-Path $env:USERPROFILE '.unsloth\studio\unsloth_studio\Scripts\python.exe'
if (Test-Path -LiteralPath $py) {
    & $py -c "import importlib.metadata as m`nfor p in ('torch','torchvision','torchaudio','unsloth','unsloth_zoo'):`n    try: print(p, m.version(p))`n    except Exception: print(p, 'MISSING')" *>&1 |
        ForEach-Object { "$_" } | Out-File -FilePath (Join-Path $OutDir 'pkgs.txt') -Encoding utf8
} else {
    "no venv at $py" | Out-File -FilePath (Join-Path $OutDir 'pkgs.txt') -Encoding utf8
}
$left = @(Get-ChildItem -LiteralPath ([System.IO.Path]::GetTempPath()) -File -ErrorAction SilentlyContinue | Where-Object {
    (Get-Content -LiteralPath $_.FullName -TotalCount 1 -ErrorAction SilentlyContinue) -match '^torch=='
})
"leftover=$($left.Count)" | Out-File -FilePath (Join-Path $OutDir 'leftover.txt') -Encoding ascii
"$rc" | Out-File -FilePath (Join-Path $OutDir 'done.txt') -Encoding ascii
