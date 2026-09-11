# Runs install.ps1 in a child Windows PowerShell 5.1 and records what a reviewer needs.
# Waits on the child process itself (not on its stdout pipe, which a lingering grandchild
# can hold open), and dumps the process table if the child does not exit.
param(
    [Parameter(Mandatory = $true)][string]$Installer,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string]$InstallArgs = '',
    [int]$TimeoutMinutes = 45
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
    "UserInteractive=$([Environment]::UserInteractive)",
    "short(USERPROFILE)=$((New-Object -ComObject Scripting.FileSystemObject).GetFolder($env:USERPROFILE).ShortPath)"
) | Out-File -FilePath (Join-Path $OutDir 'env.txt') -Encoding utf8

$env:UNSLOTH_SKIP_AUTOSTART = '1'
$env:UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = '1'
$argv = @('-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $Installer)
if ($InstallArgs) { $argv += @($InstallArgs -split ' ') }
$started = Get-Date
$p = Start-Process -FilePath 'powershell.exe' -ArgumentList $argv -NoNewWindow -PassThru `
    -RedirectStandardOutput (Join-Path $OutDir 'install.log') -RedirectStandardError (Join-Path $OutDir 'install.err.log')
$null = $p.Handle
$finished = $p.WaitForExit($TimeoutMinutes * 60 * 1000)
$elapsed = [int]((Get-Date) - $started).TotalSeconds
$procs = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name, CreationDate, CommandLine
$procs | Format-Table -AutoSize -Wrap | Out-String -Width 400 | Out-File -FilePath (Join-Path $OutDir 'processes.txt') -Encoding utf8
if ($finished) {
    $rc = $p.ExitCode
} else {
    $rc = 'TIMEOUT'
    # Which descendants of the installer are still alive, and what are they doing.
    $kids = New-Object System.Collections.ArrayList
    $frontier = @($p.Id)
    while ($frontier.Count) {
        $next = @()
        foreach ($id in $frontier) {
            foreach ($c in @($procs | Where-Object { $_.ParentProcessId -eq $id })) { [void]$kids.Add($c); $next += $c.ProcessId }
        }
        $frontier = $next
    }
    $kids | Format-List | Out-String -Width 400 | Out-File -FilePath (Join-Path $OutDir 'hung-descendants.txt') -Encoding utf8
    try {
        (Get-Process -Id $p.Id).Threads | Select-Object Id, ThreadState, WaitReason | Format-Table | Out-String |
            Out-File -FilePath (Join-Path $OutDir 'hung-threads.txt') -Encoding utf8
    } catch {}
    & taskkill.exe /PID $p.Id /T /F | Out-Null
}

$py = Join-Path $env:USERPROFILE '.unsloth\studio\unsloth_studio\Scripts\python.exe'
if (Test-Path -LiteralPath $py) {
    & $py -c "import importlib.metadata as m`nfor p in ('torch','torchvision','torchaudio','unsloth','unsloth_zoo'):`n    try: print(p, m.version(p))`n    except Exception: print(p, 'MISSING')" *>&1 |
        ForEach-Object { "$_" } | Out-File -FilePath (Join-Path $OutDir 'pkgs.txt') -Encoding utf8
    $cli = Join-Path $env:USERPROFILE '.unsloth\studio\unsloth_studio\Scripts\unsloth.exe'
    & $cli --help *>&1 | Select-Object -First 5 | ForEach-Object { "$_" } | Out-File -FilePath (Join-Path $OutDir 'cli.txt') -Encoding utf8
    "cli exit=$LASTEXITCODE" | Out-File -Append -FilePath (Join-Path $OutDir 'cli.txt') -Encoding utf8
} else {
    "no venv at $py" | Out-File -FilePath (Join-Path $OutDir 'pkgs.txt') -Encoding utf8
}
$left = @(Get-ChildItem -LiteralPath ([System.IO.Path]::GetTempPath()) -File -ErrorAction SilentlyContinue | Where-Object {
    (Get-Content -LiteralPath $_.FullName -TotalCount 1 -ErrorAction SilentlyContinue) -match '^torch=='
})
"leftover=$($left.Count)" | Out-File -FilePath (Join-Path $OutDir 'leftover.txt') -Encoding ascii
"elapsed=${elapsed}s" | Out-File -FilePath (Join-Path $OutDir 'elapsed.txt') -Encoding ascii
"$rc" | Out-File -FilePath (Join-Path $OutDir 'done.txt') -Encoding ascii
