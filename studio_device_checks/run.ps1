# Studio device checks for Windows (x64 and Arm). Uses the Studio that is already installed; installs nothing.
#   powershell -ExecutionPolicy Bypass -File run.ps1 -Mode quick
#   powershell -ExecutionPolicy Bypass -File run.ps1 -Mode full [-Yes] [-StudioHome D:\path\to\studio]
# Works on Windows PowerShell 5.1 and PowerShell 7. Driver options (--prs, --work-dir, ...) pass through.
# No param() block on purpose: any [Parameter()] attribute turns on PowerShell's common parameters, and then a
# driver option like --out is read as an ambiguous -OutVariable / -OutBuffer. $args is parsed by hand instead.
$Mode = "quick"; $StudioHome = ""; $Yes = $false; $DryRun = $false; $Rest = @()
for ($i = 0; $i -lt $args.Count; $i++) {
    $a = [string]$args[$i]
    switch -Regex ($a) {
        '^-Mode$' { $i++; $Mode = [string]$args[$i]; continue }
        '^-StudioHome$' { $i++; $StudioHome = [string]$args[$i]; continue }
        '^-Yes$' { $Yes = $true; continue }
        '^-DryRun$' { $DryRun = $true; continue }
        default { $Rest += $a }
    }
}
if ($Mode -notin @("quick", "full")) { Write-Host "-Mode must be quick or full (got '$Mode')"; exit 2 }
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$driver = Join-Path $here "devchecks.py"

# Same order as install.ps1: UNSLOTH_STUDIO_HOME > STUDIO_HOME > UNSLOTH_HOME\studio > %USERPROFILE%\.unsloth\studio
$homes = @()
if ($StudioHome) { $homes += $StudioHome }
else {
    foreach ($v in @($env:UNSLOTH_STUDIO_HOME, $env:STUDIO_HOME)) {
        if (-not [string]::IsNullOrWhiteSpace($v)) { $homes += $v.Trim() }
    }
    if (-not [string]::IsNullOrWhiteSpace($env:UNSLOTH_HOME)) { $homes += [System.IO.Path]::Combine($env:UNSLOTH_HOME.Trim(), "studio") }
    if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) { $homes += [System.IO.Path]::Combine($env:USERPROFILE, ".unsloth", "studio") }
}
$py = $null
foreach ($h in $homes) {
    foreach ($venv in @("unsloth_studio", ".venv")) {
        $cand = [System.IO.Path]::Combine($h, $venv, "Scripts", "python.exe")
        if (Test-Path -LiteralPath $cand -PathType Leaf) { $py = $cand; break }
    }
    if ($py) { break }
}
$driverArgs = @("--mode", $Mode)
if ($StudioHome) { $driverArgs += @("--studio-home", $StudioHome) }
if ($Yes) { $driverArgs += "--yes" }
if ($DryRun) { $driverArgs += "--dry-run" }
if ($Rest) { $driverArgs += $Rest }
if ($py) {
    Write-Host "Studio Python: $py"
    $driverArgs += @("--studio-python", $py)
} else {
    # No Studio venv found: let any Python run the driver so it can report what it searched.
    foreach ($name in @("py", "python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { $py = $cmd.Source; break }
    }
    if (-not $py) {
        Write-Host "No Unsloth Studio install found (searched: $($homes -join ', ')) and no Python on PATH."
        Write-Host "Pass -StudioHome <folder that holds unsloth_studio>."
        exit 2
    }
    Write-Host "No Studio venv found under: $($homes -join ', '). The driver will report what it searched."
}
& $py -u $driver @driverArgs
exit $LASTEXITCODE
