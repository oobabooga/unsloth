<#
Isolation preamble for every AMD CI job on the WINDOWS half of the pool. Run
this first, from a step whose only job is setup, under `shell: powershell`.

The counterpart of lib/preamble.sh, and deliberately a separate file rather than
a branch inside it: the Windows boxes have no bash at all, so a shared script
could not be read, let alone run.

What it encodes, each learned the same way the bash rules were:

  $env:RUNNER_TEMP   the only tree the runner reclaims between jobs, and the
                     ONLY correct spelling here. A bash-style $RUNNER_TEMP
                     inside a PowerShell block expands to "" and the step then
                     writes to the filesystem root and still exits 0, which is
                     lint error E104.
  ErrorActionPreference
                     GitHub runs the step with 'stop', so the first error record
                     aborts it. In a job whose subject is "does this command
                     fail", that throws the measurement away, exactly as
                     `bash -e` does on the Linux half.
  no BOM             PowerShell 5.1's Out-File and Add-Content -Encoding utf8
                     both write a byte order mark, and GITHUB_ENV parses the
                     first line as `<BOM>NAME=value`, so the variable silently
                     never arrives. That is lint warning W104. Every append here
                     goes through UTF8Encoding($false).
  no elevation       Nothing in these jobs needs Administrator. There is no sudo
                     to shim, so instead the elevation state is RECORDED, which
                     makes "this ran unprivileged" a checkable property of the
                     artifact rather than a claim in a comment.

Usage from a workflow step:

    - name: Preamble
      shell: powershell
      run: |
        .\amd_ci\lib\preamble.ps1 -Name "pr1234-windows"

It sets AMD_CI_WORK for this step and appends AMD_CI_WORK / TMP / TEMP / PATH to
$env:GITHUB_ENV for the ones after it.

WINDOWS ONLY. It refuses elsewhere rather than half-working, for the same reason
preamble.sh refuses under MSYS: a preamble that quietly puts the work root
somewhere the job cannot see is the silent failure this toolkit exists to stop.
#>

[CmdletBinding()]
param(
    [string] $Name = "amd_ci"
)

$ErrorActionPreference = "Continue"

# ---------------------------------------------------------------- refuse early
# $IsWindows does not exist in PowerShell 5.1, which is the only edition these
# boxes have, so it is read defensively rather than trusted.
$onWindows = $true
if (Test-Path variable:global:IsWindows) { $onWindows = $IsWindows }
elseif ($env:OS -ne "Windows_NT") { $onWindows = $false }
if (-not $onWindows) {
    Write-Error ("FATAL: preamble.ps1 ran somewhere that is not Windows. The Linux AMD " +
                 "runners take lib/preamble.sh under a bash shell. Refusing rather than " +
                 "writing the work root somewhere the job cannot see.")
    exit 1
}

# ------------------------------------------------------------------- work root
if (-not $env:RUNNER_TEMP) {
    Write-Warning ("RUNNER_TEMP is unset; falling back to the user TEMP, which is NOT " +
                   "reclaimed between jobs on these persistent runners.")
}
$root = $env:RUNNER_TEMP
if (-not $root) { $root = $env:TEMP }
if (-not $root) { $root = "C:\Windows\Temp" }

$work = Join-Path $root $Name
foreach ($sub in @("bin", "out", "src", "tmp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $work $sub) | Out-Null
}

$env:AMD_CI_WORK = $work
$env:TMP = Join-Path $work "tmp"
$env:TEMP = $env:TMP
$env:PATH = (Join-Path $work "bin") + ";" + $env:PATH

# ------------------------------------------------------- record the privileges
# There is no sudo here to shim, so the property is recorded instead of enforced.
$elevated = "unknown"
try {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    $elevated = [string] $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
} catch {
    $elevated = "unknown"
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    (Join-Path $work "out\privileges.txt"),
    ("elevated={0}`nuser={1}`nmachine={2}`n" -f $elevated, $env:USERNAME, $env:COMPUTERNAME),
    $utf8NoBom)

# --------------------------------------------------------- hand on to the step
# Never Out-File or Add-Content: PowerShell 5.1 gives both a BOM, and GITHUB_ENV
# then reads the first name with the mark glued to its front (W104).
if ($env:GITHUB_ENV) {
    $lines = @(
        "AMD_CI_WORK=$work",
        "TMP=$($env:TMP)",
        "TEMP=$($env:TEMP)",
        "PATH=$($env:PATH)"
    ) -join "`n"
    [System.IO.File]::AppendAllText($env:GITHUB_ENV, $lines + "`n", $utf8NoBom)
}

Write-Host "amd_ci work root: $work"
Write-Host ("elevated: {0}   machine: {1}   user: {2}" -f
            $elevated, $env:COMPUTERNAME, $env:USERNAME)

# The OS label is what selected this runner, so echo what it actually is. The
# machine-name prefix X2-LA01-* is shared with the Linux pool and identifies
# nothing, and RUNNER_OS has been observed blank here (W103), so both are printed
# rather than branched on.
Write-Host ("RUNNER_OS as seen here: '{0}'" -f $env:RUNNER_OS)
try {
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
    Write-Host ("{0} build {1}" -f $os.Caption, $os.BuildNumber)
} catch {
    Write-Host "could not read Win32_OperatingSystem"
}
try {
    $drive = (Get-Item $root).PSDrive
    Write-Host ("{0}: {1:N1} GB free of {2:N1} GB" -f $drive.Name,
                ($drive.Free / 1GB), (($drive.Free + $drive.Used) / 1GB))
} catch {
    Write-Host "could not read free space on $root"
}
