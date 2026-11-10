# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Run only on a disposable Windows runner: the driver creates a standard user and
# temporarily installs a narrow AppLocker rule. It restores the original policy.
param(
    [ValidateSet('Driver', 'Control', 'Recover')][string]$Mode = 'Driver',
    [ValidateSet('x86_64', 'arm64')][string]$Arch = 'x86_64',
    [string]$Root,
    [string]$Source,
    [string]$ShellTag,
    [string]$BlockedPython,
    [switch]$ExpectBlocked
)
$ErrorActionPreference = 'Stop'
$global:LASTEXITCODE = 0

if (-not $Source) { $Source = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'install.ps1' }
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Installer has parse errors' }
function Get-InstallerFunction([string]$Name) {
    $matches = @($ast.FindAll({ param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $Name
    }, $true))
    if ($matches.Count -ne 1) { throw "Expected one installer function: $Name" }
    return $matches[0].Extent.Text
}
foreach ($name in @('Test-ApplicationControlBlock', 'Invoke-PythonSelectionProbe')) {
    . ([scriptblock]::Create((Get-InstallerFunction $name)))
}

if ($Mode -eq 'Control') {
    $denied = $false
    try {
        & $BlockedPython -I -c "print('native interpreter control')"
        if ($LASTEXITCODE -ne 0) { throw "Interpreter exited $LASTEXITCODE" }
    } catch {
        $exception = $_.Exception
        while ($exception) {
            if (($exception -is [System.ComponentModel.Win32Exception] -and $exception.NativeErrorCode -eq 1260) -or
                $exception.HResult -eq -2147023636) { $denied = $true }
            $exception = $exception.InnerException
        }
        if (-not $denied) { throw }
    }
    $probe = Invoke-PythonSelectionProbe -Executable $BlockedPython -Arguments @('-I', '-c', 'print(1)')
    @{ Shell=$PSVersionTable.PSVersion.ToString(); Denied=$denied; ProbeBlocked=$probe.Blocked; Diagnostic=$probe.Diagnostic } |
        ConvertTo-Json -Compress | Write-Host
    if ($ExpectBlocked) {
        if (-not $denied -or -not $probe.Blocked -or $probe.Success) { exit 90 }
    } elseif ($denied -or -not $probe.Success) { exit 91 }
    exit 0
}

if (-not $Root) {
    $Root = Join-Path $env:RUNNER_TEMP ('signed-python-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $Root | Out-Null
}
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root 'uv-python'
$env:UV_CACHE_DIR = Join-Path $Root 'uv-cache'
$env:UV_NO_CONFIG = '1'
$env:UV_CONFIG_FILE = $null
$env:UV_OFFLINE = $null
$env:UV_PYTHON_DOWNLOADS = $null
$env:UV_WORKING_DIR = $null
$env:UV_WORKING_DIRECTORY = $null
$env:UV_PROJECT = $null
$env:RUST_LOG = $null
$script:UvExe = Join-Path $Root 'tools\uv.exe'

if ($Mode -eq 'Recover') {
    foreach ($name in @('Get-SignedPythonPackage', 'Test-SignedPythonRuntime', 'Test-SignedPythonDownloadAllowed',
                       'Resolve-SignedManagedPython', 'Resolve-UvManagedPython', 'Test-UvEnvFlag', 'Test-Arm64PythonOptOut')) {
        . ([scriptblock]::Create((Get-InstallerFunction $name)))
    }
    function Write-StudioLine { param($Message, $ForegroundColor) Write-Host $Message }
    function substep { param($Message, $Color) Write-Host $Message }
    function step { param($Name, $Message) Write-Host "$Name`: $Message" }
    function Redact-InstallOutput { param($Text) return $Text }
    function Exit-InstallFailure { param($Message) throw $Message }
    function Invoke-InstallCommand {
        param([scriptblock]$Command, [string]$Label)
        $ErrorActionPreference = 'Continue'
        $global:LASTEXITCODE = 0
        & $Command 2>&1 | ForEach-Object { Write-Host "$_" }
        return [int]$global:LASTEXITCODE
    }
    $script:Downloads = 0
    $script:ForbidDownload = $false
    function Invoke-WebRequest {
        param([switch]$UseBasicParsing, $Uri, $OutFile, $TimeoutSec, $ErrorAction)
        $script:Downloads++
        if ($script:ForbidDownload) { throw 'Unexpected signed-runtime download' }
        Microsoft.PowerShell.Utility\Invoke-WebRequest @PSBoundParameters | Out-Null
    }
    $sourceText = [IO.File]::ReadAllText($Source)
    function Get-InstallerSection([string]$Start, [string]$End) {
        $a = $sourceText.IndexOf($Start, [StringComparison]::Ordinal)
        $b = $sourceText.IndexOf($End, $a, [StringComparison]::Ordinal)
        if ($a -lt 0 -or $b -le $a) { throw "Missing installer section: $Start" }
        return [scriptblock]::Create($sourceText.Substring($a, $b - $a))
    }
    # Both executable sections come from the PR rather than a second implementation.
    $selection = Get-InstallerSection '    $DetectedPython = Resolve-UvManagedPython -Minor $PythonVersion' '    # '
    # The first following comment is the venv boundary; require the full selection block.
    if ($selection.ToString() -notmatch 'using signed managed Python') { throw 'Incomplete runtime selection section' }
    $venvCreation = Get-InstallerSection '    if (-not (Test-Path -LiteralPath $VenvPython)) {' '    # Mark the managed venv before probing'
    $PythonSkip = @('3.13.8')
    $StudioHome = Join-Path $Root ("Studio home $ShellTag")
    function Select-SignedRuntime {
        $PythonVersion='3.13'; $PythonPatchRequest=''; $PythonRequestLabel='3.13'; $PythonArch=$Arch
        . $selection
        if ($DetectedPython.Source -ne 'signed-managed') { throw 'Signed managed runtime was not selected' }
        return $DetectedPython
    }
    function Assert-Venv([string]$Python, [string]$Base) {
        $signature = Get-AuthenticodeSignature -LiteralPath $Python
        if ($signature.Status -ne 'Valid') { throw "Venv launcher signature: $($signature.Status)" }
        $result = Invoke-PythonSelectionProbe -Executable $Python -Arguments @('-I', '-c',
            'import json,sys,sysconfig,ssl,sqlite3,ctypes; print(json.dumps(dict(version=sys.version.split()[0], tag=sysconfig.get_platform(), base=sys.base_prefix, venv=sys.prefix!=sys.base_prefix)))')
        if (-not $result.Success) { throw $result.Diagnostic }
        $facts = ($result.Output -join "`n") | ConvertFrom-Json
        $tag = if ($Arch -eq 'arm64') { 'win-arm64' } else { 'win-amd64' }
        if ($facts.version -ne '3.13.17' -or $facts.tag -ne $tag -or -not $facts.venv -or
            [IO.Path]::GetFullPath($facts.base) -ne [IO.Path]::GetFullPath($Base)) { throw 'Venv facts do not match the selected signed base' }
    }
    function New-TestVenv($Runtime, [string]$Directory) {
        $DetectedPython=$Runtime; $VenvDir=$Directory; $VenvPython=Join-Path $Directory 'Scripts\python.exe'
        . $venvCreation
        Assert-Venv $VenvPython (Split-Path $Runtime.Path -Parent)
    }
    $configDir = Join-Path $Root ("policy-$ShellTag")
    New-Item -ItemType Directory -Path $configDir | Out-Null
    $env:UV_CONFIG_FILE = Join-Path $configDir 'uv.toml'
    $env:UV_NO_CONFIG = $null
    $script:ForbidDownload = $true
    foreach ($config in @('offline = true', 'python-downloads = "never"')) {
        [IO.File]::WriteAllText($env:UV_CONFIG_FILE, $config)
        if (Test-SignedPythonDownloadAllowed) { throw 'File-based download policy was ignored' }
        $accepted = $false
        try { $null = Select-SignedRuntime; $accepted = $true } catch { Write-Host $_.Exception.Message }
        if ($accepted -or $script:Downloads -ne 0) { throw 'A cold signed runtime bypassed file-based download policy' }
    }
    $env:UV_CONFIG_FILE = $null; $env:UV_NO_CONFIG = '1'
    $script:ForbidDownload = $false
    $first = Select-SignedRuntime
    if ($script:Downloads -ne 1) { throw 'Cold recovery did not download exactly one signed package' }
    $firstRoot = Split-Path $first.Path -Parent
    $firstVenv = Join-Path $StudioHome 'first environment'
    New-TestVenv $first $firstVenv
    $cfg = Join-Path $firstVenv 'pyvenv.cfg'
    $cfgHash = (Get-FileHash -LiteralPath $cfg -Algorithm SHA256).Hash
    $exeHash = (Get-FileHash -LiteralPath $first.Path -Algorithm SHA256).Hash
    # Losing pythonw.exe invalidates the base without preventing the old console venv from running.
    $missing = Join-Path $firstRoot 'pythonw.exe'
    Get-Item -LiteralPath $missing | Format-List FullName, Length | Out-Host
    Move-Item -LiteralPath $missing -Destination ($missing + '.saved')
    Assert-Venv (Join-Path $firstVenv 'Scripts\python.exe') $firstRoot
    $repaired = Select-SignedRuntime
    if ($script:Downloads -ne 2 -or $repaired.Path -eq $first.Path) { throw 'Repair did not publish a separate signed base' }
    if ((Test-Path -LiteralPath $missing) -or -not (Test-Path -LiteralPath ($missing + '.saved')) -or
        (Get-FileHash -LiteralPath $cfg -Algorithm SHA256).Hash -ne $cfgHash -or
        (Get-FileHash -LiteralPath $first.Path -Algorithm SHA256).Hash -ne $exeHash) { throw 'Repair changed the existing environment or base' }
    Assert-Venv (Join-Path $firstVenv 'Scripts\python.exe') $firstRoot
    New-TestVenv $repaired (Join-Path $StudioHome 'repaired environment')
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root ("empty-uv-$ShellTag")
    $env:UV_OFFLINE = '1'
    $script:ForbidDownload = $true
    $cached = Select-SignedRuntime
    if ($script:PythonPolicyBlocked -or $cached.Path -ne $repaired.Path -or $script:Downloads -ne 2) { throw 'Offline cache reuse required a fresh policy block or download' }
    if (@(Get-ChildItem -LiteralPath (Join-Path $StudioHome 'python-signed') -Filter '.stage-*' -Force).Count) { throw 'Staging directories were left behind' }
    $report = @{ Shell=$PSVersionTable.PSVersion.ToString(); Arch=$Arch; Downloads=$script:Downloads; Initial=$first.Path; Repaired=$repaired.Path; Offline=$cached.Path; ExistingVenvPreserved=$true }
    $report | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root "logs\recovery-$ShellTag.json") -Encoding UTF8
    $report | ConvertTo-Json | Write-Host
    exit 0
}

$logs = Join-Path $Root 'logs'
New-Item -ItemType Directory -Path $logs, (Join-Path $Root 'tools') | Out-Null
Copy-Item -LiteralPath (Get-Command uv -ErrorAction Stop).Source -Destination $script:UvExe
$uvArch = if ($Arch -eq 'arm64') { 'aarch64' } else { 'x86_64' }
$key = "cpython-3.13-windows-$uvArch-none"
& $script:UvExe python install --no-bin --no-registry $key
if ($LASTEXITCODE -ne 0) { throw 'Could not provision the uv-managed control runtime' }
$BlockedPython = (& $script:UvExe python find --system --managed-python $key).Trim()
if (-not (Test-Path -LiteralPath $BlockedPython -PathType Leaf)) { throw 'Control runtime was not found' }
$user = 'SignedPythonProbe'
$password = 'Pr17330!' + [guid]::NewGuid().ToString('N').Substring(0, 6)
$policyBackup = Join-Path $logs 'original-applocker.xml'
Get-AppLockerPolicy -Local -Xml | Set-Content -LiteralPath $policyBackup -Encoding UTF8
$createdUser = $false
try {
    & net.exe user $user $password /add /passwordchg:no
    if ($LASTEXITCODE -ne 0) { throw 'Could not create standard probe user' }
    $createdUser = $true
    $account = [System.Security.Principal.NTAccount]::new($env:COMPUTERNAME, $user)
    $sid = $account.Translate([System.Security.Principal.SecurityIdentifier]).Value
    $credential = [pscredential]::new("$env:COMPUTERNAME\$user", (ConvertTo-SecureString $password -AsPlainText -Force))
    & icacls.exe $Root /grant "${user}:(OI)(CI)M" /T /C /Q | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not grant access to probe files' }
    & icacls.exe (Split-Path $Source -Parent) /grant "${user}:(OI)(CI)RX" /T /C /Q | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not grant read access to the checkout' }
    $shells = @(
        @{ Tag='ps51'; Path=(Get-Command powershell.exe -ErrorAction Stop).Source },
        @{ Tag='ps7'; Path=(Get-Command pwsh.exe -ErrorAction Stop).Source }
    )
    function Invoke-ProbeUser($Shell, [string]$WorkerMode, [string]$Label, [switch]$Denied) {
        $arguments = @('-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File',
            ('"' + $PSCommandPath + '"'), '-Mode', $WorkerMode, '-Arch', $Arch, '-Root', ('"' + $Root + '"'),
            '-Source', ('"' + $Source + '"'), '-ShellTag', $Shell.Tag, '-BlockedPython', ('"' + $BlockedPython + '"'))
        if ($Denied) { $arguments += '-ExpectBlocked' }
        $stdout = Join-Path $logs "$Label.stdout.log"
        $stderr = Join-Path $logs "$Label.stderr.log"
        $process = Start-Process -FilePath $Shell.Path -ArgumentList $arguments -Credential $credential -LoadUserProfile `
            -WorkingDirectory $Root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru
        Get-Content -LiteralPath $stdout, $stderr | Write-Host
        if ($null -eq $process.ExitCode) { throw "No exit status for $Label" }
        return [int]$process.ExitCode
    }
    foreach ($shell in $shells) {
        if ((Invoke-ProbeUser $shell 'Control' "baseline-$($shell.Tag)") -ne 0) { throw 'Standard user cannot run the baseline interpreter' }
    }
    $escaped = [System.Security.SecurityElement]::Escape($BlockedPython)
    $policy = @"
<AppLockerPolicy Version="1">
  <RuleCollection Type="Exe" EnforcementMode="Enabled">
    <FilePathRule Id="$([guid]::NewGuid())" Name="Allow other executables" Description="Keep the disposable runner usable" UserOrGroupSid="S-1-1-0" Action="Allow">
      <Conditions><FilePathCondition Path="*" /></Conditions>
    </FilePathRule>
    <FilePathRule Id="$([guid]::NewGuid())" Name="Deny uv Python for recovery test" Description="Exercise policy-refused managed Python" UserOrGroupSid="$sid" Action="Deny">
      <Conditions><FilePathCondition Path="$escaped" /></Conditions>
    </FilePathRule>
  </RuleCollection>
</AppLockerPolicy>
"@
    $policyFile = Join-Path $logs 'test-applocker.xml'
    $policy | Set-Content -LiteralPath $policyFile -Encoding UTF8
    & sc.exe config AppIDSvc start= auto
    if ($LASTEXITCODE -ne 0) { throw 'Could not configure AppIDSvc' }
    Start-Service AppIDSvc
    $started = Get-Date
    Set-AppLockerPolicy -XmlPolicy $policyFile
    & gpupdate.exe /target:computer /force
    if ($LASTEXITCODE -ne 0) { throw 'Could not apply AppLocker policy' }
    Restart-Service AppIDSvc -Force
    (Get-Service AppIDSvc).WaitForStatus('Running', [timespan]::FromSeconds(60))
    Get-AppLockerPolicy -Effective -Xml | Set-Content -LiteralPath (Join-Path $logs 'effective-applocker.xml') -Encoding UTF8
    foreach ($shell in $shells) {
        $exitCode = 90
        foreach ($attempt in 1..12) {
            $exitCode = Invoke-ProbeUser $shell 'Control' "denied-$($shell.Tag)-$attempt" -Denied
            if ($exitCode -eq 0) { break }
            if ($exitCode -ne 90) { throw "Unexpected control failure: $exitCode" }
            Start-Sleep -Seconds 5
        }
        if ($exitCode -ne 0) { throw 'The policy was not enforced; recovery results would be invalid' }
        if ((Invoke-ProbeUser $shell 'Recover' "recovery-$($shell.Tag)") -ne 0) { throw "Signed recovery failed in $($shell.Tag)" }
    }
    $events = @(Get-WinEvent -FilterHashtable @{ LogName='Microsoft-Windows-AppLocker/EXE and DLL'; Id=8004; StartTime=$started } |
        Where-Object { $_.Message -match 'python\.exe' })
    $events | Format-List TimeCreated, Id, Message | Out-File -LiteralPath (Join-Path $logs 'applocker-denials.txt') -Width 300 -Encoding utf8
    if ($events.Count -eq 0) { throw 'No AppLocker 8004 event corroborates the runtime denial' }
    Write-Host "PASS: $Arch signed recovery, immutable repair, and offline reuse in both PowerShell shells. Recorded $($events.Count) denial events."
} finally {
    Set-AppLockerPolicy -XmlPolicy $policyBackup
    if ($createdUser) { & net.exe user $user /delete | Out-Null }
    $global:LASTEXITCODE = 0
}
