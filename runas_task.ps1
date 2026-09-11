# Runs a script as another local account through a scheduled task (a real logon with its own
# profile) and waits for the file the script writes when it finishes.
param(
    [Parameter(Mandatory = $true)][string]$Account,
    [Parameter(Mandatory = $true)][string]$PasswordFile,
    [Parameter(Mandatory = $true)][string]$Script,
    [string]$ScriptArgs = '',
    [Parameter(Mandatory = $true)][string]$DoneFile,
    [int]$TimeoutMinutes = 60
)
$ErrorActionPreference = 'Stop'
$pw = ([System.IO.File]::ReadAllText($PasswordFile)).Trim()
$name = 'unsloth-e2e-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$arg = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Script`" $ScriptArgs"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName $name -Action $action -Settings $settings -User $Account -Password $pw -RunLevel Highest | Out-Null
Start-ScheduledTask -TaskName $name
$ErrorActionPreference = 'Continue'
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$ok = $true
while (-not (Test-Path -LiteralPath $DoneFile)) {
    if ((Get-Date) -gt $deadline) { Write-Host "::error::timed out waiting for $DoneFile"; $ok = $false; break }
    Start-Sleep -Seconds 20
    $st = (Get-ScheduledTask -TaskName $name).State
    if ("$st" -eq 'Ready' -and -not (Test-Path -LiteralPath $DoneFile)) {
        Start-Sleep -Seconds 15
        if (-not (Test-Path -LiteralPath $DoneFile)) {
            Write-Host "::error::task ended without $DoneFile (result $((Get-ScheduledTaskInfo -TaskName $name).LastTaskResult))"
            $ok = $false; break
        }
    }
}
Unregister-ScheduledTask -TaskName $name -Confirm:$false
Write-Host "[$Account] $Script $ScriptArgs -> $(if ($ok) { 'finished' } else { 'NOT finished' })"
if (-not $ok) { exit 1 }
exit 0
