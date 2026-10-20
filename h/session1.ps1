# Run a command in the logged-on console session (session 1) through a scheduled task and
# stream its log until it finishes. Dot-source, then: Invoke-InSession1 -Name x -Command '...' -TimeoutMin 60
function Invoke-InSession1 {
  param([string]$Name, [string]$Command, [int]$TimeoutMin = 60, [hashtable]$Env = @{})
  $W = $env:H_W
  $log = Join-Path $W "$Name.log"
  $done = Join-Path $W "$Name.done"
  $bat = Join-Path $W "$Name.cmd"
  Remove-Item $log, $done -ErrorAction SilentlyContinue
  $lines = @('@echo off', "cd /d `"$W`"", 'set PYTHONUTF8=1', 'set PYTHONIOENCODING=utf-8')
  foreach ($k in $Env.Keys) { $lines += "set `"$k=$($Env[$k])`"" }
  $lines += "$Command > `"$log`" 2>&1"
  $lines += "echo %ERRORLEVEL% > `"$done`""
  Set-Content -LiteralPath $bat -Value $lines -Encoding ascii
  $task = "pr14245-$Name"
  $action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$bat`""
  $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Highest
  $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 6) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
  Register-ScheduledTask -TaskName $task -Action $action -Principal $principal -Settings $settings -Force | Out-Null
  Start-ScheduledTask -TaskName $task
  $deadline = (Get-Date).AddMinutes($TimeoutMin)
  $shown = 0
  while (-not (Test-Path $done)) {
    Start-Sleep -Seconds 15
    if (Test-Path $log) {
      $all = @(Get-Content -LiteralPath $log -ErrorAction SilentlyContinue)
      if ($all.Count -gt $shown) { $all[$shown..($all.Count - 1)] | ForEach-Object { Write-Host $_ }; $shown = $all.Count }
    }
    if ((Get-Date) -gt $deadline) {
      Write-Host "::warning::$Name timed out after $TimeoutMin min"
      Stop-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
      break
    }
  }
  if (Test-Path $log) {
    $all = @(Get-Content -LiteralPath $log -ErrorAction SilentlyContinue)
    if ($all.Count -gt $shown) { $all[$shown..($all.Count - 1)] | ForEach-Object { Write-Host $_ } }
  }
  Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue
  $rc = if (Test-Path $done) { (Get-Content $done | Select-Object -First 1).Trim() } else { 'timeout' }
  Write-Host "[$Name] exit: $rc"
  return $rc
}
