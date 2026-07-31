# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

$ErrorActionPreference = "Stop"

$feed = Join-Path $env:RUNNER_TEMP "windows-updater-feed"
$sourceExe = Join-Path $env:GITHUB_WORKSPACE "studio\src-tauri\target\debug\unsloth-studio.exe"
$installedExe = Join-Path $env:LOCALAPPDATA "Unsloth Studio (Desktop)\unsloth-studio.exe"
$tauriLog = Join-Path $env:USERPROFILE ".unsloth\studio\tauri.log"
$serverOut = Join-Path $env:RUNNER_TEMP "windows-updater-server.out.log"
$serverErr = Join-Path $env:RUNNER_TEMP "windows-updater-server.err.log"
$appOut = Join-Path $env:RUNNER_TEMP "windows-updater-app.out.log"
$appErr = Join-Path $env:RUNNER_TEMP "windows-updater-app.err.log"
$server = $null
$source = $null

function Show-Diagnostics {
  foreach ($path in @($tauriLog, $serverOut, $serverErr, $appOut, $appErr)) {
    Write-Host "===== $path ====="
    if (Test-Path $path) {
      Get-Content $path -Tail 300
    } else {
      Write-Host "missing"
    }
  }
  Write-Host "===== Local application directory ====="
  Get-ChildItem (Split-Path $installedExe) -Recurse -ErrorAction SilentlyContinue |
    Select-Object FullName, Length, LastWriteTime |
    Format-Table -AutoSize
  Write-Host "===== Unsloth Studio processes ====="
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "unsloth-studio.exe" } |
    Select-Object ProcessId, ExecutablePath, CommandLine |
    Format-List
}

try {
  if (-not (Test-Path $sourceExe)) {
    throw "Source desktop executable was not built: $sourceExe"
  }
  if (Test-Path $tauriLog) {
    Remove-Item $tauriLog -Force
  }

  $server = Start-Process python `
    -ArgumentList @("-m", "http.server", "8765", "--bind", "127.0.0.1", "--directory", $feed) `
    -RedirectStandardOutput $serverOut `
    -RedirectStandardError $serverErr `
    -PassThru `
    -WindowStyle Hidden

  $endpoint = "http://127.0.0.1:8765/latest.json"
  $serverDeadline = (Get-Date).AddSeconds(30)
  do {
    try {
      $response = Invoke-WebRequest $endpoint -UseBasicParsing
      if ($response.StatusCode -eq 200) { break }
    } catch {
      Start-Sleep -Milliseconds 250
    }
  } while ((Get-Date) -lt $serverDeadline)
  if (-not $response -or $response.StatusCode -ne 200) {
    throw "Local updater endpoint did not become ready"
  }

  Write-Host "Launching live Studio 9.9.1 from $sourceExe"
  $source = Start-Process $sourceExe `
    -RedirectStandardOutput $appOut `
    -RedirectStandardError $appErr `
    -PassThru

  $sourceDeadline = (Get-Date).AddMinutes(3)
  while (-not $source.HasExited -and (Get-Date) -lt $sourceDeadline) {
    Start-Sleep -Milliseconds 500
    $source.Refresh()
  }
  if (-not $source.HasExited) {
    throw "Studio 9.9.1 did not exit for the updater"
  }
  Write-Host "Studio 9.9.1 exited with code $($source.ExitCode)"

  $installDeadline = (Get-Date).AddMinutes(2)
  $installedVersion = $null
  do {
    if (Test-Path $installedExe) {
      $installedVersion = (Get-Item $installedExe).VersionInfo.ProductVersion
      if ($installedVersion -match '^9\.9\.2(?:\D|$)') { break }
    }
    Start-Sleep -Milliseconds 500
  } while ((Get-Date) -lt $installDeadline)
  if ($installedVersion -notmatch '^9\.9\.2(?:\D|$)') {
    throw "Expected installed Studio 9.9.2, found '$installedVersion' at $installedExe"
  }

  $launchDeadline = (Get-Date).AddSeconds(45)
  $relaunched = $null
  do {
    $relaunched = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
      Where-Object {
        $_.Name -eq "unsloth-studio.exe" -and
        $_.ExecutablePath -eq $installedExe
      } |
      Select-Object -First 1
    if ($relaunched) { break }
    Start-Sleep -Milliseconds 500
  } while ((Get-Date) -lt $launchDeadline)
  if (-not $relaunched) {
    throw "Updated Studio did not relaunch from $installedExe"
  }

  if (-not (Test-Path $tauriLog)) {
    throw "Studio did not create its Tauri log"
  }
  $logText = Get-Content $tauriLog -Raw
  if ($logText -notmatch "Windows job cleanup suspended for updater installer launch") {
    throw "The native updater handoff was not recorded in the Tauri log"
  }

  Write-Host "Live updater succeeded: Studio 9.9.1 downloaded, launched, and installed 9.9.2"
  Write-Host "Updated Studio relaunched with PID $($relaunched.ProcessId)"
  Write-Host "Installed executable: $installedExe"
} catch {
  Write-Host "Live updater test failed: $_"
  Show-Diagnostics
  throw
} finally {
  Get-Process -Name "unsloth-studio" -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue
  if ($server -and -not $server.HasExited) {
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
  }
}
