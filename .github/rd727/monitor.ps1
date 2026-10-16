param([string]$Flag, [string]$Log)
# Polls every second. Raises $Flag while any process command line names flash_attn, and logs
# the installer processes, network receive total and Defender CPU time.
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$i = 0
$lastFa = $false
while ($true) {
    $procs = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='uv.exe' OR Name='pip.exe' OR Name='powershell.exe' OR Name='unsloth.exe'" -ErrorAction SilentlyContinue)
    $fa = @($procs | Where-Object { $_.CommandLine -match 'flash_attn|flash-attn' })
    if ($fa.Count -gt 0) {
        if (-not (Test-Path $Flag)) { New-Item -ItemType File -Path $Flag -Force | Out-Null }
    } elseif (Test-Path $Flag) {
        Remove-Item $Flag -Force -ErrorAction SilentlyContinue
    }
    $faNow = $fa.Count -gt 0
    if (($faNow -and ($i % 5 -eq 0)) -or ($faNow -ne $lastFa) -or ($i % 15 -eq 0)) {
        $rx = (Get-NetAdapterStatistics -ErrorAction SilentlyContinue | Measure-Object -Property ReceivedBytes -Sum).Sum
        $mp = (Get-Process MsMpEng -ErrorAction SilentlyContinue | Select-Object -First 1).CPU
        $lines = @("[{0,7:N1}s] rx={1:N0} MB msmpeng_cpu={2:N1}s fa_procs={3}" -f $sw.Elapsed.TotalSeconds, ($rx / 1MB), $mp, $fa.Count)
        foreach ($p in $procs) {
            if ($p.Name -eq 'powershell.exe' -and $p.CommandLine -notmatch 'setup\.ps1') { continue }
            $cmd = ($p.CommandLine -replace '\s+', ' ')
            if ($cmd.Length -gt 260) { $cmd = $cmd.Substring(0, 260) + '...' }
            $lines += ("    pid={0} {1} cpu={2:N1}s {3}" -f $p.ProcessId, $p.Name, ((Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue).CPU), $cmd)
        }
        Add-Content -Path $Log -Value $lines -Encoding utf8
    }
    $lastFa = $faNow
    $i++
    Start-Sleep -Seconds 1
}
