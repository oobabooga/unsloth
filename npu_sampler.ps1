param([string]$Out, [string]$StopFile)
# Samples the Windows counters Task Manager reads for the NPU (GPU Engine instances of the NPU adapter)
# plus any NPU-specific counter sets, once a second, until $StopFile exists.
$ErrorActionPreference = 'Continue'
$paths = @('\GPU Engine(*)\Utilization Percentage')
try {
    $sets = Get-Counter -ListSet '*NPU*' -ErrorAction SilentlyContinue
    foreach ($s in $sets) {
        foreach ($p in $s.Paths) { if ($p -match 'Utilization|Running Time|Busy') { $paths += $p } }
    }
} catch {}
$paths += '\Process(flm*)\% Processor Time'
$paths += '\Process(lemond*)\% Processor Time'
$paths += '\Process(python*)\% Processor Time'
$paths += '\Processor(_Total)\% Processor Time'
"ts_ms,path,value" | Out-File -Encoding ascii $Out
("# paths: " + ($paths -join ' | ')) | Out-File -Append -Encoding ascii $Out
while (-not (Test-Path $StopFile)) {
    try {
        $s = Get-Counter -Counter $paths -SampleInterval 1 -MaxSamples 1 -ErrorAction SilentlyContinue
        $ts = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $lines = @("$ts,tick,0")
        foreach ($c in $s.CounterSamples) {
            if ($c.CookedValue -ne 0 -or $c.Path -notmatch 'gpu engine') {
                $lines += "$ts,""$($c.Path)"",$($c.CookedValue)"
            }
        }
        $lines | Out-File -Append -Encoding ascii $Out
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
