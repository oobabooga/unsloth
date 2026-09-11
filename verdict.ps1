# Reads probe files (key=value) and asserts the job's expectations. Exit code = failures.
param([Parameter(Mandatory = $true)][string]$Spec)   # e.g. "probe1,probe2|same=freeze_sha,secret_sha|ok=healthy,login,authed_status_call,page_has_root"
$bad = 0
$parts = $Spec -split '\|'
$names = $parts[0] -split ','
$probes = @{}
foreach ($n in $names) {
    $f = "C:\Users\Public\e2e\$n.txt"
    if (-not (Test-Path -LiteralPath $f)) { $f = Join-Path $env:RUNNER_TEMP "$n.txt" }
    $kv = @{}
    if (Test-Path -LiteralPath $f) { foreach ($l in [System.IO.File]::ReadAllLines($f)) { $i = $l.IndexOf('='); if ($i -gt 0) { $kv[$l.Substring(0, $i)] = $l.Substring($i + 1) } } }
    $probes[$n] = $kv
    Write-Host "--- $n ---"; $kv.GetEnumerator() | Sort-Object Key | ForEach-Object { Write-Host "  $($_.Key)=$($_.Value)" }
}
foreach ($p in $parts[1..($parts.Count - 1)]) {
    $kind, $keys = $p -split '=', 2
    foreach ($k in ($keys -split ',')) {
        if ($kind -eq 'ok') {
            foreach ($n in $names) { if ($probes[$n][$k] -ne 'True') { Write-Host "::error::$n.$k = '$($probes[$n][$k])'"; $bad++ } }
        } elseif ($kind -eq 'same') {
            $vals = @($names | ForEach-Object { $probes[$_][$k] } | Sort-Object -Unique)
            if ($vals.Count -ne 1 -or -not $vals[0]) { Write-Host "::error::$k differs across $($names -join ','): $($vals -join ' / ')"; $bad++ } else { Write-Host "same $k across $($names -join ','): $($vals[0])" }
        }
    }
}
Write-Host "verdict failures: $bad"
exit $bad
