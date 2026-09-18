# Prints the launch contract of both Studio shortcuts, one field per line, so two runs can be diffed.
$sh = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
foreach ($l in @((Join-Path $desktop "Unsloth Studio.lnk"), (Join-Path $startMenu "Unsloth Studio.lnk"))) {
    if (-not (Test-Path -LiteralPath $l)) { Write-Output "LNK-MISSING $l"; continue }
    $s = $sh.CreateShortcut($l)
    foreach ($f in "TargetPath", "Arguments", "WorkingDirectory", "IconLocation", "WindowStyle", "Description") {
        Write-Output ("LNK {0} {1}={2}" -f (Split-Path -Leaf (Split-Path -Parent $l)), $f, $s.$f)
    }
}
