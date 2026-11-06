# PR 16857 cells on gfx1151 Windows 11: published unslothai/llama.cpp bundles (Vulkan = what Studio picks on
# gfx1151 when an AMD Vulkan ICD is present, plus the ROCm gfx1151 bundle), no rebuild.
$ErrorActionPreference = "Continue"
$W = $env:AMD_CI_WORK; $OUT = "$W\out"; $R = "$W\p16857"
New-Item -ItemType Directory -Force -Path $OUT, "$R\bins\rocm", "$R\bins\vulkan", "$R\models" | Out-Null
$TAG = if ($env:TAG) { $env:TAG } else { "b12128-mix-2f23ac9" }
Write-Host "runner $env:RUNNER_NAME"
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | Out-String | Write-Host
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM | Out-String | Write-Host
$M = "$R\models\gemma-4-E2B-it-BF16.gguf"
& curl.exe -fL --retry 8 --retry-delay 10 -s -S -o $M "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/0314792d7f1f7e229411f620751375812bb9faf2/gemma-4-E2B-it-BF16.gguf"
Write-Host "model rc=$LASTEXITCODE size $((Get-Item $M).Length)"
foreach ($b in @("rocm-gfx1151", "vulkan")) {
  $d = if ($b -eq "vulkan") { "vulkan" } else { "rocm" }
  $z = "$R\$d.zip"
  & curl.exe -fL --retry 5 -s -S -o $z "https://github.com/unslothai/llama.cpp/releases/download/$TAG/app-$TAG-windows-x64-$b.zip"
  Write-Host "bundle $b rc=$LASTEXITCODE"
  Expand-Archive -Force -Path $z -DestinationPath "$R\bins\$d"
}
if ((Get-Item $M).Length -ne 9311305568) { Write-Host "model size mismatch"; exit 1 }
$fs = [System.IO.File]::OpenRead($M); $buf = New-Object byte[] (64MB)
while ($fs.Read($buf, 0, $buf.Length) -gt 0) { }; $fs.Close()
& $env:AMD_CI_PY -u "$env:GITHUB_WORKSPACE\amd_ci\pr16857\cells.py" --bins "$R\bins" --backends "vulkan,rocm" `
  --model $M --out $OUT --fit-target 16384 2>&1 | Tee-Object -FilePath "$OUT\cells_console.log"
if (-not (Test-Path "$OUT\cells.jsonl")) { exit 1 }
