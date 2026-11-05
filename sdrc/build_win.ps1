# build_win.ps1 -Label L -Sha S -Patch 0|1 -Out DIR
# Windows Vulkan build like unsloth-sd-prebuilt.yml's win-vulkan-x64 leg (MSVC + Ninja, static ggml),
# optionally with igpu_fit.patch on top (the head arm). Run after msvc-dev-cmd + Vulkan SDK.
param([string]$Label, [string]$Sha, [string]$Patch, [string]$Out)
$ErrorActionPreference = 'Continue'
$H = $PSScriptRoot
git config --global core.autocrlf false
$src = Join-Path $env:RUNNER_TEMP "src_$Label"
if (Test-Path $src) { Remove-Item -Recurse -Force $src }
git -c core.autocrlf=false clone -q https://github.com/unslothai/stable-diffusion.cpp $src
if ($LASTEXITCODE -ne 0) { throw 'clone failed' }
Set-Location $src
git checkout -q $Sha
if ($LASTEXITCODE -ne 0) { throw 'checkout failed' }
git -c core.autocrlf=false submodule update --init --depth 1 ggml
foreach ($p in (Get-ChildItem scripts/unsloth/ggml-patches -Filter *.patch | Sort-Object Name)) {
  git -C ggml apply "../scripts/unsloth/ggml-patches/$($p.Name)"
  if ($LASTEXITCODE -ne 0) { throw "patch $($p.Name) failed" }
}
if ($Patch -eq '1') {
  # actions/checkout under core.autocrlf=true hands the patch over as CRLF; the source tree is LF
  $lf = Join-Path $env:RUNNER_TEMP 'igpu_fit_lf.patch'
  [IO.File]::WriteAllText($lf, ([IO.File]::ReadAllText("$H/igpu_fit.patch") -replace "`r`n", "`n"))
  git apply --verbose $lf
  if ($LASTEXITCODE -ne 0) { throw 'igpu_fit.patch failed' }
}
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='/bigobj' -DSD_BUILD_EXAMPLES=ON `
  -DSD_SERVER_BUILD_FRONTEND=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_VULKAN=ON -DGGML_NATIVE=OFF
if ($LASTEXITCODE -ne 0) { throw 'cmake failed' }
$t0 = Get-Date
cmake --build build --config Release -j 4 --target sd-cli
if ($LASTEXITCODE -ne 0) { throw 'build failed' }
Write-Host "build $Label took $([int]((Get-Date) - $t0).TotalSeconds) s"
New-Item -ItemType Directory -Force $Out | Out-Null
Copy-Item build/bin/* $Out -Recurse
"label=$Label sha=$(git rev-parse HEAD) patched=$Patch ggml=$(git -C ggml rev-parse HEAD) kind=vk-win" | Set-Content (Join-Path $Out srcinfo.txt)
git diff --stat | Add-Content (Join-Path $Out srcinfo.txt)
Get-ChildItem $Out
exit 0
