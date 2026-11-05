# build_win.ps1 -Label L -Pr N -Sha S -Out DIR
# Builds unslothai/stable-diffusion.cpp at Sha exactly like unsloth-sd-prebuilt.yml's win-vulkan-x64 leg
# (MSVC + Ninja, -DSD_VULKAN=ON, static ggml), plus ggml's test-backend-ops. Run after msvc-dev-cmd + Vulkan SDK.
param([string]$Label, [string]$Pr, [string]$Sha, [string]$Out)
$ErrorActionPreference = 'Continue'
$H = $PSScriptRoot
# LF everywhere: the hosted image sets core.autocrlf=true, which turned some patch / target pairs CRLF vs LF.
git config --global core.autocrlf false
$src = Join-Path $env:RUNNER_TEMP "src_$Label"
if (Test-Path $src) { Remove-Item -Recurse -Force $src }
git -c core.autocrlf=false clone -q https://github.com/unslothai/stable-diffusion.cpp $src
if ($LASTEXITCODE -ne 0) { throw 'clone failed' }
Set-Location $src
git fetch -q origin $Sha
if ($LASTEXITCODE -ne 0) { git fetch -q origin "refs/pull/$Pr/head" }
git checkout -q $Sha
if ($LASTEXITCODE -ne 0) { throw 'checkout failed' }
git -c core.autocrlf=false submodule update --init --depth 1 ggml
foreach ($p in (Get-ChildItem scripts/unsloth/ggml-patches -Filter *.patch | Sort-Object Name)) {
  git -C ggml apply --verbose "../scripts/unsloth/ggml-patches/$($p.Name)"
  if ($LASTEXITCODE -ne 0) { throw "patch $($p.Name) failed" }
}
python "$H/tbo_perf_cases.py" ggml/tests/test-backend-ops.cpp
if ($LASTEXITCODE -ne 0) { throw 'tbo cases failed' }
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='/bigobj' -DSD_BUILD_EXAMPLES=ON `
  -DSD_SERVER_BUILD_FRONTEND=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_BUILD_TESTS=ON
if ($LASTEXITCODE -ne 0) { throw 'cmake failed' }
$t0 = Get-Date
cmake --build build --config Release -j 4 --target sd-cli sd-server test-backend-ops
if ($LASTEXITCODE -ne 0) { throw 'build failed' }
Write-Host "build $Label took $([int]((Get-Date) - $t0).TotalSeconds) s"
New-Item -ItemType Directory -Force $Out | Out-Null
Copy-Item build/bin/* $Out -Recurse
Get-ChildItem build -Recurse -Filter test-backend-ops.exe | Select-Object -First 1 | Copy-Item -Destination $Out
"label=$Label pr=$Pr sha=$(git rev-parse HEAD) ggml=$(git -C ggml rev-parse HEAD) kind=vk-win" | Set-Content (Join-Path $Out srcinfo.txt)
Get-ChildItem $Out
exit 0
