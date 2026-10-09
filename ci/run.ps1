# Windows Vulkan A/B. Env: BASE HEAD TBO_OPS H3 H3_HEAD_ENVS ZIMG HEAD_CMAKE
$ErrorActionPreference = 'Continue'
$W = Join-Path $env:RUNNER_TEMP ("sdrev-" + $env:GITHUB_RUN_ID + "-" + $PID)
New-Item -ItemType Directory -Force -Path "$W\tmp","$W\hf","$W\models","$W\runs","$W\art" | Out-Null
Add-Content $env:GITHUB_ENV "ART=$W\art"
$S = "$W\art\summary.md"
$CI = Split-Path -Parent $MyInvocation.MyCommand.Path
function Log($m) { Write-Host $m; Add-Content -Path $S -Value $m }
$env:TMPDIR = "$W\tmp"; $env:TEMP = "$W\tmp"; $env:HF_HOME = "$W\hf"; $env:HF_HUB_ENABLE_HF_TRANSFER = "1"
Log "## windows-vulkan base=$env:BASE head=$env:HEAD"
Log ("host " + $env:COMPUTERNAME + " " + (Get-CimInstance Win32_OperatingSystem).Caption + " RAM " + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB) + " GB")

python -m venv "$W\venv"
$py = "$W\venv\Scripts\python.exe"
& $py -m pip -q install ninja numpy pillow huggingface_hub hf_transfer 2>&1 | Out-Null
$env:PATH = "$W\venv\Scripts;" + $env:PATH

# models (background)
$dl = @()
if ($env:H3 -eq "1") { $dl += "unsloth/MiniMax-H3-GGUF:minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf","unsloth/MiniMax-H3-GGUF:qwen3vl_32b_minimax_h3-Q2_K_M.gguf","unsloth/MiniMax-H3-GGUF:vae/minimax_h3_video_vae_fp16.safetensors","unsloth/MiniMax-H3-GGUF:vae/minimax_h3_audio_vae_fp32.safetensors" }
if ($env:ZIMG) { $dl += "unsloth/Z-Image-Turbo-GGUF:z-image-turbo-Q4_K_M.gguf","unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf","Comfy-Org/z_image_turbo:split_files/vae/ae.safetensors" }
$dlpy = @"
import sys, os
from huggingface_hub import hf_hub_download as d
for a in sys.argv[1:]:
    r, f = a.split(':', 1)
    for t in range(3):
        try: print(d(r, f, local_dir=os.environ['MODELS_DIR']), flush=True); break
        except Exception as e: print('retry', f, e, flush=True)
print('DONE', flush=True)
"@
Set-Content -Path "$W\dl.py" -Value $dlpy
$env:MODELS_DIR = "$W\models"
$dlp = Start-Process -FilePath $py -ArgumentList (@("$W\dl.py") + $dl) -RedirectStandardOutput "$W\dl.log" -RedirectStandardError "$W\dl.err" -PassThru -NoNewWindow

# Vulkan SDK (same pin as the prebuilt workflow)
$installer = "$W\VulkanSDK-Installer.exe"; $sdk = "$W\VulkanSDK"
curl.exe --fail --location --retry 3 --silent --output $installer "https://sdk.lunarg.com/sdk/download/1.4.328.1/windows/vulkansdk-windows-X64-1.4.328.1.exe"
if ((Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash -ne "a8675df6d538079c2a719a9373994948091db785b48f142e024254e76348d16c") { Log "Vulkan SDK sha mismatch"; exit 1 }
$p = Start-Process -FilePath $installer -ArgumentList @('--root', "`"$sdk`"", '--accept-licenses', '--default-answer', '--confirm-command', 'install') -Wait -PassThru
if ($p.ExitCode -ne 0) { Log "Vulkan SDK install failed"; exit 1 }
$env:VULKAN_SDK = $sdk; $env:PATH = "$sdk\Bin;" + $env:PATH

# MSVC env
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
Log ("msvc " + (cl 2>&1 | Select-Object -First 1))

# sources
git clone -q --filter=blob:none https://github.com/unslothai/stable-diffusion.cpp "$W\src"
git -C "$W\src" fetch -q origin $env:BASE $env:HEAD 2>$null
git clone -q https://github.com/leejet/ggml "$W\ggml-clone"
foreach ($n in "base","head") {
  $sha = if ($n -eq "base") { $env:BASE } else { $env:HEAD }
  git -C "$W\src" worktree add -q --detach "$W\$n" $sha
  if ($LASTEXITCODE -ne 0) { Log "checkout $n failed"; exit 1 }
  $g = ((git -C "$W\$n" ls-tree HEAD ggml) -split '\s+')[2]
  Remove-Item -Recurse -Force "$W\$n\ggml" -ErrorAction SilentlyContinue
  git clone -q --shared "$W\ggml-clone" "$W\$n\ggml"
  git -C "$W\$n\ggml" fetch -q https://github.com/leejet/ggml $g; git -C "$W\$n\ggml" checkout -q $g
  if ((git -C "$W\$n\ggml" rev-parse HEAD) -ne $g) { Log "ggml at wrong commit"; exit 1 }
  $np = 0
  foreach ($pf in (Get-ChildItem "$W\$n\scripts\unsloth\ggml-patches\*.patch" -ErrorAction SilentlyContinue | Sort-Object Name)) {
    git -C "$W\$n\ggml" apply $pf.FullName; if ($LASTEXITCODE -ne 0) { Log "PATCH FAIL $n $($pf.Name)"; exit 1 }; $np++ }
  & $py "$CI\harness.py" "$W\$n\examples\cli\main.cpp" | Out-Null
  Log ("$n " + (git -C "$W\$n" log --oneline -1) + " ggml=$($g.Substring(0,8)) patches=$np")
}
foreach ($n in "base","head") {
  $extra = @(); if ($n -eq "head" -and $env:HEAD_CMAKE) { $extra = $env:HEAD_CMAKE -split ' ' }
  $t0 = Get-Date
  cmake -S "$W\$n" -B "$W\$n\build" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS=/bigobj" -DSD_BUILD_EXAMPLES=ON -DSD_SERVER_BUILD_FRONTEND=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_BUILD_TESTS=ON @extra *> "$W\art\cmake-$n.log"
  if ($LASTEXITCODE -ne 0) { Get-Content "$W\art\cmake-$n.log" -Tail 40; Log "CMAKE FAIL $n"; exit 1 }
  cmake --build "$W\$n\build" -j 16 --target sd-cli sd-server test-backend-ops *> "$W\art\build-$n.log"
  if ($LASTEXITCODE -ne 0) { Select-String -Path "$W\art\build-$n.log" -Pattern "error" | Select-Object -First 40 | ForEach-Object { Write-Host $_.Line }; Log "BUILD FAIL $n"; exit 1 }
  Log ("built $n in " + [int]((Get-Date) - $t0).TotalSeconds + "s; warnings: " + (Select-String -Path "$W\art\build-$n.log" -Pattern "warning" -SimpleMatch).Count)
}
$bin = @{ base = "$W\base\build\bin\sd-cli.exe"; head = "$W\head\build\bin\sd-cli.exe" }
$tbo = @{ base = "$W\base\build\bin\test-backend-ops.exe"; head = "$W\head\build\bin\test-backend-ops.exe" }
& $tbo.head -o ADD *> "$W\art\devcheck.log"
if (-not (Select-String -Path "$W\art\devcheck.log" -Pattern "Vulkan0" -Quiet)) { Get-Content "$W\art\devcheck.log"; Log "DEVICE Vulkan0 NOT FOUND"; exit 1 }
Log ("device: " + ((Select-String -Path "$W\art\devcheck.log" -Pattern "Vulkan0" | Select-Object -First 1).Line))

if ($env:TBO_OPS) { foreach ($op in ($env:TBO_OPS -split ',')) { foreach ($n in "base","head") {
  & $tbo[$n] test -b Vulkan0 -o $op *> "$W\art\tbo-$op-$n.log"
  $r = (Select-String -Path "$W\art\tbo-$op-$n.log" -Pattern "tests passed" | Select-Object -Last 1).Line
  $f = (Select-String -Path "$W\art\tbo-$op-$n.log" -Pattern "[FAIL]" -SimpleMatch).Count
  Log "tbo $op ${n}: $r fail=$f" } } }

function WaitModels { for ($i = 0; $i -lt 360; $i++) { if ((Test-Path "$W\dl.log") -and (Select-String -Path "$W\dl.log" -Pattern "DONE" -Quiet)) { return }; Start-Sleep 15 }; Log "MODEL DOWNLOAD TIMEOUT"; exit 1 }
function Times($f) { ((Select-String -Path $f -Pattern "sampling completed|decode_first_stage completed|decoding audio latent completed|generate_image completed") | ForEach-Object { $_.Line -replace '.*\] ','' }) -join ' ' }
function RunSd($name, $exe, $envs, $argv) {
  New-Item -ItemType Directory -Force -Path "$W\runs\$name" | Out-Null
  $saved = @{}; foreach ($e in $envs) { if ($e) { $k,$v = $e -split '=',2; $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $v) } }
  $sw = [Diagnostics.Stopwatch]::StartNew()
  & $exe @argv *> "$W\runs\$name.log"; $rc = $LASTEXITCODE
  $sw.Stop(); foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
  Log ("run $name rc=$rc WALL " + [math]::Round($sw.Elapsed.TotalSeconds,1) + " s | " + (Times "$W\runs\$name.log"))
  $fa = (Select-String -Path "$W\runs\$name.log" -Pattern "flash attention" | ForEach-Object { $_.Line -replace '.*\] ','' }) -join ' / '
  Log ("   attention: $fa")
  Copy-Item "$W\runs\$name.log" -Destination "$W\art\$name.log" -ErrorAction SilentlyContinue
}
$M = "$W\models"
function H3Args($name) { @("-M","vid_gen","--diffusion-model","$M\minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf","--vae","$M\vae\minimax_h3_video_vae_fp16.safetensors","--audio-vae","$M\vae\minimax_h3_audio_vae_fp32.safetensors","--llm","$M\qwen3vl_32b_minimax_h3-Q2_K_M.gguf","-p",$(if ($env:H3_PROMPT) { $env:H3_PROMPT } else { "A red fox trots through fresh snow in a pine forest at sunrise, breath steaming, soft crunching footsteps and distant birdsong." }),"--cfg-scale","1.0","-W",$(if ($env:H3_W) { $env:H3_W } else { "640" }),"-H",$(if ($env:H3_H) { $env:H3_H } else { "384" }),"--video-frames",$(if ($env:H3_F) { $env:H3_F } else { "56" }),"--steps",$(if ($env:H3_STEPS) { $env:H3_STEPS } else { "4" }),"--seed","42","--rng","cpu","--fps","24","--diffusion-fa","--offload-to-cpu","--max-vram","-1","-v","-o","$W\runs\$name\f_%03d.png") }
if ($env:H3 -eq "1") {
  WaitModels
  if ($env:H3_WARMUP -eq "1") { RunSd "warm" $bin.base @() (H3Args "warm") }
  $reps = if ($env:H3_AB_REPS) { [int]$env:H3_AB_REPS } else { 2 }
  for ($r = 1; $r -le $reps; $r++) { RunSd "base$r" $bin.base @() (H3Args "base$r"); RunSd "head$r" $bin.head @() (H3Args "head$r") }
  if ($env:H3_REFCPU -eq "1") {
    RunSd "ref_base" $bin.base @() ((H3Args "ref_base") + @("--backend","cpu")); RunSd "ref_head" $bin.head @() ((H3Args "ref_head") + @("--backend","cpu"))
    Log ("REF cpu base vs cpu head: " + (& $py "$CI\compare.py" "$W\runs\ref_base" "$W\runs\ref_head"))
    Log ("REF gpu base vs cpu base: " + (& $py "$CI\compare.py" "$W\runs\ref_base" "$W\runs\base1"))
    Log ("REF gpu head vs cpu base: " + (& $py "$CI\compare.py" "$W\runs\ref_base" "$W\runs\head1"))
    Log ("REF gpu head vs cpu head: " + (& $py "$CI\compare.py" "$W\runs\ref_head" "$W\runs\head1"))
  }
  Log ("cmp base1 base2: " + (& $py "$CI\compare.py" "$W\runs\base1" "$W\runs\base2"))
  Log ("cmp head1 head2: " + (& $py "$CI\compare.py" "$W\runs\head1" "$W\runs\head2"))
  Log ("cmp base1 head1: " + (& $py "$CI\compare.py" "$W\runs\base1" "$W\runs\head1"))
  if ($env:H3_HEAD_ENVS) { foreach ($arm in ($env:H3_HEAD_ENVS -split ';')) { if (-not $arm) { continue }
    $nm, $ev = $arm -split ':',2; RunSd "head_$nm" $bin.head ($ev -split ' ') (H3Args "head_$nm")
    Log ("cmp base1 head_$nm [$ev]: " + (& $py "$CI\compare.py" "$W\runs\base1" "$W\runs\head_$nm")) } }
}
if ($env:ZIMG) {
  WaitModels
  foreach ($spec in ($env:ZIMG -split ' ')) {
    $wh, $fl = $spec -split ':',2; $w, $h = $wh -split 'x'; $flags = @(); if ($fl) { $flags = $fl -split ',' }
    $k = "z_${wh}_" + (($fl -replace '[^a-z0-9]','')); 
    foreach ($n in "base","head") {
      $argv = @("--diffusion-model","$M\z-image-turbo-Q4_K_M.gguf","--llm","$M\Qwen3-4B-Q4_K_M.gguf","--vae","$M\split_files\vae\ae.safetensors","-p","A lighthouse on a rocky coast at dusk, waves crashing, warm light in the windows, detailed photograph","--cfg-scale","1.0","--steps","8","-W",$w,"-H",$h,"--seed","7","--diffusion-fa") + $flags + @("-v","-o","$W\runs\${k}_$n\out.png")
      RunSd "${k}_$n" $bin[$n] @() $argv }
    Log ("cmp ${k}: " + (& $py "$CI\compare.py" "$W\runs\${k}_base" "$W\runs\${k}_head"))
  }
}
Get-ChildItem "$W\runs" -Filter *.log | Copy-Item -Destination "$W\art"
Get-ChildItem "$W\runs" -Directory | ForEach-Object { $d = New-Item -ItemType Directory -Force -Path "$W\art\$($_.Name)"; Get-ChildItem $_.FullName -Filter *.png | Sort-Object Name | Where-Object { $i = [int]($_.BaseName -replace '\D',''); $_.Name -eq 'out.png' -or $i -eq 0 -or $i % 20 -eq 0 } | Copy-Item -Destination $d; Get-ChildItem $_.FullName -Filter *.wav | Copy-Item -Destination $d }
Get-Content $S
