# win.ps1: gfx1151 Windows 11. Trees as linux.sh (i3 = public CI copy of k19-int3 1f09fc4a8fb, d6 = + d6.patch = 11-06 dc86a8f271b,
# rp = + rp.patch = k19-gdnreplay trial merge 2b1738df6c3), tree hashes checked. i3 / d6: the release's Windows ROCm flags (shared libs,
# GGML_BACKEND_DL, CPU_ALL_VARIANTS, RPC, OPENMP off); the model-based tests are not registered with shared libs on Windows, so rp (and
# i3s = i3 again) are also built with static libs, which registers them (test-llama-archs / -context / replay tests). Then run.py.
# Writes only under $env:AMD_CI_WORK.
$ErrorActionPreference = "Continue"
$W = $env:AMD_CI_WORK; $OUT = "$W\out"; $J = "$env:GITHUB_WORKSPACE\k19p"
New-Item -ItemType Directory -Force -Path $OUT, "$OUT\build" | Out-Null
function Log($s) { Write-Host $s; Add-Content -Encoding utf8 -Path "$OUT\diag.txt" -Value $s }
Write-Host "RUNNER_OS='$env:RUNNER_OS' RUNNER_NAME='$env:RUNNER_NAME'"
$PY = $null
foreach ($c in @("C:\Program Files\Python312\python.exe", "C:\Program Files\Python311\python.exe")) { if (Test-Path -LiteralPath $c) { $PY = $c; break } }
if ($null -eq $PY) { Write-Host "FATAL: no system Python"; exit 1 }
Get-CimInstance Win32_ComputerSystem | ForEach-Object { Log "RAM $($_.TotalPhysicalMemory) procs $($_.NumberOfLogicalProcessors)" }
Get-CimInstance Win32_Processor | ForEach-Object { Log "CPU $($_.Name)" }
Get-CimInstance Win32_VideoController | ForEach-Object { Log "GPU $($_.Name) driver $($_.DriverVersion)" }

$ROCM_VER = if ($env:ROCM_VER) { $env:ROCM_VER } else { "10.2.0a20261017" }
$t0 = Get-Date
& curl.exe -fL --retry 5 -s -S -o "$W\rocm.tar.gz" "https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-$ROCM_VER.tar.gz"
Log "TheRock tarball rc=$LASTEXITCODE"
$ROCM = "$W\rocm"
New-Item -ItemType Directory -Force -Path $ROCM | Out-Null
tar.exe -xzf "$W\rocm.tar.gz" -C $ROCM --strip-components=1
Remove-Item -Force "$W\rocm.tar.gz"
Log "TheRock $ROCM_VER extracted in $([int]((Get-Date) - $t0).TotalSeconds) s"
$CLANG = "$ROCM\lib\llvm\bin\clang.exe"; $CLANGXX = "$ROCM\lib\llvm\bin\clang++.exe"
if (-not (Test-Path $CLANG)) { Write-Host "FATAL: no clang at $CLANG"; exit 1 }
$vsp = (& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
$vc = if ($vsp) { "$vsp\VC\Auxiliary\Build\vcvars64.bat" } else { "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" }
cmd /c "`"$vc`" >nul 2>&1 && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
$env:HIP_PATH = $ROCM; $env:ROCM_PATH = $ROCM; $env:HIP_PLATFORM = "amd"
$env:PATH = "$ROCM\bin;$ROCM\lib\llvm\bin;$env:PATH"
Log ((& $CLANG --version) -join " | ")
$ocml = Get-ChildItem -Path $ROCM -Recurse -Filter "ocml.bc" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $ocml) { Write-Host "FATAL: no ocml.bc"; exit 1 }
$ROCMF = $ROCM.Replace('\', '/'); $BCF = $ocml.DirectoryName.Replace('\', '/')
$env:HIP_DEVICE_LIB_PATH = $ocml.DirectoryName
$HFLAGS = "--rocm-path=$ROCMF --rocm-device-lib-path=$BCF -Wno-unused-command-line-argument"
& curl.exe -fsSL -o "$W\relocate_kpack.py" "https://raw.githubusercontent.com/unslothai/llama.cpp/f95b52162ed4b916a58d4f16b72662483cc55075/scripts/unsloth/relocate_kpack.py"

function Build-Tree($src, $bdir, $name, $static) {
  $t0 = Get-Date
  $libflags = if ($static) { @("-DBUILD_SHARED_LIBS=OFF", "-DGGML_BACKEND_DL=OFF", "-DGGML_CPU_ALL_VARIANTS=OFF", "-DGGML_NATIVE=ON") } else { @("-DBUILD_SHARED_LIBS=ON", "-DGGML_BACKEND_DL=ON", "-DGGML_CPU_ALL_VARIANTS=ON", "-DGGML_NATIVE=OFF") }
  cmake -S $src -B $bdir -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_C_COMPILER=$CLANG" "-DCMAKE_CXX_COMPILER=$CLANGXX" "-DCMAKE_HIP_COMPILER=$CLANGXX" `
    "-DCMAKE_PREFIX_PATH=$ROCM" "-DROCM_DIR=$ROCM" "-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types" `
    "-DCMAKE_CXX_FLAGS=-I$ROCMF/include $HFLAGS" "-DCMAKE_HIP_FLAGS=$HFLAGS" `
    -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF `
    -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_RPC=ON -DGGML_HIP_ROCWMMA_FATTN=OFF `
    -DGGML_STATIC=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF @libflags *> "$OUT\build\cfg_$name.log"
  if ($LASTEXITCODE -ne 0) { Get-Content "$OUT\build\cfg_$name.log" -Tail 25; Log "$name configure FAILED"; return $false }
  cmake --build $bdir -j 32 -- -k 0 *> "$OUT\build\build_$name.log"
  $brc = $LASTEXITCODE
  Log "build $name rc=$brc $([int]((Get-Date) - $t0).TotalSeconds) s warnings $((Select-String -Path "$OUT\build\build_$name.log" -Pattern ' warning:').Count)"
  if ($brc -ne 0) {
    Select-String -Path "$OUT\build\build_$name.log" -Pattern " error:|FAILED:|undefined symbol|unresolved external" | Select-Object -First 40 | ForEach-Object { Log "  $name $($_.Line)" }
    return $false
  }
  $d = "$bdir\bin"
  Copy-Item "$ROCM\bin\*.dll" "$d\" -Force
  foreach ($lib in "rocblas", "hipblaslt") {
    $ld = Get-ChildItem -Path $ROCM -Recurse -Directory -Filter "library" -ErrorAction SilentlyContinue | Where-Object { $_.Parent.Name -eq $lib } | Select-Object -First 1
    if ($ld) { New-Item -ItemType Directory -Force -Path "$d\$lib" | Out-Null; Copy-Item -Path $ld.FullName -Destination "$d\$lib\library" -Recurse -Force }
  }
  & $PY "$W\relocate_kpack.py" --bundle $d --kpack-src "$ROCM\.kpack" --archs gfx1151 *>> "$OUT\build\kpack_$name.log"
  Log "$name kpack rc=$LASTEXITCODE"
  $c = (& "$d\llama-server.exe" --list-devices 2>&1 | Select-String "ROCm0").Count
  Log "devices $name : ROCm0 x$c"
  return $true
}

$TREES = Get-Content -Encoding utf8 "$J\TREES"
$I3C = (($TREES | Where-Object { $_.StartsWith("i3 ") }) -split " ")[1]
$TARGS = @()
foreach ($spec in @("i3:i3:0", "d6:d6:0", "rp:rp:1", "i3s:i3:1")) {
  $n, $src, $st = $spec -split ":"
  $S = "$W\s_$src"
  if (-not (Test-Path "$S\.git")) {
    New-Item -ItemType Directory -Force -Path $S | Out-Null
    git -C $S init -q
    git -C $S config core.autocrlf false
    git -C $S remote add origin https://github.com/danielhanchen/llama.cpp-staging-ci
    git -C $S fetch -q --depth 1 origin $I3C
    git -C $S checkout -q --detach $I3C
    if ($src -ne "i3") {
      & $PY -c "import gzip,shutil,sys; shutil.copyfileobj(gzip.open(sys.argv[1]), open(sys.argv[2],'wb'))" "$J\$src.patch.gz" "$W\$src.patch"
      git -C $S apply --index --whitespace=nowarn "$W\$src.patch"
      if ($LASTEXITCODE -ne 0) { Log "FATAL: $src patch does not apply"; continue }
    }
    $exp = (($TREES | Where-Object { $_.StartsWith("$src ") }) -split " ")[2]
    $tree = (git -C $S write-tree).Trim()
    Log "tree $src $tree expect $exp"
    if ($tree -ne $exp) { Log "FATAL: the applied tree $src is not the expected one"; Remove-Item -Recurse -Force "$S\.git"; continue }
  }
  $bd = "$S\build_$n"
  if (Build-Tree $S $bd $n ($st -eq "1")) { $TARGS += @("--tree", "$n=$bd|$bd\bin") }
}
& $PY -c "import gzip,shutil,sys; shutil.copyfileobj(gzip.open(sys.argv[1]), open(sys.argv[2],'wb'))" "$W\s_i3\ci\tiny-qwen3moe-f16.gguf.gz" "$W\tiny-qwen3moe-f16.gguf"
$env:PATH = "$ROCM\bin;$env:PATH"
& $PY -u "$J\run.py" --out $OUT @TARGS --ctest i3,d6,rp,i3s --ctest-replay i3s --tbo i3,d6 --replay i3s,rp --smoke i3,d6 `
  --models "$W\s_i3\build_i3s\tests\test-models" --tinymoe "$W\tiny-qwen3moe-f16.gguf" *> "$OUT\run.txt"
Get-Content "$OUT\run.txt" -Tail 60
if (-not (Test-Path "$OUT\rows.jsonl")) { exit 1 }
