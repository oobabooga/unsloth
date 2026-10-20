# win_gpu.ps1 (k9 platforms): gfx1151 Windows 11. Build k8 / k9 (b11721 + k8 / k9 .patch.gz) with TheRock (pip, gfx1151 nightly index) and the release's
# Windows ROCm flags (unsloth-prebuilt-rocm.yml windows leg: shared libs, GGML_BACKEND_DL, CPU_ALL_VARIANTS, RPC, OPENMP off,
# ROCWMMA FA off), ship TheRock's DLLs + rocblas/hipblaslt libraries beside the exes (as the release does), then measure release
# b11734-mix-c208e5f windows-x64-rocm-gfx1151 (rel) vs k8 vs k9 (arms in k6p_plan.py) with k6p_bench.py. Rows -> out\rows_win.jsonl
$ErrorActionPreference = "Continue"
$W = $env:AMD_CI_WORK; $OUT = "$W\out"; $J = "$env:GITHUB_WORKSPACE\k6p"; $R = "$W\k6p"
New-Item -ItemType Directory -Force -Path $OUT, "$R\bins\rel", "$R\models" | Out-Null
function Log($s) { Write-Host $s; Add-Content -Encoding utf8 -Path "$OUT\win_diag.txt" -Value $s }
Write-Host "RUNNER_OS='$env:RUNNER_OS' RUNNER_NAME='$env:RUNNER_NAME'"
$PY = $null
foreach ($c in @("C:\Program Files\Python312\python.exe", "C:\Program Files\Python311\python.exe")) { if (Test-Path -LiteralPath $c) { $PY = $c; break } }
if ($null -eq $PY) { Write-Host "FATAL: no system Python"; exit 1 }
Log "python $PY"
Get-CimInstance Win32_ComputerSystem | ForEach-Object { Log "RAM $($_.TotalPhysicalMemory) procs $($_.NumberOfLogicalProcessors)" }
Get-CimInstance Win32_VideoController | ForEach-Object { Log "GPU $($_.Name) driver $($_.DriverVersion) AdapterRAM $($_.AdapterRAM)" }

$env:K6P_DIR = $R; $env:K6P_DEV = "ROCm0"; $env:ARMS = "rel:rel,k8:k8,k9:k9"; $env:PYTHONIOENCODING = "utf-8"
$dl = Start-Process -FilePath $PY -PassThru -NoNewWindow -RedirectStandardOutput "$OUT\dl.log" -RedirectStandardError "$OUT\dl.err" `
  -ArgumentList @("-u", "`"$J\k6p_bench.py`"", "dl", "--models", "q35,gptoss,q3m")

# release bundle
$TAG = "b11734-mix-c208e5f"
& curl.exe -fL --retry 5 -s -S -o "$R\rel.zip" "https://github.com/unslothai/llama.cpp/releases/download/$TAG/app-$TAG-windows-x64-rocm-gfx1151.zip"
Log "release zip rc=$LASTEXITCODE"
Expand-Archive -Force -Path "$R\rel.zip" -DestinationPath "$R\bins\rel"
Set-Content -Encoding utf8 -Path "$R\bins\rel\TAG" -Value "$TAG windows-x64-rocm-gfx1151"

# TheRock toolchain: the dist tarball the release's weekly pin resolves to for b11734 (built 2026-10-19 -> cutoff 2026-10-17)
$ROCM_VER = if ($env:ROCM_VER) { $env:ROCM_VER } else { "10.2.0a20261017" }
$t0 = Get-Date
& curl.exe -fL --retry 5 -s -S -o "$W\rocm.tar.gz" "https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-$ROCM_VER.tar.gz"
Log "TheRock tarball rc=$LASTEXITCODE size $((Get-Item "$W\rocm.tar.gz").Length)"
$ROCM = "$W\rocm"
New-Item -ItemType Directory -Force -Path $ROCM | Out-Null
tar.exe -xzf "$W\rocm.tar.gz" -C $ROCM --strip-components=1
Remove-Item -Force "$W\rocm.tar.gz"
Log "TheRock $ROCM_VER extracted in $([int]((Get-Date) - $t0).TotalSeconds) s to $ROCM"
& $PY -m venv "$W\venv"
$VP = "$W\venv\Scripts\python.exe"
$CLANG = "$ROCM\lib\llvm\bin\clang.exe"; $CLANGXX = "$ROCM\lib\llvm\bin\clang++.exe"
if (-not (Test-Path $CLANG)) { Write-Host "FATAL: no clang at $CLANG"; exit 1 }
$vsp = (& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
$vc = if ($vsp) { "$vsp\VC\Auxiliary\Build\vcvars64.bat" } else { "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" }
Log "vcvars $vc"
cmd /c "`"$vc`" >nul 2>&1 && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
$env:HIP_PATH = $ROCM; $env:ROCM_PATH = $ROCM; $env:HIP_PLATFORM = "amd"
$env:PATH = "$ROCM\bin;$ROCM\lib\llvm\bin;$env:PATH"
Log ((& $CLANG --version) -join " | ")

$BASE = "ff5a23376725e5a889fa0bfb987b7541dc304da0"

$ocml = Get-ChildItem -Path $ROCM -Recurse -Filter "ocml.bc" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $ocml) { Write-Host "FATAL: no ocml.bc"; exit 1 }
$ROCMF = $ROCM.Replace('\', '/'); $BCF = $ocml.DirectoryName.Replace('\', '/')
$env:HIP_DEVICE_LIB_PATH = $ocml.DirectoryName
$HFLAGS = "--rocm-path=$ROCMF --rocm-device-lib-path=$BCF -Wno-unused-command-line-argument"
# one release-flag build of a source tree into $R\bins\<name>, TheRock runtime + rocblas / hipblaslt kernels beside the exes
function Build-Tree($tree, $name, $tests) {
  $t0 = Get-Date
  cmake -S $tree -B "$tree\build" -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_C_COMPILER=$CLANG" "-DCMAKE_CXX_COMPILER=$CLANGXX" "-DCMAKE_HIP_COMPILER=$CLANGXX" `
    "-DCMAKE_PREFIX_PATH=$ROCM" "-DROCM_DIR=$ROCM" "-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types" `
    "-DCMAKE_CXX_FLAGS=-I$ROCMF/include $HFLAGS" "-DCMAKE_HIP_FLAGS=$HFLAGS" `
    -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF `
    -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_RPC=ON -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_NATIVE=OFF `
    -DGGML_STATIC=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF *> "$OUT\${name}_configure.log"
  if ($LASTEXITCODE -ne 0) { Get-Content "$OUT\${name}_configure.log" -Tail 25; Log "$name configure FAILED"; return $false }
  cmake --build "$tree\build" -j 32 -- -k 0 *> "$OUT\${name}_build.log"
  $brc = $LASTEXITCODE
  Log "$name build rc=$brc $([int]((Get-Date) - $t0).TotalSeconds) s"
  if ($brc -ne 0) { Get-Content "$OUT\${name}_build.log" | Select-String -Pattern "error:|FAILED" -Context 2, 6 | Select-Object -First 40; return $false }
  if ($tests) {
    cmake -S $tree -B "$tree\build" -DLLAMA_BUILD_TESTS=ON *>> "$OUT\${name}_configure.log"
    cmake --build "$tree\build" -j 32 -- -k 0 *> "$OUT\${name}_build_tests.log"
    Log "$name tests build rc=$LASTEXITCODE"
    $op = $env:PATH; $env:PATH = "$tree\build\bin;$ROCM\bin;$env:PATH"
    Push-Location "$tree\build"
    & ctest -L main --output-on-failure --timeout 900 *> "$OUT\${name}_ctest.log"
    Log "$name ctest rc=$LASTEXITCODE $((Get-Content "$OUT\${name}_ctest.log" | Select-String 'tests passed') -join ' ')"
    Get-Content "$OUT\${name}_ctest.log" | Select-String -Pattern "\*\*\*Failed|Not Run|Timeout|Exception" | Select-Object -First 20 | ForEach-Object { Log "  $_" }
    Pop-Location; $env:PATH = $op
  }
  $d = "$R\bins\$name"
  New-Item -ItemType Directory -Force -Path $d | Out-Null
  Copy-Item "$tree\build\bin\*" "$d\" -Recurse -Force
  Copy-Item "$ROCM\bin\*.dll" "$d\" -Force
  foreach ($lib in "rocblas", "hipblaslt") {
    $ld = Get-ChildItem -Path $ROCM -Recurse -Directory -Filter "library" -ErrorAction SilentlyContinue | Where-Object { $_.Parent.Name -eq $lib } | Select-Object -First 1
    if ($ld) { New-Item -ItemType Directory -Force -Path "$d\$lib" | Out-Null; Copy-Item -Path $ld.FullName -Destination "$d\$lib\library" -Recurse -Force }
  }
  return $true
}
$TRV = $ROCM_VER
# k8 / k9 = upstream b11721 + the k8 / k9 patch, one source tree each
foreach ($n in "k8", "k9") {
  $S = "$W\src_$n"
  New-Item -ItemType Directory -Force -Path $S | Out-Null
  git -C $S init -q
  git -C $S config core.autocrlf false
  git -C $S remote add origin https://github.com/ggml-org/llama.cpp
  git -C $S fetch -q --depth 1 origin $BASE
  git -C $S checkout -q --detach $BASE
  if ((git -C $S rev-parse HEAD).Trim() -ne $BASE) { Write-Host "FATAL: checkout $n"; exit 1 }
  & $PY -c "import gzip,shutil,sys; shutil.copyfileobj(gzip.open(sys.argv[1]), open(sys.argv[2],'wb'))" "$J\$n.patch.gz" "$W\$n.patch"
  git -C $S apply --whitespace=nowarn "$W\$n.patch"
  if ($LASTEXITCODE -ne 0) { Write-Host "FATAL: $n patch does not apply"; exit 1 }
  git -C $S diff --shortstat | ForEach-Object { Log "$n diff $_" }
  if (-not (Build-Tree $S $n $true)) { Write-Host "FATAL: $n build"; exit 1 }
  $sha = (Get-Content "$J\$($n.ToUpper())_SHA").Substring(0, 10)
  Set-Content -Encoding utf8 -Path "$R\bins\$n\TAG" -Value "$n $sha hip gfx1151 TheRock $TRV"
}
Get-ChildItem "C:\Windows\System32\amdhip64*" -ErrorAction SilentlyContinue | ForEach-Object { Log "system32: $($_.Name) $($_.VersionInfo.FileVersion)" }
Get-ChildItem -Recurse "$R\bins\rel" -Filter "amdhip64*.dll" | ForEach-Object { Log "release: $($_.FullName) $($_.VersionInfo.FileVersion)" }
Get-ChildItem "$R\bins\k9" -Filter "amdhip64*.dll" | ForEach-Object { Log "k9: $($_.Name) $($_.VersionInfo.FileVersion)" }

$dl.WaitForExit()
Get-Content "$OUT\dl.log", "$OUT\dl.err" -ErrorAction SilentlyContinue | Where-Object { -not $_.StartsWith("ROW") } | Select-Object -Last 5
# a server per arm must see ROCm0 (a bundle that falls back to the CPU would measure nothing)
$op = $env:PATH
foreach ($n in "rel", "k8", "k9") { $sv = Get-ChildItem -Recurse "$R\bins\$n" -Filter "llama-server.exe" | Select-Object -First 1; $env:PATH = "$($sv.DirectoryName);$op"; $c = (& $sv.FullName --list-devices 2>&1 | Select-String "ROCm0").Count; Log "devices $n : ROCm0 x$c"; if ($c -lt 1) { Write-Host "FATAL: $n sees no ROCm0"; exit 1 } }
$env:PATH = $op
& $PY -u "$J\k6p_plan.py" "$OUT\plan"
Copy-Item "$R\out\rows.jsonl" "$OUT\rows_win.jsonl" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path "$OUT\logs" | Out-Null
Copy-Item "$R\out\logs\*.filtered.log" "$OUT\logs\" -ErrorAction SilentlyContinue
if (-not (Test-Path "$OUT\rows_win.jsonl")) { exit 1 }
