# win_gpu.ps1 (k15): gfx1151 Windows 11. Build k10 (ff5a2337 + k10.patch.gz = a162dce854) and k15 (+ k15d.patch.gz = 95b695923af) with
# TheRock (the release's weekly pin) and the release's Windows ROCm flags (unsloth-prebuilt-rocm.yml windows leg: shared libs,
# GGML_BACKEND_DL, CPU_ALL_VARIANTS, RPC, OPENMP off, ROCWMMA FA off, GGML_HIP_GRAPHS at its default), tests ON for both (test-backend-ops
# is the base arm of the existing cases), TheRock's DLLs + rocblas / hipblaslt kernels beside the exes (as the release does), then the
# k15_plan.py matrix (K15_STEPS from the workflow). Rows -> out\rows_win.jsonl. Writes only under $env:AMD_CI_WORK (= RUNNER_TEMP).
$ErrorActionPreference = "Continue"
$min = if ($env:K15_STEP_MIN) { [int]$env:K15_STEP_MIN } else { 350 }
$env:K15_DEADLINE = "$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + $min * 60 - 300)"
$W = $env:AMD_CI_WORK; $OUT = "$W\out"; $J = "$env:GITHUB_WORKSPACE\k15"; $R = "$W\k15"
New-Item -ItemType Directory -Force -Path $OUT, "$R\bins", "$R\models" | Out-Null
function Log($s) { Write-Host $s; Add-Content -Encoding utf8 -Path "$OUT\win_diag.txt" -Value $s }
Write-Host "RUNNER_OS='$env:RUNNER_OS' RUNNER_NAME='$env:RUNNER_NAME'"
$PY = $null
foreach ($c in @("C:\Program Files\Python312\python.exe", "C:\Program Files\Python311\python.exe")) { if (Test-Path -LiteralPath $c) { $PY = $c; break } }
if ($null -eq $PY) { Write-Host "FATAL: no system Python"; exit 1 }
Log "python $PY"
Get-CimInstance Win32_ComputerSystem | ForEach-Object { Log "RAM $($_.TotalPhysicalMemory) procs $($_.NumberOfLogicalProcessors)" }
Get-CimInstance Win32_Processor | ForEach-Object { Log "CPU $($_.Name)" }
Get-CimInstance Win32_VideoController | ForEach-Object { Log "GPU $($_.Name) driver $($_.DriverVersion) AdapterRAM $($_.AdapterRAM)" }

$env:K6P_DIR = $R; $env:K6P_OUT = "$OUT\bench"; $env:K6P_DEV = "ROCm0"; $env:PYTHONIOENCODING = "utf-8"
$dl = Start-Process -FilePath $PY -PassThru -NoNewWindow -RedirectStandardOutput "$OUT\dl.log" -RedirectStandardError "$OUT\dl.err" `
  -ArgumentList @("-u", "`"$J\k15_bench.py`"", "dl", "--models", $(if ($env:K15_DL_MODELS) { $env:K15_DL_MODELS } else { "q35,q3m,g4e2b" }))

# TheRock toolchain: the dist tarball of the release's weekly pin (as the k9 / k10 / k10s runs)
$ROCM_VER = if ($env:ROCM_VER) { $env:ROCM_VER } else { "10.2.0a20261017" }
$t0 = Get-Date
& curl.exe -fL --retry 5 -s -S -o "$W\rocm.tar.gz" "https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-$ROCM_VER.tar.gz"
Log "TheRock tarball rc=$LASTEXITCODE size $((Get-Item "$W\rocm.tar.gz").Length)"
$ROCM = "$W\rocm"
New-Item -ItemType Directory -Force -Path $ROCM | Out-Null
tar.exe -xzf "$W\rocm.tar.gz" -C $ROCM --strip-components=1
Remove-Item -Force "$W\rocm.tar.gz"
Log "TheRock $ROCM_VER extracted in $([int]((Get-Date) - $t0).TotalSeconds) s to $ROCM"
$CLANG = "$ROCM\lib\llvm\bin\clang.exe"; $CLANGXX = "$ROCM\lib\llvm\bin\clang++.exe"
if (-not (Test-Path $CLANG)) { Write-Host "FATAL: no clang at $CLANG"; exit 1 }
$vsp = (& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
$vc = if ($vsp) { "$vsp\VC\Auxiliary\Build\vcvars64.bat" } else { "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" }
Log "vcvars $vc"
cmd /c "`"$vc`" >nul 2>&1 && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
$env:HIP_PATH = $ROCM; $env:ROCM_PATH = $ROCM; $env:HIP_PLATFORM = "amd"
$env:PATH = "$ROCM\bin;$ROCM\lib\llvm\bin;$env:PATH"
Log ((& $CLANG --version) -join " | ")

$BASE = (Get-Content -Encoding utf8 "$J\UP_SHA").Trim()
$ocml = Get-ChildItem -Path $ROCM -Recurse -Filter "ocml.bc" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $ocml) { Write-Host "FATAL: no ocml.bc"; exit 1 }
$ROCMF = $ROCM.Replace('\', '/'); $BCF = $ocml.DirectoryName.Replace('\', '/')
$env:HIP_DEVICE_LIB_PATH = $ocml.DirectoryName
$HFLAGS = "--rocm-path=$ROCMF --rocm-device-lib-path=$BCF -Wno-unused-command-line-argument"
# one release-flag build of a source tree into $R\bins\<name>, TheRock runtime + rocblas / hipblaslt kernels beside the exes
function Build-Tree($tree, $name, $ctest) {
  $t0 = Get-Date
  cmake -S $tree -B "$tree\build" -G Ninja -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_C_COMPILER=$CLANG" "-DCMAKE_CXX_COMPILER=$CLANGXX" "-DCMAKE_HIP_COMPILER=$CLANGXX" `
    "-DCMAKE_PREFIX_PATH=$ROCM" "-DROCM_DIR=$ROCM" "-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types" `
    "-DCMAKE_CXX_FLAGS=-I$ROCMF/include $HFLAGS" "-DCMAKE_HIP_FLAGS=$HFLAGS" `
    -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF `
    -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_RPC=ON -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_NATIVE=OFF `
    -DGGML_STATIC=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF *> "$OUT\${name}_configure.log"
  if ($LASTEXITCODE -ne 0) { Get-Content "$OUT\${name}_configure.log" -Tail 25; Log "$name configure FAILED"; return $false }
  Get-Content -Encoding utf8 "$tree\build\CMakeCache.txt" | Select-String -Pattern "^GGML_HIP_GRAPHS|^GGML_HIP:" | ForEach-Object { Log "$name cache: $_" }
  cmake --build "$tree\build" -j 32 -- -k 0 *> "$OUT\${name}_build.log"
  $brc = $LASTEXITCODE
  Log "$name build rc=$brc $([int]((Get-Date) - $t0).TotalSeconds) s"
  if ($brc -ne 0) { Get-Content "$OUT\${name}_build.log" | Select-String -Pattern "error:|FAILED" -Context 2, 6 | Select-Object -First 40; return $false }
  if ($ctest) {
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
  # the release's kpack step (unslothai/unsloth#13663): rocBLAS / hipBLASLt kernels from $ROCM\.kpack into rocblas\k, references rewritten
  & curl.exe -fsSL -o "$W\relocate_kpack.py" "https://raw.githubusercontent.com/unslothai/llama.cpp/f95b52162ed4b916a58d4f16b72662483cc55075/scripts/unsloth/relocate_kpack.py"
  & $PY "$W\relocate_kpack.py" --bundle $d --kpack-src "$ROCM\.kpack" --archs gfx1151 *>> "$OUT\win_diag.txt"
  Log "$name kpack rc=$LASTEXITCODE $((Get-ChildItem "$d\rocblas\k" -ErrorAction SilentlyContinue | ForEach-Object { $_.Name }) -join ' ')"
  if ($LASTEXITCODE -ne 0) { return $false }
  foreach ($t in "test-backend-ops.exe", "test-graph-src-type.exe") { Log "$name $t present=$(Test-Path "$d\$t")" }
  return $true
}
$TREES = Get-Content -Encoding utf8 "$J\TREES"
foreach ($n in "k10", "k15") {
  $S = "$W\src_$n"
  New-Item -ItemType Directory -Force -Path $S | Out-Null
  git -C $S init -q
  git -C $S config core.autocrlf false
  git -C $S remote add origin https://github.com/ggml-org/llama.cpp
  git -C $S fetch -q --depth 1 origin $BASE
  git -C $S checkout -q --detach $BASE
  if ((git -C $S rev-parse HEAD).Trim() -ne $BASE) { Write-Host "FATAL: checkout $n"; exit 1 }
  $patches = if ($n -eq "k10") { @("k10") } else { @("k10", "k15d") }
  foreach ($p in $patches) {
    & $PY -c "import gzip,shutil,sys; shutil.copyfileobj(gzip.open(sys.argv[1]), open(sys.argv[2],'wb'))" "$J\$p.patch.gz" "$W\$p.patch"
    git -C $S apply --index --whitespace=nowarn "$W\$p.patch"
    if ($LASTEXITCODE -ne 0) { Write-Host "FATAL: $p patch does not apply ($n)"; exit 1 }
  }
  $want = (Get-Content -Encoding utf8 "$J\$($n.ToUpper())_SHA").Trim()
  $exp = (($TREES | Where-Object { $_.StartsWith("$want ") }) -split " ")[1]
  $tree = (git -C $S write-tree).Trim()
  Log "$n tree $tree expect $exp ($($want.Substring(0, 11)))"
  if ($tree -ne $exp) { Write-Host "FATAL: the applied $n tree is not $want"; exit 1 }
  $hm = (Select-String -Path "$S\ggml\src\ggml-cuda\vendors\hip.h" -Pattern "^#define cudaErrorInvalidValue hipErrorInvalidValue").Count
  Log "$n hip.h cudaErrorInvalidValue mapping lines: $hm"
  if (-not (Build-Tree $S $n ($n -eq "k15"))) { Write-Host "FATAL: $n build"; exit 1 }
  Set-Content -Encoding utf8 -Path "$R\bins\$n\TAG" -Value "$n $($want.Substring(0, 11)) hip gfx1151 TheRock $ROCM_VER"
}
Get-ChildItem "C:\Windows\System32\amdhip64*" -ErrorAction SilentlyContinue | ForEach-Object { Log "system32: $($_.Name) $($_.VersionInfo.FileVersion)" }
Get-ChildItem "$R\bins\k15" -Filter "amdhip64*.dll" | ForEach-Object { Log "k15: $($_.Name) $($_.VersionInfo.FileVersion)" }

$dl.WaitForExit()
Get-Content "$OUT\dl.log", "$OUT\dl.err" -ErrorAction SilentlyContinue | Where-Object { -not $_.StartsWith("ROW") } | Select-Object -Last 5
# a server per arm must see ROCm0 (a bundle that falls back to the CPU would measure nothing)
$op = $env:PATH
foreach ($n in "k10", "k15") { $sv = Get-ChildItem -Recurse "$R\bins\$n" -Filter "llama-server.exe" | Select-Object -First 1; $env:PATH = "$($sv.DirectoryName);$op"; $c = (& $sv.FullName --list-devices 2>&1 | Select-String "ROCm0").Count; Log "devices $n : ROCm0 x$c"; if ($c -lt 1) { Write-Host "FATAL: $n sees no ROCm0"; exit 1 } }
$env:PATH = $op
& $PY -u "$J\k15_plan.py" "$OUT\plan"
if (-not (Test-Path "$OUT\bench\rows.jsonl")) { exit 1 }
