# win_gpu.ps1: gfx1151 Windows 11. Build k6 (b11721 + k6.patch.gz) with TheRock (pip, gfx1151 nightly index) and the release's
# Windows ROCm flags (unsloth-prebuilt-rocm.yml windows leg: shared libs, GGML_BACKEND_DL, CPU_ALL_VARIANTS, RPC, OPENMP off,
# ROCWMMA FA off), ship TheRock's DLLs + rocblas/hipblaslt libraries beside the exes (as the release does), then measure release
# b11696-mix-7baeb9a windows-x64-rocm-gfx1151 (rel / rel2) vs k6 (k6 / k6b) with k6p_bench.py. Rows -> out\rows_win.jsonl
$ErrorActionPreference = "Continue"
$W = $env:AMD_CI_WORK; $OUT = "$W\out"; $J = "$env:GITHUB_WORKSPACE\k6p"; $R = "$W\k6p"
New-Item -ItemType Directory -Force -Path $OUT, "$R\bins\rel", "$R\bins\k6", "$R\models" | Out-Null
function Log($s) { Write-Host $s; Add-Content -Encoding utf8 -Path "$OUT\win_diag.txt" -Value $s }
Write-Host "RUNNER_OS='$env:RUNNER_OS' RUNNER_NAME='$env:RUNNER_NAME'"
$PY = $null
foreach ($c in @("C:\Program Files\Python312\python.exe", "C:\Program Files\Python311\python.exe")) { if (Test-Path -LiteralPath $c) { $PY = $c; break } }
if ($null -eq $PY) { Write-Host "FATAL: no system Python"; exit 1 }
Log "python $PY"
Get-CimInstance Win32_ComputerSystem | ForEach-Object { Log "RAM $($_.TotalPhysicalMemory) procs $($_.NumberOfLogicalProcessors)" }
Get-CimInstance Win32_VideoController | ForEach-Object { Log "GPU $($_.Name) driver $($_.DriverVersion) AdapterRAM $($_.AdapterRAM)" }

$env:K6P_DIR = $R; $env:K6P_DEV = "ROCm0"; $env:ARMS = "rel:rel,rel2:rel,k6:k6,k6b:k6"; $env:PYTHONIOENCODING = "utf-8"
$dl = Start-Process -FilePath $PY -PassThru -NoNewWindow -RedirectStandardOutput "$OUT\dl.log" -RedirectStandardError "$OUT\dl.err" `
  -ArgumentList @("-u", "`"$J\k6p_bench.py`"", "dl", "--models", "q35,lfm2,gptoss")

# release bundle
$TAG = "b11696-mix-7baeb9a"
& curl.exe -fL --retry 5 -s -S -o "$R\rel.zip" "https://github.com/unslothai/llama.cpp/releases/download/$TAG/app-$TAG-windows-x64-rocm-gfx1151.zip"
Log "release zip rc=$LASTEXITCODE"
Expand-Archive -Force -Path "$R\rel.zip" -DestinationPath "$R\bins\rel"
Set-Content -Encoding utf8 -Path "$R\bins\rel\TAG" -Value "$TAG windows-x64-rocm-gfx1151"

# TheRock toolchain
& $PY -m venv "$W\venv"
$VP = "$W\venv\Scripts\python.exe"
$t0 = Get-Date
& $VP -m pip install -q --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ "rocm[libraries,devel]" *> "$OUT\therock_pip.log"
if ($LASTEXITCODE -ne 0) { Get-Content "$OUT\therock_pip.log" -Tail 20; Write-Host "FATAL: TheRock pip"; exit 1 }
& "$W\venv\Scripts\rocm-sdk.exe" init *> "$OUT\therock_init.log"
$ROCM = (& "$W\venv\Scripts\rocm-sdk.exe" path --root).Trim()
Log "TheRock $([int]((Get-Date) - $t0).TotalSeconds) s root $ROCM"
& $VP -m pip list 2>$null | Select-String -Pattern "rocm" | ForEach-Object { Log "$_" }
$CLANG = "$ROCM\lib\llvm\bin\clang.exe"; $CLANGXX = "$ROCM\lib\llvm\bin\clang++.exe"
if (-not (Test-Path $CLANG)) { Write-Host "FATAL: no clang at $CLANG"; exit 1 }
$vc = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vc`" >nul 2>&1 && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
$env:HIP_PATH = $ROCM; $env:ROCM_PATH = $ROCM; $env:HIP_PLATFORM = "amd"
$env:PATH = "$ROCM\bin;$ROCM\lib\llvm\bin;$env:PATH"
Log ((& $CLANG --version) -join " | ")

# source = upstream b11721 + k6 patch
$BASE = "ff5a23376725e5a889fa0bfb987b7541dc304da0"
$SRC = "$W\src_k6"
New-Item -ItemType Directory -Force -Path $SRC | Out-Null
git -C $SRC init -q
git -C $SRC config core.autocrlf false
git -C $SRC remote add origin https://github.com/ggml-org/llama.cpp
git -C $SRC fetch -q --depth 1 origin $BASE
git -C $SRC checkout -q --detach $BASE
if ((git -C $SRC rev-parse HEAD).Trim() -ne $BASE) { Write-Host "FATAL: checkout"; exit 1 }
& $PY -c "import gzip,shutil,sys; shutil.copyfileobj(gzip.open(sys.argv[1]), open(sys.argv[2],'wb'))" "$J\k6.patch.gz" "$W\k6.patch"
git -C $SRC apply --whitespace=nowarn "$W\k6.patch"
if ($LASTEXITCODE -ne 0) { Write-Host "FATAL: k6 patch does not apply"; exit 1 }
git -C $SRC diff --shortstat | ForEach-Object { Log "k6 diff $_" }

$ocml = Get-ChildItem -Path $ROCM -Recurse -Filter "ocml.bc" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $ocml) { Write-Host "FATAL: no ocml.bc"; exit 1 }
$ROCMF = $ROCM.Replace('\', '/'); $BCF = $ocml.DirectoryName.Replace('\', '/')
$env:HIP_DEVICE_LIB_PATH = $ocml.DirectoryName
$HFLAGS = "--rocm-path=$ROCMF --rocm-device-lib-path=$BCF -Wno-unused-command-line-argument"
$t0 = Get-Date
cmake -S $SRC -B "$SRC\build" -G Ninja -DCMAKE_BUILD_TYPE=Release `
  "-DCMAKE_C_COMPILER=$CLANG" "-DCMAKE_CXX_COMPILER=$CLANGXX" "-DCMAKE_HIP_COMPILER=$CLANGXX" `
  "-DCMAKE_PREFIX_PATH=$ROCM" "-DROCM_DIR=$ROCM" "-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types" `
  "-DCMAKE_CXX_FLAGS=-I$ROCMF/include $HFLAGS" "-DCMAKE_HIP_FLAGS=$HFLAGS" `
  -DGPU_TARGETS=gfx1151 -DAMDGPU_TARGETS=gfx1151 -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF `
  -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_RPC=ON -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_NATIVE=OFF `
  -DGGML_STATIC=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF *> "$OUT\k6_configure.log"
$crc = $LASTEXITCODE
Get-Content "$OUT\k6_configure.log" -Tail 25
if ($crc -ne 0) { Write-Host "FATAL: configure"; exit 1 }
cmake --build "$SRC\build" -j 32 *> "$OUT\k6_build.log"
$brc = $LASTEXITCODE
Log "k6 build rc=$brc $([int]((Get-Date) - $t0).TotalSeconds) s"
if ($brc -ne 0) { Get-Content "$OUT\k6_build.log" | Select-String -Pattern "error:|FAILED" -Context 2, 6 | Select-Object -First 40; exit 1 }
cmake -S $SRC -B "$SRC\build" -DLLAMA_BUILD_TESTS=ON *>> "$OUT\k6_configure.log"
cmake --build "$SRC\build" -j 32 --target test-backend-ops *> "$OUT\k6_build_tbo.log"
Log "test-backend-ops build rc=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { Get-Content "$OUT\k6_build_tbo.log" | Select-String -Pattern "error:|FAILED" -Context 2, 6 | Select-Object -First 20 }
Copy-Item "$SRC\build\bin\*" "$R\bins\k6\" -Recurse -Force
# TheRock runtime beside the exes (Windows searches the exe dir before System32), rocblas / hipblaslt kernels in bin\<lib>\library
Copy-Item "$ROCM\bin\*.dll" "$R\bins\k6\" -Force
foreach ($lib in "rocblas", "hipblaslt") {
  $d = Get-ChildItem -Path $ROCM -Recurse -Directory -Filter "library" -ErrorAction SilentlyContinue | Where-Object { $_.Parent.Name -eq $lib } | Select-Object -First 1
  if ($d) { New-Item -ItemType Directory -Force -Path "$R\bins\k6\$lib" | Out-Null; Copy-Item -Path $d.FullName -Destination "$R\bins\k6\$lib\library" -Recurse -Force; Log "$lib library from $($d.FullName)" }
}
Set-Content -Encoding utf8 -Path "$R\bins\k6\TAG" -Value "k6 $((Get-Content "$J\K6_SHA").Substring(0,10)) hip gfx1151 TheRock $((& $VP -m pip show rocm 2>$null | Select-String '^Version').ToString())"
# the release bundle's own rocblas library dir, and ours, are found next to rocblas.dll; no global env so each arm uses its own
Get-ChildItem "C:\Windows\System32\amdhip64*" -ErrorAction SilentlyContinue | ForEach-Object { Log "system32: $($_.Name) $($_.VersionInfo.FileVersion)" }
Get-ChildItem -Recurse "$R\bins\rel" -Filter "amdhip64*.dll" | ForEach-Object { Log "release: $($_.FullName) $($_.VersionInfo.FileVersion)" }
Get-ChildItem "$R\bins\k6" -Filter "amdhip64*.dll" | ForEach-Object { Log "k6: $($_.Name) $($_.VersionInfo.FileVersion)" }

$dl.WaitForExit()
Get-Content "$OUT\dl.log", "$OUT\dl.err" -ErrorAction SilentlyContinue | Where-Object { -not $_.StartsWith("ROW") }
function Bench($argv, $logname) {
  & $PY -u "$J\k6p_bench.py" @argv *> "$OUT\$logname"
  Write-Host "$logname rc=$LASTEXITCODE"
  Get-Content "$OUT\$logname" | Where-Object { -not $_.StartsWith("ROW") } | Select-Object -Last 14
}
Bench @("devinfo") "devinfo.log"
foreach ($c in @("q35:full", "lfm2:full", "lfm2:spill1536", "gptoss:full", "gptoss:spill6144")) {
  Bench @("time", "--cell", $c, "--reps", "3") "time_$($c.Replace(':','_')).log"
}
Bench @("kld", "--model", "q35", "--modes", "full", "--arms", "rel,rel2,k6") "kld_q35.log"
Bench @("kld", "--model", "lfm2", "--modes", "full,spill1536", "--arms", "rel,rel2,k6") "kld_lfm2.log"
Bench @("kld", "--model", "gptoss", "--modes", "full,spill6144", "--arms", "rel,rel2,k6") "kld_gptoss.log"
$OPS = "RMS_NORM,RMS_NORM_MUL_ADD,ADD_RMS_NORM,RMS_NORM_MUL_ROPE,MUL_MAT_VEC_SHARED_SRC1,MUL_MAT_ID_FALLBACK_SHARED_SRC1,MUL_MAT_ID_SHARED,MUL_MAT_VEC_FUSION,MUL_MAT_ID_FUSION,HC_POST_NORM,EW_CHAIN,TOPK_MOE,TOP_K,GET_ROWS,SET_ROWS,CPY,MUL_MAT_ID"
Bench @("tbo", "--bin", "k6", "--ops", $OPS, "--timeout", "2400") "tbo.log"
Copy-Item "$R\out\rows.jsonl" "$OUT\rows_win.jsonl" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path "$OUT\logs" | Out-Null
Copy-Item "$R\out\logs\*.filtered.log" "$OUT\logs\" -ErrorAction SilentlyContinue
if (-not (Test-Path "$OUT\rows_win.jsonl")) { exit 1 }
