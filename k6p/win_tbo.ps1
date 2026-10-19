# win_tbo.ps1: Windows gfx1151 test-backend-ops MUL_MAT differential (run 5). Run 4: k6's MUL_MAT suite aborted (0xC0000409) in
# hipblasSgemm with rocBLAS 'hipErrorInvalidKernelFile'. Same TheRock 10.2.0a20261010 for both trees: rs = release source + tests,
# k6 = b11721 + k6.patch.gz + tests; arms k6 / rs with their own rocblas\library, and k6 with the release bundle's rocblas\library.
$ErrorActionPreference = "Continue"
$W = $env:AMD_CI_WORK; $OUT = "$W\out"; $J = "$env:GITHUB_WORKSPACE\k6p"; $R = "$W\k6p"
New-Item -ItemType Directory -Force -Path $OUT, "$R\bins\rel" | Out-Null
function Log($s) { Write-Host $s; Add-Content -Encoding utf8 -Path "$OUT\win_diag.txt" -Value $s }
$PY = "C:\Program Files\Python312\python.exe"
$TAG = "b11696-mix-7baeb9a"
& curl.exe -fL --retry 5 -s -S -o "$R\rel.zip" "https://github.com/unslothai/llama.cpp/releases/download/$TAG/app-$TAG-windows-x64-rocm-gfx1151.zip"
Expand-Archive -Force -Path "$R\rel.zip" -DestinationPath "$R\bins\rel"
$ROCM_VER = "10.2.0a20261010"
& curl.exe -fL --retry 5 -s -S -o "$W\rocm.tar.gz" "https://nightly.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-$ROCM_VER.tar.gz"
$ROCM = "$W\rocm"; New-Item -ItemType Directory -Force -Path $ROCM | Out-Null
tar.exe -xzf "$W\rocm.tar.gz" -C $ROCM --strip-components=1
Get-ChildItem -Path $ROCM -Recurse -Directory -Filter "library" -ErrorAction SilentlyContinue | Where-Object { $_.Parent.Name -in @("rocblas", "hipblaslt") } | ForEach-Object { Log "rocm library dir: $($_.FullName) files $((Get-ChildItem $_.FullName -Recurse -File).Count)" }
Get-ChildItem -Recurse -Directory "$R\bins\rel" -Filter "library" | ForEach-Object { Log "release library dir: $($_.FullName) files $((Get-ChildItem $_.FullName -Recurse -File).Count)" }
$CLANG = "$ROCM\lib\llvm\bin\clang.exe"; $CLANGXX = "$ROCM\lib\llvm\bin\clang++.exe"
$vsp = (& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
cmd /c "`"$vsp\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
$env:HIP_PATH = $ROCM; $env:ROCM_PATH = $ROCM; $env:HIP_PLATFORM = "amd"; $env:PATH = "$ROCM\bin;$ROCM\lib\llvm\bin;$env:PATH"
$ocml = Get-ChildItem -Path $ROCM -Recurse -Filter "ocml.bc" -ErrorAction SilentlyContinue | Select-Object -First 1
$ROCMF = $ROCM.Replace('\', '/'); $BCF = $ocml.DirectoryName.Replace('\', '/')
$HFLAGS = "--rocm-path=$ROCMF --rocm-device-lib-path=$BCF -Wno-unused-command-line-argument"
function Build-Tbo($tree, $name) {
  cmake -S $tree -B "$tree\build" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_C_COMPILER=$CLANG" "-DCMAKE_CXX_COMPILER=$CLANGXX" `
    "-DCMAKE_HIP_COMPILER=$CLANGXX" "-DCMAKE_PREFIX_PATH=$ROCM" "-DROCM_DIR=$ROCM" "-DCMAKE_C_FLAGS=-Wno-error=incompatible-pointer-types" `
    "-DCMAKE_CXX_FLAGS=-I$ROCMF/include $HFLAGS" "-DCMAKE_HIP_FLAGS=$HFLAGS" -DGPU_TARGETS=gfx1151 -DBUILD_SHARED_LIBS=ON `
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_RPC=ON -DGGML_HIP_ROCWMMA_FATTN=OFF `
    -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF *> "$OUT\${name}_cfg.log"
  cmake --build "$tree\build" -j 32 -- -k 0 *> "$OUT\${name}_build.log"
  Log "$name tbo build rc=$LASTEXITCODE"
  $d = "$R\bins\$name"; New-Item -ItemType Directory -Force -Path $d | Out-Null
  Copy-Item "$tree\build\bin\*" "$d\" -Recurse -Force
  Copy-Item "$ROCM\bin\*.dll" "$d\" -Force
  foreach ($lib in "rocblas", "hipblaslt") {
    $ld = "$ROCM\bin\$lib\library"
    if (Test-Path $ld) { New-Item -ItemType Directory -Force -Path "$d\$lib" | Out-Null; Copy-Item -Path $ld -Destination "$d\$lib\library" -Recurse -Force; Log "$name $lib from $ld" }
  }
}
$BASE = "ff5a23376725e5a889fa0bfb987b7541dc304da0"
$SRC = "$W\src_k6"; New-Item -ItemType Directory -Force -Path $SRC | Out-Null
git -C $SRC init -q; git -C $SRC config core.autocrlf false; git -C $SRC remote add origin https://github.com/ggml-org/llama.cpp
git -C $SRC fetch -q --depth 1 origin $BASE; git -C $SRC checkout -q --detach $BASE
& $PY -c "import gzip,shutil,sys; shutil.copyfileobj(gzip.open(sys.argv[1]), open(sys.argv[2],'wb'))" "$J\k6.patch.gz" "$W\k6.patch"
git -C $SRC apply --whitespace=nowarn "$W\k6.patch"
Build-Tbo $SRC "k6"
& curl.exe -fL --retry 5 -s -S -o "$W\rs.tar.gz" "https://github.com/unslothai/llama.cpp/releases/download/$TAG/llama.cpp-source-$TAG.tar.gz"
New-Item -ItemType Directory -Force -Path "$W\src_rs" | Out-Null
tar.exe -xzf "$W\rs.tar.gz" -C "$W\src_rs"
Build-Tbo (Get-ChildItem -Directory "$W\src_rs" | Select-Object -First 1).FullName "rs"
$relLib = (Get-ChildItem -Recurse -Directory "$R\bins\rel" -Filter "library" | Where-Object { $_.Parent.Name -eq "rocblas" } | Select-Object -First 1).FullName
$env:K6P_DIR = $R; $env:K6P_DEV = "ROCm0"; $env:PYTHONIOENCODING = "utf-8"
$env:ARMS = "rel:rel,k6:k6,rs:rs,k6r:k6:ROCBLAS_TENSILE_LIBPATH=$relLib"
foreach ($b in "k6", "rs", "k6r") {
  & $PY -u "$J\k6p_bench.py" tbo --bin $b --ops MUL_MAT --timeout 3000 *> "$OUT\tbo_$b.log"
  Get-Content "$OUT\tbo_$b.log" | Where-Object { $_ -match "TBO" } | ForEach-Object { Log $_ }
}
Copy-Item "$R\out\rows.jsonl" "$OUT\rows_wintbo.jsonl" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path "$OUT\logs" | Out-Null
Copy-Item "$R\out\logs\*.filtered.log" "$OUT\logs\" -ErrorAction SilentlyContinue
if (-not (Test-Path "$OUT\rows_wintbo.jsonl")) { exit 1 }
