# 'Install CUDA toolkit' step from unsloth-cuda-windows-setup @ master 4b9f01b527; inputs.cuda comes from $env:CUDA_INPUT
$ErrorActionPreference = 'Stop'
switch ("$($env:CUDA_INPUT)") {
  '12.8.0' { $components = @('cuda_cudart:12.8.57', 'cuda_nvcc:12.8.61', 'cuda_nvrtc:12.8.61', 'libcublas:12.8.3.14', 'cuda_nvtx:12.8.55', 'cuda_profiler_api:12.8.55', 'cuda_cccl:12.8.55', 'visual_studio_integration:12.8.55') }
  '13.3'   { $components = @('cuda_crt:13.3.33', 'cuda_cudart:13.3.29', 'cuda_nvcc:13.3.33', 'cuda_nvrtc:13.3.33', 'libcublas:13.5.1.27', 'libnvvm:13.3.33', 'cuda_nvtx:13.3.29', 'cuda_profiler_api:13.3.27', 'cccl:13.3.3.3.1', 'visual_studio_integration:13.3.27') }
  default  { throw "no redist component table for CUDA $($env:CUDA_INPUT)" }
}
$line = "$($env:CUDA_INPUT)" -replace '^(\d+\.\d+).*$', '$1'
$prefix = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$line"
$stage = Join-Path $env:RUNNER_TEMP 'cuda-redist'
New-Item -ItemType Directory -Force -Path $prefix, $stage | Out-Null
foreach ($cv in $components) {
  $comp, $ver = $cv.Split(':')
  $name = "$comp-windows-x86_64-$ver-archive"
  $zip = Join-Path $stage "$name.zip"
  curl.exe -fsSL --retry 3 -o $zip "https://developer.download.nvidia.com/compute/cuda/redist/$comp/windows-x86_64/$name.zip"
  if ($LASTEXITCODE -ne 0) { throw "download failed: $name" }
  tar.exe -xf $zip -C $stage
  if ($LASTEXITCODE -ne 0) { throw "extract failed: $name" }
  robocopy (Join-Path $stage $name) $prefix /E /NFL /NDL /NJH /NJS /NP | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "copy failed: $name ($LASTEXITCODE)" }
  Remove-Item -Recurse -Force $zip, (Join-Path $stage $name)
}
"$prefix\bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
"CUDA_PATH=$prefix" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"CUDA_HOME=$prefix" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"CUDA_PATH_V$($line.Replace('.', '_'))=$prefix" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
exit 0
