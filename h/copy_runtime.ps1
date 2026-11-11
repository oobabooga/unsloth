# Verbatim 'Copy the CUDA runtime into the bundle' step from unslothai/llama.cpp PR 362 @ 24a7ce78bd
$bin = Join-Path $env:CUDA_PATH 'bin'
foreach ($pattern in @('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll')) {
  $hits = @(Get-ChildItem -Path $bin -Recurse -Filter $pattern | Where-Object { $_.DirectoryName -notmatch '\\arm64$' })
  if ($hits.Count -ne 1) { throw "expected one $pattern under $bin, found $($hits.Count)" }
  Copy-Item $hits[0].FullName src/build/bin/ -Force
}
