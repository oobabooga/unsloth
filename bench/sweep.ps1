# Vulkan-on-Windows prefill sweep for issue 10656. One llama-bench per arm, arms interleaved
# with a ROCm reference so drift shows. JSON per arm; stderr kept for the device line.
param([string]$Out, [string]$Vk, [string]$Ro, [string]$Gemma, [string]$Qwen)
$ErrorActionPreference = "Continue"
$arms = @(
  @("gemma_rocm_fa1",        $Ro, $Gemma, "-fa 1", @{}),
  @("gemma_vk_fa1",          $Vk, $Gemma, "-fa 1", @{}),
  @("gemma_vk_fa0",          $Vk, $Gemma, "-fa 0", @{}),
  @("gemma_vk_fa1_nocoopmat",$Vk, $Gemma, "-fa 1", @{GGML_VK_DISABLE_COOPMAT="1"}),
  @("gemma_vk_fa1_ub2048",   $Vk, $Gemma, "-fa 1 -b 2048 -ub 2048", @{}),
  @("gemma_vk_fa1_nofasparse",$Vk, $Gemma, "-fa 1", @{GGML_VK_FA_SPARSE_DISABLE="1"}),
  @("gemma_rocm_fa1_b",      $Ro, $Gemma, "-fa 1", @{}),
  @("gemma_vk_fa1_b",        $Vk, $Gemma, "-fa 1", @{}),
  @("qwen_rocm_fa1",         $Ro, $Qwen,  "-fa 1", @{}),
  @("qwen_vk_fa1",           $Vk, $Qwen,  "-fa 1", @{}),
  @("qwen_vk_fa0",           $Vk, $Qwen,  "-fa 0", @{})
)
foreach ($a in $arms) {
  $name = $a[0]; $bin = $a[1]; $model = $a[2]; $flags = $a[3].Split(" "); $envs = $a[4]
  foreach ($k in $envs.Keys) { Set-Item -Path "env:$k" -Value $envs[$k] }
  Write-Host "=== $name ($($a[3]) $(($envs.Keys | ForEach-Object { "$_=$($envs[$_])" }) -join ' '))"
  $t0 = Get-Date
  & "$bin\llama-bench.exe" -m $model -ngl 99 @flags -p 512,4096 -n 128 -d 0,16384 -r 2 -o json 2> "$Out\$name.err" | Out-File -FilePath "$Out\$name.json" -Encoding utf8
  Write-Host "exit $LASTEXITCODE in $([int]((Get-Date) - $t0).TotalSeconds)s"
  foreach ($k in $envs.Keys) { Remove-Item -Path "env:$k" -ErrorAction SilentlyContinue }
}
# Per-op timings for the slow case: where does Vulkan spend pp512 at depth?
foreach ($fa in @("1", "0")) {
  $env:GGML_VK_PERF_LOGGER = "1"
  & "$Vk\llama-bench.exe" -m $Gemma -ngl 99 -fa $fa -p 512 -n 0 -d 16384 -r 1 -o json 2> "$Out\perflog_gemma_vk_fa$fa.err" | Out-File -FilePath "$Out\perflog_gemma_vk_fa$fa.json" -Encoding utf8
  Remove-Item env:GGML_VK_PERF_LOGGER
}
