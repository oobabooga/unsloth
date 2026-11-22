# Criteria for ggml-patches/0070 (Vulkan AMD picks: llama.cpp #31587 large coopmat tile on AMD proprietary 2.0.416+, #30280 RDNA3/4 split-K)

Judge from the probe outputs (BASE and HEAD built in the same job on the same box).

Which pick can engage (read vulkaninfo.txt):
- Linux / RADV (driverID = DRIVER_ID_MESA_RADV): only #30280 (split-K 3 when tiles in (cores, 4/3 cores]); gfx1151 is
  classified AMD_RDNA3 by ggml (min/max subgroup 32/64, mixed-signedness int8 dot accelerated). #31587 cannot change
  anything on RADV (its gate is the proprietary driver only).
- Windows / AMD proprietary (DRIVER_ID_AMD_PROPRIETARY): #31587 engages only when driverVersion >> 22 > 2 or
  (== 2 and driverVersion & 0x3fffff >= 416), i.e. Adrenalin 26.10.1+. Record the driverVersion; below that HEAD == BASE
  for the tile pick and the run proves only no-regression.

PASS (merge) when all hold:
1. Correctness: test_head.txt shows the same MUL_MAT pass count as test_base.txt and no FAIL lines that BASE does not have.
2. No regression: for every MUL_MAT perf case, median HEAD us/run <= 1.03 x median BASE us/run, where the noise floor is
   the spread of the two BASE runs (a delta inside it is a tie).
3. Gain (optional for merge, required to claim a speedup): on Windows with driver >= 2.0.416, the large-matmul cases
   (m, n >= 2048) improve beyond the BASE-vs-BASE spread. On RADV, expect ties except shapes whose 128x128 tile count
   lands in (cores, 4/3 cores] (small matmuls); diffusion-size DiT matmuls are far past that, so a tie there is the
   expected Linux result.

FAIL (do not merge) on any correctness difference, or any case slower beyond 3% plus the BASE spread.

Windows variant: run the same steps in PowerShell (vulkaninfoSDK / vulkaninfo --summary; cmake with -DGGML_VULKAN=ON
needs the Vulkan SDK for glslc), or reuse the fork's Windows Vulkan prebuilt job with the patch applied.
