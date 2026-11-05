#!/usr/bin/env python3
"""Write the Linux gfx1151 workflow for specs/amd_studio_vs_comfy.json into an amd_ci scaffold dir.

  python diffusion_bench/amd/gen_studio_vs_comfy_workflow.py --out ci_studio_vs_comfy

Five GPU jobs, chained with needs (+ if: always()) so only one of them ever waits on the shared GPU group:
  j1 Z-Image-Turbo + FLUX.2-klein-4B   j2 FLUX.1-schnell + SDXL   j3 Qwen-Image-2.1   j4 Wan2.2-TI2V-5B   j5 MiniMax-H3
Each job: states base = unsloth main at BASE_SHA, head = the same commit + amd_ci/patches/rocm_flash.patch (the local
Studio fix, applied on the runner; nothing of it is pushed to any unsloth repo); Studio via install.sh --local;
DAO-AILab flash-attention (Triton AMD backend, FA_SHA) into the Studio venv; ComfyUI at COMFY_SHA in two venvs (README
ROCm torch, and the gfx1151 torch Studio uses) + the same flash-attention; then differential.py over the job's cells.
No secret of any kind is referenced; HF_TOKEN never exists in the job.
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

BASE_SHAS = {"v2": "642da4cef9c1acd62c4c14ef74fd2e7317f4574a", "v3": "813d8af7ba1f1d90765b52a3dfbe17d4a0f6b9b6",
             "v4": "689e1b0cf5b814fe3d0431d05fbb80c97a7a91b3",
             "v5": "813d8af7ba1f1d90765b52a3dfbe17d4a0f6b9b6",
             "v6": "5971d280d4b645c8d470d6bb3171b082c4d4d2b8",
             "v7": "79116720a3f2d130ae84125126ffdb1f3bc564ed"}
COMFY_SHA = "a7169322485d0049380fb207fa17e9fb3ec40486"
FA_SHA = "bbe25bacff9f337568b66e69950e549df9839068"
BRANCH = "amd-ci-studio-vs-comfy"

JOB_SETS = {
    # id, title, base --only, head --state-only, criteria, defect, head patch, fixture args (None = no fixture)
    "v2": [
        ("j1_zimg_klein", "Z-Image-Turbo + FLUX.2-klein-4B", ["zimg_*", "klein_*"], ["zimg_s_*", "klein_s_*"],
         "differential", "slow:zimg_s_flash:zimg_s_def:0.9", "rocm_flash.patch", None),
        ("j2_flux1_sdxl", "FLUX.1-schnell + SDXL", ["flux1_*", "sdxl_*"], ["flux1_s_*", "sdxl_s_*"],
         "differential", "slow:flux1_s_flash:flux1_s_def:0.9", "rocm_flash.patch", None),
        ("j3_q21", "Qwen-Image-2.1", ["q21_*"], ["q21_s_*"], "differential", "slow:q21_s_flash:q21_s_def:0.9",
         "rocm_flash.patch", None),
        ("j4_wan", "Wan2.2-TI2V-5B short clip", ["wan_*"], ["wan_s_*"], "differential",
         "slow:wan_s_flash:wan_s_def:0.9", "rocm_flash.patch", None),
        ("j5_h3", "MiniMax-H3 (ComfyUI int8_convrot vs Studio GGUF)", ["h3_s_def", "h3_c_*"], ["h3_s_def"],
         "no_regression", None, "rocm_flash.patch", None),
    ],
    "v3": [
        # Base = main: the unified-memory refusal reproduced with 31 GiB left free (the refusing runner's state);
        # head = main + the Linux ROCm APU reserve fix.
        ("k1_unified_reserve", "Z-Image-Turbo unified-memory refusal (31 GiB free)", ["zimg_s_def"], ["zimg_s_def"],
         "differential", "broken:zimg_s_def", "unified_reserve.patch", ["--leave-free-gib", "31"]),
        # The AOTriton opt-in from the inference package: base = main falls back to SDPA math when nothing upstream
        # set the flag; head = main + the opt-in. klein_s_def (flag set by the harness, as the server does) is the
        # reference both states share.
        ("k0_aotriton_optin", "AOTriton opt-in from the inference package (FLUX.2-klein-4B)",
         ["klein_s_def", "klein_s_noflag"], ["klein_s_def", "klein_s_noflag"], "differential",
         "slow:klein_s_noflag:klein_s_def:1.5", "aotriton_optin.patch", None),
        # Qwen-Image-2.1 (evicted from run 37136344284) with the flash patch as head, as in v2.
        ("k3_q21", "Qwen-Image-2.1", ["q21_*"], ["q21_s_*"], "differential", "slow:q21_s_flash:q21_s_def:0.9",
         "rocm_flash.patch", None),
        # Studio's MiniMax-H3 (GGUF through its sd.cpp engine); ComfyUI's H3 cells ran in run 37136344284.
        ("k4_h3", "MiniMax-H3 Studio (GGUF)", ["h3_s_def"], ["h3_s_def"], "no_regression", None,
         "rocm_flash.patch", None),
        # Measurement, main only (the head re-runs one cell so the harness has two states): speed tier and video
        # decode levers, with ComfyUI on the same torch as the reference.
        ("k2_levers", "Speed tier + video decode levers (main)",
         ["zimg_s_defx", "zimg_s_eager", "zimg_c_rock", "klein_s_def", "klein_s_eager", "flux1_s_def", "flux1_s_eager",
          "wan_s_def", "wan_s_notile", "wan_s_nocudnn", "wan_s_nocudnn_notile", "wan_c_rock"],
         ["klein_s_def"], "no_regression", None, "unified_reserve.patch", None),
    ],
    "v4": [
        # Fix 3 again, now with a reference cell so the "base rendered at least one cell" gate can hold.
        ("m1_unified_reserve", "Z-Image-Turbo unified-memory refusal (31 GiB free)", ["zimg_s_def", "klein_s_def"],
         ["zimg_s_def", "klein_s_def"], "differential", "broken:zimg_s_def", "unified_reserve.patch",
         ["--leave-free-gib", "31"]),
        ("m2_wan_vae_bf16", "Wan2.2-TI2V-5B: bf16 VAE decode on ROCm", ["wan_s_vae", "wan_s_vaefp32"],
         ["wan_s_vae", "wan_s_vaefp32"], "differential", "slow:wan_s_vae:wan_s_vaefp32:0.8", "wan_vae_bf16.patch",
         None),
        ("m3_rocm_eager", "Z-Image-Turbo: eager tier for unset speed on ROCm",
         ["zimg_s_unset", "zimg_s_off", "zimg_c_rock"],
         ["zimg_s_unset", "zimg_s_off"], "differential", "slow:zimg_s_unset:zimg_s_off:0.95", "rocm_eager.patch",
         None),
    ],
    "v5": [
        # Studio's native MiniMax-H3 on a ROCm APU: auto resident (head) vs streamed. Head = fix 3 + the H3 policy.
        ("n1_h3_apu_resident", "MiniMax-H3 native: resident vs streamed on a unified-memory APU",
         ["h3_s_group", "h3_s_res"], ["h3_s_group", "h3_s_res"], "differential",
         "slow:h3_s_res:h3_s_group:0.8", "h3_apu_resident.patch", None),
    ],
    # v6 (outputs/comfy_amd_study/analysis.json cells), base = main 5971d280. Patches are each branch's net diff
    # rebased onto it; *_miopen combos stack two branches. Measurement jobs give the base the same patch so every
    # lever cell runs once (base) and the head re-runs one reference cell.
    "v6": [
        # Fix 4 as shipped now (weight cast by default): base fp32 decode, head bf16.
        ("r1_wan_vae", "Wan2.2-TI2V-5B: bf16 VAE weights decode on ROCm (fix 4)", ["wan_s_vae", "wan_s_vaefp32"],
         ["wan_s_vae", "wan_s_vaefp32"], "differential", "slow:wan_s_vae:wan_s_vaefp32:0.8", "wan_vae_bf16.patch",
         None),
        # V2 weights vs autocast, M1 MIOpen whole-process vs VAE-only, M2 untiled, M3/K1/D1 on ComfyUI's Wan.
        ("r2_wan_levers", "Wan2.2-TI2V-5B decode levers (V2, M1-M3, K1, D1)",
         ["wan_s_vae", "wan_s_vae_*_repeat", "wan_s_vaeac", "wan_s_vae_nocudnn", "wan_s_vae_miopenvae",
          "wan_s_vae_notile", "wan_s_vae_nocudnn_notile", "wan_c_rock", "wan_c_rock_*_repeat",
          "wan_c_miopen", "wan_c_nockhip", "wan_c_dyn"],
         ["wan_s_vae"], "no_regression", None, "wan_vae_miopen.patch", None,
         {"base_patch": "wan_vae_miopen.patch"}),
        # F1 flash vs default (A/A both), K1 / D1 on ComfyUI, G1 TunableOp, fused rope / AdaLN arm.
        ("r3_image_levers", "FLUX.2-klein / FLUX.1 / Qwen-Image levers (F1, K1, D1, G1, fused RoPE / AdaLN)",
         ["klein_s_def", "klein_s_def_*_repeat", "klein_s_flash", "klein_s_flash_*_repeat", "klein_s_tunable",
          "klein_s_nofused", "klein_s_ropeonly", "flux1_s_def", "flux1_s_def_*_repeat", "flux1_s_flash", "flux1_s_flash_*_repeat",
          "flux1_s_tunable", "flux1_s_nofused", "flux1_s_ropeonly", "klein_c_rock", "klein_c_rock_*_repeat", "klein_c_nockhip",
          "klein_c_dyn", "flux1_c_rock", "flux1_c_rock_*_repeat", "flux1_c_nockhip", "flux1_c_dyn", "q21_c_rock",
          "q21_c_nockhip"],
         ["klein_s_def"], "no_regression", None, "{image_patch}", None, {"base_patch": "{image_patch}"}),
        # M3 on SDXL, V1 2048^2 decodes; head = RDNA2 fp16 + VAE MIOpen scope, both inert on gfx1151 by design.
        ("r4_sdxl_2048_inert", "SDXL MIOpen (M3), 2048^2 decode (V1), RDNA2-fp16 + MIOpen-scope inert check",
         ["sdxl_s_def", "sdxl_c_rock", "sdxl_c_miopen", "sdxl_s_2048", "sdxl_c_rock_2048", "flux1_s_2048",
          "flux1_c_rock_2048", "klein_s_def"],
         ["sdxl_s_def", "klein_s_def"], "no_regression", None, "rdna2_miopen.patch", None),
        # T1: Studio on the pytorch.org rocm7.2 torch vs its own gfx1151 torch, same source. The installer change
        # (studio/amd-strix-pytorch-rocm72) is NOT the head here: install.sh runs from the head state, so it would
        # move the Studio venv itself; the T1 venvs carry exactly the torch that branch installs instead.
        ("r5_torch_pt72", "Studio on pytorch.org rocm7.2 torch (T1)",
         ["klein_s_def", "klein_s_def_*_repeat", "klein_s_pt72", "klein_s_pt72_*_repeat", "flux1_s_def",
          "flux1_s_pt72", "sdxl_s_def", "sdxl_s_pt72", "q21_s_def", "q21_s_pt72", "wan_s_def", "wan_s_pt72", "klein_s_pt72n", "flux1_s_pt72n", "wan_s_pt72n"],
         ["klein_s_def"], "no_regression", None, "{t1_patch}", None, {"pt72": True}),
        # E1: eager tier for an unset speed on ROCm (fix 5), klein + FLUX.1 (v4 m3 covers Z-Image).
        # Includes v4 m3 (Z-Image, evicted from the GPU group before it got a runner in run 37185571235).
        ("r6_rocm_eager", "Eager tier for unset speed on ROCm (Z-Image-Turbo, FLUX.2-klein, FLUX.1)",
         ["zimg_s_unset", "zimg_s_off", "zimg_c_rock", "klein_s_unset", "klein_s_off", "flux1_s_unset",
          "flux1_s_off"],
         ["zimg_s_unset", "zimg_s_off", "klein_s_unset", "klein_s_off", "flux1_s_unset", "flux1_s_off"],
         "differential", "slow:zimg_s_unset:zimg_s_off:0.95", "rocm_eager.patch", None),
    ],
    # v7, base = main 79116720 (has #12667 flash, #12668 AOTriton opt-in, #12675 unified reserve, #12676 RDNA2 fp16).
    # Every job first waits for a clean GPU pool (preflight_free.py): v6 r2 ran on a runner where another tenant
    # held ~31 GiB before the first cell, and v5 n1 was evicted from the GPU group before it got a runner.
    "v7": [
        ("s1_wan_levers", "Wan2.2-TI2V-5B decode levers (V2, M1-M3, K1, D1), rerun of v6 r2",
         ["wan_s_vae", "wan_s_vae_*_repeat", "wan_s_vaeac", "wan_s_vae_nocudnn", "wan_s_vae_miopenvae",
          "wan_s_vae_notile", "wan_s_vae_nocudnn_notile", "wan_c_rock", "wan_c_rock_*_repeat",
          "wan_c_miopen", "wan_c_nockhip", "wan_c_dyn"],
         ["wan_s_vae"], "no_regression", None, "wan_vae_miopen.patch", None,
         {"base_patch": "wan_vae_miopen.patch", "preflight_gib": 54}),
        # v5 n1 again: native H3 resident (head) vs streamed (kill switch) on the APU. Fix 3 is in main now, so the
        # head patch is the H3 policy alone.
        ("s2_h3_apu_resident", "MiniMax-H3 native: resident vs streamed on a unified-memory APU (rerun of v5 n1)",
         ["h3_s_group", "h3_s_res"], ["h3_s_group", "h3_s_res"], "differential",
         "slow:h3_s_res:h3_s_group:0.8", "h3_apu_resident.patch", None, {"preflight_gib": 54}),
    ],
}


HEADER = f"""name: "AMD CI: Studio vs ComfyUI diffusion on gfx1151 (Linux)"

# Generated by diffusion_bench/amd/gen_studio_vs_comfy_workflow.py from the amd_ci scaffold (templates/workflow.yml):
# same preamble, gate, Studio install, differential and artifact plumbing, one GPU job per model group.
# Measurement only: no secret is referenced, every model is a public download (token=False).

on:
  push:
    branches: [{BRANCH}]
  workflow_dispatch:

permissions:
  contents: read

env:
  UPSTREAM: "https://github.com/unslothai/unsloth"
  BASE_SHA: "{{BASE_SHA}}"
  COMFY_SHA: "{COMFY_SHA}"
  FA_SHA: "{FA_SHA}"

jobs:
"""

JOB = """
  {jid}:
    name: "{title} (gfx1151, exclusive GPU, {job_set})"
{needs}    runs-on: [self-hosted, Linux, strix-halo, devlab-dispatch]
    timeout-minutes: 600
    # The shared GPU group (see templates/workflow.yml): one GPU measurement at a time across all branches.
    concurrency:
      group: amd-ci-gfx1151-gpu
      cancel-in-progress: false
    steps:
      - uses: actions/checkout@v4
        with: {{ persist-credentials: false }}

      - name: Preamble
        run: |
          source amd_ci/lib/preamble.sh "svc-{jid}"

      - name: Hardware gate
        run: |
          set -euo pipefail
          python3 amd_ci/lib/gate.py --no-torch --require rocm gpu \\
            --expect-arch gfx1151 --min-gpus 1 \\
            --out "$AMD_CI_WORK/out/host.json" || \\
          echo "note: gate ran before the venv exists; re-gated after install below"

      - name: Disk, memory, cores
        run: |
          set +e
          df -h "$RUNNER_TEMP"
          free -g
          nproc
          exit 0

      - name: Resolve states (main, and main + the local ROCm flash patch)
        run: |
          set -euo pipefail
          git clone -q --filter=blob:none "$UPSTREAM" "$AMD_CI_WORK/repo"
          git -C "$AMD_CI_WORK/repo" worktree add -q --detach "$AMD_CI_WORK/states/base" "$BASE_SHA"
          git -C "$AMD_CI_WORK/repo" worktree add -q --detach "$AMD_CI_WORK/states/head" "$BASE_SHA"
          git -C "$AMD_CI_WORK/states/head" apply --index "$GITHUB_WORKSPACE/amd_ci/patches/{patch}"
          git -C "$AMD_CI_WORK/states/head" diff --cached --stat
{base_patch_cmd}          python3 "$GITHUB_WORKSPACE/diffusion_bench/amd/write_states.py" --work "$AMD_CI_WORK" \\
            --base-sha "$BASE_SHA" --patch {patch} --out "$AMD_CI_WORK/out/states.json"

      - name: Environment and real gate
        timeout-minutes: 60
        run: |
          set -euo pipefail
          cd "$AMD_CI_WORK/states/head"
          export UNSLOTH_STUDIO_HOME="$AMD_CI_WORK/studio"
          echo "UNSLOTH_STUDIO_HOME=$UNSLOTH_STUDIO_HOME" >> "$GITHUB_ENV"
          # The runners' uv index is a PyPI mirror that lags; give uv pypi.org as an extra index (scaffold default).
          export UV_EXTRA_INDEX_URL="https://pypi.org/simple"
          set +e
          bash install.sh --local 2>&1 | tail -15
          rc=${{PIPESTATUS[0]}}
          set -e
          [ "$rc" -eq 0 ] || exit "$rc"
          V=""
          for c in "$UNSLOTH_STUDIO_HOME"/unsloth_studio "$UNSLOTH_STUDIO_HOME"/.venv_t5_*; do
            [ -x "$c/bin/python" ] && {{ V="$c"; break; }}
          done
          [ -n "$V" ] || {{ echo "FATAL: no Studio venv"; exit 1; }}
          echo "AMD_CI_PY=$V/bin/python" >> "$GITHUB_ENV"
          "$V/bin/python" "$GITHUB_WORKSPACE/amd_ci/lib/gate.py" --require rocm gpu \\
            --expect-arch gfx1151 --min-gpus 1 --out "$AMD_CI_WORK/out/host.json"

      - name: flash-attention (DAO-AILab, Triton AMD backend) into the Studio venv
        timeout-minutes: 30
        run: |
          set +e
          git clone -q https://github.com/Dao-AILab/flash-attention "$AMD_CI_WORK/src/flash-attention"
          git -C "$AMD_CI_WORK/src/flash-attention" checkout -q "$FA_SHA"
          "$AMD_CI_PY" -m pip --version > /dev/null 2>&1 || "$AMD_CI_PY" -m ensurepip -q
          "$AMD_CI_PY" -c "import torch; print('studio torch before', torch.__version__)"
          "$AMD_CI_PY" -m pip install -q setuptools wheel packaging psutil ninja einops > "$AMD_CI_WORK/out/fa_studio_deps.log" 2>&1
          cd "$AMD_CI_WORK/src/flash-attention"
          FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE "$AMD_CI_PY" -m pip install --no-build-isolation --no-deps . > "$AMD_CI_WORK/out/fa_studio_build.log" 2>&1
          rc=$?
          echo "flash-attn into Studio venv: rc=$rc"
          tail -3 "$AMD_CI_WORK/out/fa_studio_build.log"
          "$AMD_CI_PY" -c "import torch; print('studio torch after', torch.__version__)"
          # Three views of the same kernels: SDPA as torch ships it, with the AOTriton opt-in Studio's server and
          # ComfyUI both set, and with CK preferred for SDPA.
          FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE "$AMD_CI_PY" "$GITHUB_WORKSPACE/diffusion_bench/amd/fa_smoke.py" \\
            --out "$AMD_CI_WORK/out/fa_smoke_studio_noflag.json"
          FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 "$AMD_CI_PY" \\
            "$GITHUB_WORKSPACE/diffusion_bench/amd/fa_smoke.py" --out "$AMD_CI_WORK/out/fa_smoke_studio_aotriton.json"
          FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 TORCH_ROCM_FA_PREFER_CK=1 \\
            "$AMD_CI_PY" "$GITHUB_WORKSPACE/diffusion_bench/amd/fa_smoke.py" --out "$AMD_CI_WORK/out/fa_smoke_studio_ck.json"
          exit 0

      - name: ComfyUI (README ROCm install, own venv)
        timeout-minutes: 45
        run: |
          set +e
          git clone -q https://github.com/comfyanonymous/ComfyUI "$AMD_CI_WORK/ComfyUI"
          git -C "$AMD_CI_WORK/ComfyUI" checkout -q "$COMFY_SHA"
          BASEPY=$("$AMD_CI_PY" -c "import sys; print(sys._base_executable)")
          "$BASEPY" -m venv "$AMD_CI_WORK/comfy_venv_default"
          P="$AMD_CI_WORK/comfy_venv_default/bin/python"
          "$P" -m pip install -q --upgrade pip > /dev/null 2>&1
          # README, "AMD GPUs (Linux)": the stable ROCm wheel, then the requirements.
          "$P" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2 > "$AMD_CI_WORK/out/comfy_default_torch.log" 2>&1
          echo "README torch install rc=$?"
          "$P" -m pip install -r "$AMD_CI_WORK/ComfyUI/requirements.txt" > "$AMD_CI_WORK/out/comfy_default_reqs.log" 2>&1
          echo "requirements rc=$?"
          "$P" -c "import torch; print('comfy default torch', torch.__version__, 'hip', torch.version.hip, 'gpu', torch.cuda.is_available())"
          exit 0

      - name: ComfyUI venv on the gfx1151 torch Studio uses, + flash-attention
        timeout-minutes: 45
        run: |
          set +e
          TV=$("$AMD_CI_PY" -c "import torch; print(torch.__version__)")
          VV=$("$AMD_CI_PY" -c "import torchvision; print(torchvision.__version__)")
          BASEPY=$("$AMD_CI_PY" -c "import sys; print(sys._base_executable)")
          "$BASEPY" -m venv "$AMD_CI_WORK/comfy_venv_rock"
          P="$AMD_CI_WORK/comfy_venv_rock/bin/python"
          "$P" -m pip install -q --upgrade pip > /dev/null 2>&1
          "$P" -m pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "torch==$TV" "torchvision==$VV" > "$AMD_CI_WORK/out/comfy_rock_torch.log" 2>&1
          echo "gfx1151 torch $TV install rc=$?"
          "$P" -m pip install -r "$AMD_CI_WORK/ComfyUI/requirements.txt" > "$AMD_CI_WORK/out/comfy_rock_reqs.log" 2>&1
          echo "requirements rc=$?"
          "$P" -m pip install -q setuptools wheel packaging psutil ninja einops > /dev/null 2>&1
          cd "$AMD_CI_WORK/src/flash-attention"
          FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE "$P" -m pip install --no-build-isolation --no-deps . > "$AMD_CI_WORK/out/fa_comfy_build.log" 2>&1
          echo "flash-attn into ComfyUI venv: rc=$?"
          "$P" -c "import torch; print('comfy rock torch', torch.__version__, 'expected $TV')"
          exit 0

{extra_steps}      - name: "Differential ({title})"
        run: |
          set +e
          export TORCH_HOME="$AMD_CI_WORK/torch_home"
          export HF_HOME="$AMD_CI_WORK/hf_home"
          "$AMD_CI_PY" amd_ci/lib/differential.py \\
            --states "$AMD_CI_WORK/out/states.json" \\
            --probe diffusion_bench/amd/diffusion_probe.py \\
            --criteria diffusion_bench/amd/criteria_{criteria}.py \\
            --out-dir "$AMD_CI_WORK/out/gpu" {fixture}\\
            --python "$AMD_CI_PY" -- {probe_args}
          cat "$AMD_CI_WORK/out/gpu/VERDICT.md" >> "$GITHUB_STEP_SUMMARY"
          python3 "$GITHUB_WORKSPACE/amd_ci/lib/announce.py" "$AMD_CI_WORK/out/gpu"

      - name: Stop anything this job started
        if: always()
        run: |
          set +e
          for d in /proc/[0-9]*; do
            pid=${{d#/proc/}}
            [ "$pid" = "$$" ] && continue
            cwd=$(readlink "$d/cwd" 2>/dev/null)
            case "$cwd" in
              "$AMD_CI_WORK"*) echo "stopping pid $pid (cwd $cwd)"; kill "$pid" 2>/dev/null ;;
            esac
          done
          exit 0

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: svc-{jid}
          path: |
            ${{{{ env.AMD_CI_WORK }}}}/out
            ${{{{ env.AMD_CI_WORK }}}}/dbench/states/*/runs/**/record.json
            ${{{{ env.AMD_CI_WORK }}}}/dbench/states/*/runs/**/*.png
            ${{{{ env.AMD_CI_WORK }}}}/dbench/states/*/runs/**/*.log
            ${{{{ env.AMD_CI_WORK }}}}/dbench/states/*/logs
          retention-days: 14
          if-no-files-found: warn
"""


PREFLIGHT_STEP = """      - name: Clean GPU pool preflight (fail loudly on a co-tenant)
        timeout-minutes: 35
        run: |
          set -euo pipefail
          "$AMD_CI_PY" "$GITHUB_WORKSPACE/diffusion_bench/amd/preflight_free.py" --min-free-gib {gib} \\
            --timeout-s 1800 --out "$AMD_CI_WORK/out/preflight.json"

"""


# T1: a second Studio venv on the torch ComfyUI's README installs (download.pytorch.org rocm7.2), every other package
# as in the Studio venv. Cells point at it with "venv"; the Studio source is still each state's checkout.
PT72_STEP = """      - name: Studio venvs on the pytorch.org rocm7.2 torch (T1 2.11.0, T1b 2.14.1)
        timeout-minutes: 60
        run: |
          set +e
          BASEPY=$("$AMD_CI_PY" -c "import sys; print(sys._base_executable)")
          "$AMD_CI_PY" -m pip --version > /dev/null 2>&1 || "$AMD_CI_PY" -m ensurepip -q
          "$AMD_CI_PY" -m pip freeze > "$AMD_CI_WORK/out/pt72_studio_freeze.txt" 2>/dev/null
          grep -viE '^(-e )?(torch|torchvision|torchaudio|triton|triton[-_]rocm|pytorch[-_]triton[-_a-z]*|rocm[-_a-z0-9]*|flash[-_]attn)( |=|@|$)' \\
            "$AMD_CI_WORK/out/pt72_studio_freeze.txt" > "$AMD_CI_WORK/out/pt72_reqs.txt"
          for spec in "studio_pt72|torch==2.11.0+rocm7.2 torchvision==0.26.0+rocm7.2 torchaudio==2.11.0+rocm7.2" \\
                      "studio_pt72n|torch==2.14.1+rocm7.2 torchvision==0.29.1+rocm7.2"; do
            name=${spec%%|*}; pkgs=${spec#*|}
            "$BASEPY" -m venv "$AMD_CI_WORK/$name"
            P="$AMD_CI_WORK/$name/bin/python"
            "$P" -m pip install -q --upgrade pip > /dev/null 2>&1
            "$P" -m pip install $pkgs --index-url https://download.pytorch.org/whl/rocm7.2 > "$AMD_CI_WORK/out/${name}_torch.log" 2>&1
            echo "$name torch ($pkgs) rc=$?"
            "$P" -m pip install --no-deps -r "$AMD_CI_WORK/out/pt72_reqs.txt" > "$AMD_CI_WORK/out/${name}_reqs.log" 2>&1
            rc=$?
            echo "$name Studio packages (bulk) rc=$rc"
            if [ "$rc" -ne 0 ]; then
              while IFS= read -r line; do
                "$P" -m pip install --no-deps "$line" >> "$AMD_CI_WORK/out/${name}_reqs.log" 2>&1 || echo "$name skipped: $line"
              done < "$AMD_CI_WORK/out/pt72_reqs.txt"
            fi
            (cd "$AMD_CI_WORK/src/flash-attention" && FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE "$P" -m pip install --no-build-isolation --no-deps . > "$AMD_CI_WORK/out/fa_${name}_build.log" 2>&1)
            echo "flash-attn into $name: rc=$?"
            "$P" -c "import torch, diffusers, transformers; print('$name torch', torch.__version__, 'hip', torch.version.hip, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_properties(0).gcnArchName, 'archs', torch._C._cuda_getArchFlags(), 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)"
            "$P" -m pip check 2>&1 | tail -5
            TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 "$P" "$GITHUB_WORKSPACE/diffusion_bench/amd/fa_smoke.py" --out "$AMD_CI_WORK/out/fa_smoke_${name}_aotriton.json"
          done
          exit 0

"""

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required = True, type = Path)
    ap.add_argument("--budget-min", type = float, default = 420.0)
    ap.add_argument("--cell-timeout", type = float, default = 5400.0)
    ap.add_argument("--set", dest = "job_set", choices = sorted(JOB_SETS), default = "v2")
    ap.add_argument("--image-patch", default = "flash_fused.patch", help = "v6 r3 patch (flash + fused rope)")
    ap.add_argument("--t1-patch", default = "rdna2_miopen.patch", help = "v6 r5 head patch (installer change)")
    args = ap.parse_args()
    branch = f"{BRANCH}-{args.job_set}"
    text = HEADER.replace("{BASE_SHA}", BASE_SHAS[args.job_set]).replace(f"branches: [{BRANCH}]",
                                                                         f"branches: [{branch}]")
    prev = None
    jobs = sorted(JOB_SETS[args.job_set], key = lambda j: j[0])  # k0, k1, k2... / j1, j2...: chain order
    for job in jobs:
        jid, title, only, head_only, criteria, defect, patch, fixture = job[:8]
        extras = dict(job[8]) if len(job) > 8 else {}
        subst = {"{image_patch}": args.image_patch, "{t1_patch}": args.t1_patch}
        patch = subst.get(patch, patch)
        if extras.get("base_patch"):
            extras["base_patch"] = subst.get(extras["base_patch"], extras["base_patch"])
        pa = ["--spec", "diffusion_bench/specs/amd_studio_vs_comfy.json", "--budget-min", str(args.budget_min),
              "--cell-timeout", str(args.cell_timeout)]
        for o in only:
            pa += ["--only", o]
        for h in head_only:
            pa += ["--state-only", f"head={h}"]
        if defect:
            pa += ["--defect", defect]
        needs = f"    needs: [{prev}]\n    if: ${{{{ always() }}}}\n" if prev else ""
        fx = ""
        if fixture:
            fx = ("\\\n            --fixture diffusion_bench/amd/hold_to_free.py "
                  + " ".join(f"--fixture-arg={shlex.quote(a)}" for a in fixture) + " --fixture-timeout 900 ")
        bp = extras.get("base_patch")
        base_patch_cmd = ""
        if bp:
            # Measurement jobs: both states carry the levers under test; the base gets them as well (staged, so
            # write_states still sees the base commit and the head's own staged diff).
            base_patch_cmd = (f'          git -C "$AMD_CI_WORK/states/base" apply --index '
                              f'"$GITHUB_WORKSPACE/amd_ci/patches/{bp}"\n'
                              f'          git -C "$AMD_CI_WORK/states/base" diff --cached --stat\n')
        extra_steps = PT72_STEP if extras.get("pt72") else ""
        if extras.get("preflight_gib"):
            extra_steps += PREFLIGHT_STEP.replace("{gib}", str(extras["preflight_gib"]))
        text += JOB.format(jid = jid, title = title, needs = needs, criteria = criteria, patch = patch, fixture = fx,
                           probe_args = " ".join(shlex.quote(a) for a in pa), base_patch_cmd = base_patch_cmd,
                           extra_steps = extra_steps, job_set = args.job_set)
        prev = jid
    wf = args.out / ".github" / "workflows" / f"{branch}.yml"
    wf.parent.mkdir(parents = True, exist_ok = True)
    for old in wf.parent.glob("*.yml"):
        if old != wf:
            old.unlink()
    wf.write_text(text, encoding = "utf-8")
    print(f"wrote {wf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
