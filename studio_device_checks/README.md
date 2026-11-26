# Studio device checks

One command that checks open Unsloth Studio PRs on a real machine, against the Studio that is already
installed there. Nothing is reinstalled and nothing is added to the Studio venv.

How it works:

- It finds the installed Studio's own Python, in the same order as the installers:
  `UNSLOTH_STUDIO_HOME`, then `STUDIO_HOME`, then `UNSLOTH_HOME\studio`, then `~/.unsloth/studio`
  (`%USERPROFILE%\.unsloth\studio` on Windows). Inside it, `unsloth_studio\Scripts\python.exe` (Windows) or
  `unsloth_studio/bin/python` (Linux). Pass `-StudioHome` / `--studio-home` to point elsewhere.
- For each PR it downloads two GitHub zips without a token: the PR head and the PR's merge base on main. Only
  `studio/backend` and `tests/_shared` are unpacked, into a work folder.
- Every check runs in a fresh process under the Studio Python, with `PYTHONPATH` set to one tree's
  `studio/backend`: the installed release, the merge base, or the PR head. Each arm records where `core` was
  imported from, so an arm that picked up the wrong tree is an ERROR, not a result.
- Before running a PR, it imports that PR's modules under the installed venv. If the PR needs a package or version
  the venv does not have, the PR's checks are NOT_RUN with the missing name, and nothing is installed.
- If pytest is not in the Studio venv, pinned pytest wheels (sha256 checked) are unpacked into the work cache and
  put on `PYTHONPATH`. The venv is not touched.
- Each arm gets its own scratch Studio home, prequant cache and compile caches under the results folder, so
  nothing the PR code writes reaches the real Studio install. Model weights use Studio's own Hugging Face cache, so
  models you already have are not downloaded again.
- No token is read or required. `HF_TOKEN` is used only if it is already set and a gated repo needs it. Nothing is
  uploaded.

## Get the scripts

```
git clone --depth 1 https://github.com/unslothai/scripts unslothai_scripts
```

## One-line commands

RTX 5090, Windows x64 (PowerShell):

```
powershell -ExecutionPolicy Bypass -File unslothai_scripts\studio_device_checks\run.ps1 -Mode quick
powershell -ExecutionPolicy Bypass -File unslothai_scripts\studio_device_checks\run.ps1 -Mode full
```

RTX Spark, Windows on Arm (PowerShell, same commands):

```
powershell -ExecutionPolicy Bypass -File unslothai_scripts\studio_device_checks\run.ps1 -Mode quick
powershell -ExecutionPolicy Bypass -File unslothai_scripts\studio_device_checks\run.ps1 -Mode full
```

DGX Spark, Linux:

```
bash unslothai_scripts/studio_device_checks/run.sh --mode quick
bash unslothai_scripts/studio_device_checks/run.sh --mode full
```

Useful options (PowerShell takes the same `--` options after the named ones):

| Option | Effect |
|---|---|
| `-StudioHome D:\studio` / `--studio-home /path` | the folder that holds `unsloth_studio` |
| `-DryRun` / `--dry-run` | find Studio, resolve the PR zips, print the plan; downloads nothing |
| `-Yes` / `--yes` | answer yes to the model download questions (full mode) |
| `--prs 19665,19588` | only these PRs (default: all listed below) |
| `--repeats 3` | timing repeats per arm in full mode (default 3) |
| `--fp8-modes eager` | FP8 renders eager only (default `eager,comp`) |
| `--skip-video-download` | skip the live video download sample in full mode |
| `--work-dir DIR` | cache and results root (default `~/unsloth_device_checks`) |

`quick` takes a few minutes plus the zip downloads (about 70 MB per zip, cached for later runs). `full` adds model
renders. It shows the download size first, says how much is already in your Hugging Face cache, and asks before
downloading. Without a terminal it does not download unless `--yes` is given.

## Results

Each run writes one folder, printed at the start and the end:

```
~/unsloth_device_checks/run_<mode>_<time>/
  summary.md      the table to paste back
  summary.json    everything, machine readable
  logs/           one log per arm
  arms/           one JSON per arm
  images/         renders (full mode)
```

`summary.md` and `summary.json` hold no tokens and no environment values other than `UNSLOTH_*` flags.

Statuses:

| Status | Meaning |
|---|---|
| PASS | the PR behaves as claimed on this machine |
| FAIL | it does not; details are in `summary.json` and the arm logs |
| INFO | an observation with no pass bar (for example which fp8 kernels this torch build has) |
| NOT_RUN | this machine cannot answer it, with the reason (no NVIDIA GPU, wrong architecture, missing package, download declined) |
| ERROR | the harness itself failed |

## What is checked

| Check | PR | Mode | Machines | PASS means |
|---|---|---|---|---|
| `sanity.compile` | #19569 (merged) | quick | any NVIDIA | torch and Triton import, and a tiny `torch.compile` on the GPU matches eager. Notes a torch 2.15 + non-3.9 Triton pairing |
| `fp8.kernels` | #19665 | quick | sm89+ | INFO: whether this torch build runs the row-wise, the tensorwise and the fp32-output tensorwise `torch._scaled_mm` |
| `fp8.smoke_probe` | #19665 | quick | sm89+ | where the row-wise kernel is missing (Windows wheels): the PR head's smoke probe passes and `gemm_path` is `unit_scale_fallback`; the installed release and merge base are expected to fail. Where it exists (Linux): head still passes on `torchao_rowwise` |
| `fp8.tests` | #19665 | quick | any | the PR's CPU tests pass on head; the count failing on base and passing on head is shown |
| `fp8.render` | #19665 | full | sm89+ | FLUX.2 klein 4B renders with FP8 on head, eager and compiled, with the `fp8_gemm` status field set; LPIPS and PSNR against BF16 at the same seeds; s/step medians over the repeats and the repeat spread (A/A noise). The merge base's FP8 render is the before arm |
| `nvfp4.tests` | #19664 | quick | any | the PR's prequant cache and LoRA gate tests pass on head |
| `nvfp4.cache_timing` | #19664 | full | sm100+ (DGX Spark, RTX 5090) | first and later NVFP4 klein 4B loads with `UNSLOTH_NVFP4_DIFFUSION=1`, base and head interleaved: later head loads read the prequant cache and are faster than later base loads (NVIDIA reported 59.5 s for a cached-model load) |
| `nvfp4.lora_s0` | #19664 | full | sm100+ | an NVFP4 build with a baked LoRA at strength 0 renders bit-identical to an NVFP4 base render |
| `w6a8.check_tile` | #19588 | quick | sm120 (RTX 5090) | every fused W6A8 phase tile passes `check_tile` on head and packed GEMM calls take the fused path (`fused_call_count` rises) with bit-identical output. The Triton version is reported: the merge base is expected to fail on Triton 3.9 only |
| `w6a8.tests` | #19588 | quick | Triton importable | the PR's tests (Triton interpreter) pass on head |
| `video_dl.tests` | #19621 | quick | any | the PR's download-progress tests pass on head (the symlink-less cache case matters on Windows) |
| `video_dl.live` | #19621 | full | NVIDIA | a real hosted video download for about 90 s per arm into a scratch cache that is deleted afterwards: the head's progress counter moves with the bytes on disk |

Full mode downloads, from `unsloth/*` repos (and the small `adirik/pixel-art-lora-flux.2-klein-4B` LoRA for the
NVFP4 LoRA check): the FLUX.2 klein 4B pipeline (about 16 GB) and, for FP8, the hosted FP8 transformer and text
encoder (about 9 GB). The prompt shows the exact sizes and what is already cached.

## Per machine, what to expect

- RTX 5090, Windows x64: FP8 (the row-wise kernel is expected to be missing), W6A8 on sm120, NVFP4 on sm120, video
  download counter, torch / Triton sanity.
- RTX Spark, Windows on Arm (GB10, sm121): FP8 and NVFP4. W6A8 is NOT_RUN (sm120 only). Wheels that do not exist for
  win_arm64 show up as NOT_RUN with the missing package, not as a failure.
- DGX Spark, Linux (GB10, sm121): NVFP4 cache timing and LoRA strength 0, FP8 as a no-regression check (the row-wise
  kernel exists on Linux wheels).

## Files

| File | Role |
|---|---|
| `run.ps1`, `run.sh` | entry points: find the Studio Python and start the driver with it |
| `devchecks.py` | the driver: discovery, zip fetch, dependency gate, arms, judging, summary (standard library only) |
| `probes/*.py` | one process per arm, run under the Studio Python with one backend tree on `PYTHONPATH` |

To add a PR: add it to `PRS` in `devchecks.py` (title, test files, modules to import), and add a check function
next to the others if it needs more than its unit tests.
