# AMD CI

A reusable harness for validating changes on the AMD CI runners (Strix Halo
gfx1151). The pool has a Linux half (ROCm) and a Windows half (Windows 11, no
Docker); both are first-class targets here.

It exists because of a measured observation: across 19 CI runs, **zero** failed
because of the platform. All seven failures were harness authoring: an errexit
that swallowed an exit code, a `pkill -f` that matched its own shell, a detector
that read the wrong field, a glob that failed under `pipefail`. The runner is
reliable; writing correct one-off workflows is not. So this toolkit standardises
the plumbing and encodes each of those mistakes as a lint rule or a default.

## The one rule

> In differential mode, if the base state does not exhibit the defect, the
> verdict is **VOID**. Not a pass.

A green head leg with no demonstrated base failure shows only that the harness
ran. This is enforced in `lib/differential.py` and is deliberately not a
per-probe option: a reusable harness whose pass criteria are negotiable becomes
a green-tick generator, and then none of its results mean anything.

## Shape

A **probe** observes and never judges. A **criteria** module judges and never
observes. Keeping them apart is what stops a probe being written so it always
passes.

```
lib/preamble.sh       LINUX isolation: sudo shim, roots under $RUNNER_TEMP, set +e
lib/preamble.ps1      the WINDOWS counterpart, for `shell: powershell`
lib/gate.py           hardware gate; FAILS rather than skips
lib/capability.py     what this host can and cannot answer
lib/states.py         base / head / merge (or merge commit + parent) as worktrees
lib/differential.py   run probe per state, apply gates, decide, render
lib/lint_workflow.py  catch the bugs that cost real runs, before spending one
lib/announce.py       annotate the verdict; fail the job on a NON-result
probes/               pytest_probe.py, plus your own
criteria/             pytest_no_regression.py, plus your own
templates/workflow.yml, templates/workflow_windows.yml, scaffold.py, selftest.py
```

## Use

Most PRs need nothing written at all:

```bash
python amd_ci/scaffold.py --pr 9487 --out ci_pr9487 \
    --tests tests/test_sd_cpp_install.py
```

For a PR already merged, compare the merge commit to its parent:

```bash
python amd_ci/scaffold.py --pr 9315 --merged --out ci_pr9315 --no-gpu --tests tests/
```

Scaffolding lints the generated workflow and refuses to print the push command
while error-level findings remain.

For anything the generic pytest probe cannot express, write a probe that emits
JSON and a criteria module with `MODE`, `gates()`, `table()` and either
`base_shows_defect`/`head_is_fixed` (differential) or `head_is_worse`
(regression). See `criteria/pytest_no_regression.py`.

### Fixtures

When the condition being measured has to hold across *every* state's probe, use
a fixture. The motivating case is "another process is holding VRAM": that must
be equally true for the base and head readings or they are not comparable, and
a per-state probe cannot arrange it.

```bash
python amd_ci/lib/differential.py \
  --states .../states.json \
  --probe amd_ci/probes/gpu_summary_probe.py \
  --criteria amd_ci/criteria/gpu_summary_sees_others.py \
  --fixture amd_ci/probes/vram_holder_fixture.py \
  --fixture-arg=--gib --fixture-arg=4 \
  --out-dir out/
```

A fixture prints one JSON line containing `"status": "READY"` when it is up;
the runner polls for that rather than sleeping. It is stopped by PID, never by
`pkill -f`. If it never becomes ready the verdict is **INCONCLUSIVE**, because a
differential measured against a condition that was never established is no
result rather than a failing one.

Its READY payload lands in the observations as `_fixture`, which criteria read
for non-vacuity gates (for example "the holder really held >= 2 GiB").

## Lint rules, and the run each one cost

| code | catches |
|---|---|
| E000 | `bash -n` syntax error in a `run:` block |
| E001 | `rc=$?` with no `set +e`; GitHub's shell is `bash -e`, so the step aborts first |
| E002 | `pkill -f` whose pattern matches the calling shell, so the step kills itself |
| E003 | heredoc closed by an indented terminator without `<<-` |
| E004 | parsing a program's stdout as JSON; import banners corrupt it |
| E005 | glob or pipeline under `pipefail`; `2>/dev/null` hides the message, not the status |
| E006 | `nohup ... &` backgrounding the work out of the step's shell |
| E007 | a relative `amd_ci/...` path in a step that has `cd`'d away from the checkout |
| W101 | a GPU-measuring job with no concurrency group |
| W102 | `exit 0` when hardware is absent, which misreports a miss as nothing to see |
| E100 | a Windows job whose shell is not `powershell`, including an unset one |
| E101 | `docker` in a Windows job; it is not installed on any of them |
| E102 | bash, `sh` or a `.sh` script in a Windows job |
| W103 | branching on `RUNNER_OS` in a Windows job; it has been observed blank |
| E104 | a bash-style `$RUNNER_TEMP` in a PowerShell block, which expands to `""` |
| E106 | a `strix-halo` selector with no OS label, now that both OSes answer |
| W104 | `Out-File` into `GITHUB_ENV` from PowerShell 5.1, which writes a BOM |

Verified against git history: E001 fires on the two revisions whose runs it cost,
E005 on the revision whose run it cost. The E10x family is verified against the
runners themselves; see "Windows" below.

The bash rules are skipped on a Windows job rather than adapted. `rc=$?`,
`pipefail` and `bash -n` describe a shell the step will never run under, and a
linter that reports an unfixable bash syntax error for a PowerShell block gets
ignored wholesale. W101 and W102 are about the job, not the shell, so they still
apply on both.

## What the hardware can and cannot answer

`lib/capability.py` holds this, and every report auto-appends a "Not tested here"
section from it. Do not hand-write that section and do not omit it: on a Linux
runner NVIDIA UUID masks, MIG, multi-GPU, XPU, MLX and Windows are all
unreachable, and a report that stays silent about them overstates its reach.

`windows` is a gap for the JOB, not for the pool. The Windows runners are
reachable (see below), so the gap now says which selector reaches them rather
than claiming the pool is Linux. `windows_docker` is a separate capability
precisely so this stays honest in both directions: one combined gap would either
report a Windows-only change as unreachable when it is not, or report a
containerised one as reachable when it is not. Both are computed from an observed
profile, never asserted.

**The gaps are computed, but `NEEDS` is authored.** `untested_section` renders
`NEEDS` minus what the host has, so an under-declared `NEEDS` yields no gaps.
A Windows-only PR whose criteria declared only `["rocm"]` produced a report with
no bounds at all. So the section is now rendered even when nothing is missing,
saying explicitly that it is a claim about the declaration, and `differential.py`
emits a `::warning::` for that case. Writing a criteria module still means
listing every capability the change touches, not only the ones the host has.

This is the part that does **not** generalise. The harness runs anywhere; the
*claims* of any one run are bounded by the machine that ran it, which is one
integrated GPU, on Linux under RADV, or one Windows box with no Docker.

### Multi-GPU wiring, on a one-GPU box

`multi_gpu` used to be a permanent gap. It is not, for one class of question.
`scaffold.py --spoof-devices N` preloads `lib/device_multiplier.py`, an LD_PRELOAD
shim over the HIP runtime reporting N extra devices and folding the extra ordinals
onto the real one. Measured here: `torch.cuda.device_count()` is 2, `cuda:1` accepts
real tensors, and a real matmul runs on it. Selection logic, arch gates and code
assuming a single device all become reachable.

The phantom device is the real GPU wearing another number, so sharding, throughput,
collectives and mixed architectures get a **confident wrong answer** rather than an
error. Activation and disclosure are therefore one action: the flag sets
`AMD_CI_SPOOFED_DEVICES`, `capability.py` subtracts the fabricated devices, and
`multi_gpu` stays UNMET with an explicit note. `selftest.py` mutation-checks that.

It only fools HIP: amd-smi, sysfs and KFD still report one GPU, so code enumerating
through amd-smi (including Studio's `utils/hardware/hardware.py`) needs its own stub.
And symbol coverage is only what torch drives; an unwrapped ordinal-taking HIP call
dies with `hipErrorInvalidDevice` 101, as the first version did inside rocBLAS.

## Windows

The pool's Windows half is reachable and does not queue:

```bash
python amd_ci/scaffold.py --pr 9487 --windows --out ci_pr9487w \
    --tests tests/test_sd_cpp_install.py
```

That emits `runs-on: [self-hosted, Windows, strix-halo, devlab-dispatch]` with
`shell: powershell`, `lib/preamble.ps1`, and a `linux-control` job that differs
from it only in the OS label.

Measured on five distinct machines (`X2-LA01-S05-D01`, `X2-LA01-S04-D02`,
`DESKTOP-O81NVEC`, `DESKTOP-52OM6HM`, `DESKTOP-NILQUD2`), all alike:

- Windows 11 Pro 10.0.26200 build 26200, 64-bit; GMKtec NucBox EVO-X2, 32 logical
  processors, 68,340,748,288 bytes RAM, ~1.75 TB free on C:.
- `AMD Radeon(TM) 8060S Graphics` visible to Windows, driver `32.0.22018.5`,
  device `0x1586`, status OK.
- Present: `wsl.exe`, `git.exe`, `C:\Program Files\Python312\python.exe`,
  `amd-smi.exe` in `system32`.

The shell traps, each of which fails before the step's first line runs:

- `shell: pwsh` gives `pwsh: command not found`. **There is no PowerShell 7**, only
  PowerShell 5.1.26100.9168 Desktop edition. The documented Actions default for
  Windows is `pwsh`, so these jobs work only through its fallback: say
  `shell: powershell` explicitly.
- `shell: bash` gives `bash: command not found`. One box has a WindowsApps
  `bash.exe` execution alias which is the WSL stub; the others have nothing.
  `preamble.sh` refuses under both rather than putting the work root somewhere the
  job cannot see.
- `$env:RUNNER_OS` was recorded EMPTY on four of the boxes in an earlier session,
  but read `Windows` correctly on `DESKTOP-NILQUD2` under runner 2.337.0. The
  likely explanation for the blank is the E104 trap itself: `${RUNNER_OS:-?}` is
  bash syntax, and inside a PowerShell block it expands to the empty string
  regardless of the environment. Treat it as suspect (W103), echo it before you
  branch on it, and prefer branching on the job.
- The machine-name prefix `X2-LA01-*` is SHARED with the Linux pool, so the name
  does not tell you which OS answered either.

**`docker` is not installed on any of them** (`docker : The term 'docker' is not
recognized`), so Windows here is a target for non-Docker workloads only.
`capability.py` keeps `windows_docker` as a gap for exactly that reason.

`--spoof-devices` is refused with `--windows`: the device multiplier is an
LD_PRELOAD shim over the HIP runtime and has no Windows equivalent, so multi-GPU
wiring stays unreachable there.

### It has been run end to end

A `--windows --merged` scaffold for PR 10289, pushed to the fork, matched
`DESKTOP-NILQUD2` within seconds and ran the whole path: `preamble.ps1`, the host
gate (`windows: true`, `docker: false`, `windows_docker: false`, `amd_smi: true`),
`states.py` resolving both worktrees with Git for Windows, a venv, `differential.py`
probing base and head, `announce.py`, artifact upload. Total 2m04s.

It ended **INCONCLUSIVE**, and the job went red, which is the toolkit working: the
Studio backend suite collects 25 tests and skips all 25 on Windows, so there was no
comparison to make. A harness that reported that as a pass would be the failure.

### Telling "no matching runner" from "busy pool"

A label matching no runner does not fail the job, it queues it forever, and from
the run list that is indistinguishable from a full pool. The diagnosis is to run
the same workflow twice differing only in the OS label. `--windows` ships that as
a `linux-control` job in the same workflow, so one push answers it: if the control
leg starts and the Windows leg does not, the selector is wrong; if both stall, the
pool is busy. It dispatched in parallel every time it was tried.

## A green tick is not a result

Both Differential steps deliberately survive an unwelcome finding, so job status
reports the plumbing. A run that selects zero tests and returns INCONCLUSIVE in
90 seconds used to show a tick. `lib/announce.py` now closes each Differential
step: it annotates the verdict, and **fails the job on VOID or INCONCLUSIVE**,
which are non-results and belong with a failed hardware gate. Real findings
(CONFIRMED, FIX_INCOMPLETE, NO_REGRESSION) keep the job green. Read the verdict
from the artifact regardless.

## Runner facts worth knowing

- **The pool is MANY machines, not one.** This README previously said four
  ephemeral slots on one machine; that is wrong. A 24-way matrix ran essentially
  concurrently across **seven distinct Linux machines** (`X2-LA01-S01-D05`,
  `S01-D06`, `S02-D01`, `S02-D02`, `S02-D03`, `S02-D04`, `S02-D06`), plus the four
  Windows boxes above. Slot names like `d01`-`d04` are per machine.
- **Consequence, and it is the expensive one: jobs share NO local state.** No
  Docker image cache, no pip cache, no disk. **A workflow that pulls an image in
  job A and uses it in job B is wrong**, and it does not fail loudly, it silently
  re-pulls. That exact assumption cost a re-pull of a 14 GB image.
- Two jobs may still land on the *same* machine, so a GPU measurement still needs
  the shared concurrency group. Nothing in the label set can pin or exclude a
  machine, so "two GPU jobs never run at once" is the only enforceable form of
  "no co-tenant moved my number".
- A queued job waits for a free slot; slot turnaround is about 20 s.
- `$RUNNER_TEMP` is wiped between jobs (19-20 G baseline observed across runs,
  against 69 G at the end of a heavy one). The OS is persistent.
- No root on Linux. Nothing can break system packages. The `sudo` shim makes that
  checkable rather than assumed; `preamble.ps1` writes the same shim on Windows.
- Nothing is cached: every run re-downloads. That is a cost, not a risk.
- gfx1151's KFD encoding is **`110501`**, read off the machine's
  `gfx_target_version` (11 * 10000 + 5 * 100 + 1). Not `110511`.

## Self-test

```bash
python amd_ci/selftest.py
```

Covers the VOID rule, the added-tests-are-not-a-regression rule, absent-test
handling, that the lint rules fire on the shapes that caused real failures, and
that Windows is reachable while Docker on Windows is still declared a gap. Run it
before trusting a change to the toolkit itself.
