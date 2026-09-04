#!/usr/bin/env python3
"""Catch the workflow bugs that cost a real CI run, before spending one.

Every rule here corresponds to a run that was actually lost. They are cheap,
local, and none of them need a runner:

  E001  `rc=$?` in a step that never disables errexit. GitHub runs step shells
        as `bash -e`, so the failing command aborts the step and rc is never
        read. Fatal in any job whose subject is "does this fail".
  E002  `pkill -f <pattern>` where the pattern also matches the calling shell's
        own command line, so the step kills itself. Kill by recorded PID.
  E003  A heredoc terminator that is indented without `<<-`, so it never closes.
  E004  Parsing a probe's stdout as JSON. Importing an application module often
        prints a banner and corrupts it. Probes must write to a file.
  E005  `set -e`/pipefail plus a bare glob in a command substitution: a miss
        becomes a step failure.
  E006  A backgrounded payload inside `run_in_background`-style usage, or a
        trailing `&` on the work itself, which reports completion immediately.
  E007  A relative `amd_ci/...` path used after the step has `cd`'d elsewhere.
        The toolkit lives at $GITHUB_WORKSPACE; a state worktree does not
        contain it, so the command dies with ENOENT.
  W101  A GPU-measuring job without a concurrency group. Two GPU jobs can land
        on the same machine and move each other's numbers.
  W102  `--fail-under`-style skips: `exit 0` on a missing device, which reports
        "nothing to see" for "we did not get the hardware".

The E10x family is the Windows AMD runners, which are a first-class target and
not a Linux runner with a different label. Every one of these was measured on
the four boxes, and every one of them fails the job before the first line of the
step runs, or worse, succeeds while measuring nothing:

  E100  A Windows job whose shell is not `powershell`. `pwsh` is not installed
        (PowerShell 5.1 Desktop only) and neither is `bash`. An UNSET shell is
        flagged too: the documented Actions default for Windows is `pwsh`, and
        these jobs only work by way of its fallback.
  E101  `docker` in a Windows job. Not installed on any of the four.
  E102  bash, `sh`, or a `.sh` script in a Windows job, `preamble.sh` above all.
        The Windows preamble is `lib/preamble.ps1`.
  E103  Branching on `$env:RUNNER_OS` in a Windows job. It is EMPTY there, so
        the branch silently takes the wrong arm rather than failing.
  E104  A bash-style `$RUNNER_TEMP` inside a PowerShell block. PowerShell reads
        that as an undefined variable and expands it to the empty string, so the
        step writes to the wrong place and still exits 0.
  E106  A self-hosted strix-halo selector with NO OS label. The pool now answers
        with both Linux and Windows machines, so such a job lands on whichever
        replies first and the workflow silently ran on the other OS.
  W104  `Out-File -Encoding utf8` into GITHUB_ENV from PowerShell 5.1, which
        writes a BOM (there is no utf8NoBOM before PowerShell 6). `Add-Content`
        does not.

Usage:  python lint_workflow.py path/to/workflow.yml [...]
Exit 1 if any E-rule fires.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

Finding = tuple[str, str, str]  # (code, where, message)


def _default_shell(container: dict | None) -> str | None:
    """`defaults: {run: {shell: ...}}` at the workflow or job level."""
    if not isinstance(container, dict):
        return None
    run = (container.get("defaults") or {}).get("run") or {}
    return run.get("shell") if isinstance(run, dict) else None


def _runs_on_labels(job: dict) -> list[str]:
    ro = job.get("runs-on")
    if isinstance(ro, str):
        return [ro]
    if isinstance(ro, list):
        return [str(x) for x in ro]
    if isinstance(ro, dict):  # {group: ..., labels: [...]}
        labels = ro.get("labels") or []
        return [str(x) for x in (labels if isinstance(labels, list) else [labels])]
    return []


def is_windows_job(job: dict) -> bool:
    """True when the selector asks for a Windows runner.

    Label matching on the AMD pool is exact but case-insensitive, and the label
    the Windows boxes carry is `Windows`. `windows-latest` is here so the rules
    also fire on a hosted-runner job someone pasted in.
    """
    return any(l.lower() == "windows" or l.lower().startswith("windows-")
               for l in _runs_on_labels(job))


def _steps(doc: dict):
    wf_default = _default_shell(doc)
    for job_name, job in (doc.get("jobs") or {}).items():
        job_default = _default_shell(job) or wf_default
        for i, step in enumerate(job.get("steps") or []):
            if isinstance(step, dict) and "run" in step:
                label = step.get("name") or f"step[{i}]"
                shell = step.get("shell") or job_default
                yield job_name, job, f"{job_name}/{label}", step["run"], shell


def lint_text(where: str, body: str) -> list[Finding]:
    out: list[Finding] = []
    disables_errexit = re.search(r"^\s*set\s+\+e\b", body, re.M) is not None

    if re.search(r"^\s*rc=\$\?", body, re.M) and not disables_errexit:
        out.append(("E001", where,
                    "reads `rc=$?` but never runs `set +e`; GitHub's shell is `bash -e`, so a "
                    "non-zero exit aborts the step before rc is read"))

    for m in re.finditer(r"pkill\s+(-\w+\s+)*-f\s+[\"']?([^\"'\n]+)", body):
        pat = m.group(2).strip()
        out.append(("E002", where,
                    f"`pkill -f {pat}`: the pattern also matches the shell running it, so the "
                    f"step can kill itself. Record the PID and `kill \"$PID\"`"))

    for m in re.finditer(r"<<\s*(-?)\s*'?([A-Za-z_][A-Za-z0-9_]*)'?", body):
        dash, tag = m.group(1), m.group(2)
        if dash:
            continue
        # An unindented terminator must exist somewhere after the opener.
        rest = body[m.end():]
        if not re.search(rf"^{re.escape(tag)}\s*$", rest, re.M):
            if re.search(rf"^\s+{re.escape(tag)}\s*$", rest, re.M):
                out.append(("E003", where,
                            f"heredoc `{tag}` is closed by an INDENTED terminator without `<<-`, "
                            f"so it never closes"))

    if re.search(r"(python3?|jq)[^\n|]*\|\s*(python3?\b[^\n]*json\.load|jq\b)", body):
        out.append(("E004", where,
                    "pipes a program's stdout into a JSON parser; module import banners corrupt "
                    "it. Have the probe write JSON with `--out` and read the file"))

    if re.search(r"pipefail", body) and not disables_errexit:
        for m in re.finditer(r"\$\(\s*(ls|find|grep|pgrep)\s+[^)]*\)", body):
            frag = m.group(0)
            has_glob = "*" in frag
            piped = "|" in frag
            if not (has_glob or piped):
                continue
            # `2>/dev/null` hides the message, NOT the exit status, and under
            # pipefail the pipeline still inherits the failure. Only `|| true`
            # (or a conditional context) actually makes the miss survivable.
            # This exact shape cost a run: an ls over a not-yet-created path
            # returned 2 and failed the gate step.
            if "|| true" in frag or "||true" in frag:
                continue
            out.append(("E005", where,
                        f"`{frag[:60]}` can miss under `set -e` + `pipefail` and fail the step; "
                        f"`2>/dev/null` suppresses the message, not the exit status. "
                        f"Append `|| true`"))

    for line in body.splitlines():
        s = line.strip()
        if s.endswith("&") and not s.endswith("&&") and "nohup" in s:
            out.append(("E006", where,
                        "`nohup ... &` backgrounds the work out of the step's shell, so the step "
                        "exits before the work finishes"))

    # E007. The toolkit is only at the checkout root, so a relative reference to
    # it stops resolving the moment the step cd's into a state worktree. A `cd`
    # back to $GITHUB_WORKSPACE (or a bare `cd -`) restores it.
    cwd_moved = False
    for line in body.splitlines():
        s = line.strip()
        m = re.match(r"cd\s+(\S+)", s)
        if m:
            target = m.group(1).strip('"\'')
            cwd_moved = target not in ("-", "$GITHUB_WORKSPACE", "${GITHUB_WORKSPACE}")
            continue
        if not cwd_moved:
            continue
        for ref in re.findall(r"(?<![\w/$\"'])amd_ci/[\w/.\-]+\.(?:py|sh)", s):
            out.append(("E007", where,
                        f"`{ref}` is relative but this step has cd'd away from $GITHUB_WORKSPACE, "
                        f"so it resolves inside the worktree and does not exist; "
                        f'use "$GITHUB_WORKSPACE/{ref}"'))
    return out


# Env vars a step is likely to reach for. In bash `$RUNNER_TEMP`; in PowerShell
# the same text is an undefined variable that expands to "", which is why E104 is
# an error and not a style note.
GITHUB_ENV_NAMES = (
    "RUNNER_TEMP", "RUNNER_OS", "RUNNER_NAME", "GITHUB_ENV", "GITHUB_WORKSPACE",
    "GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "GITHUB_PATH", "AMD_CI_WORK",
    "AMD_CI_PY", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
)


def lint_windows(where: str, body: str, shell: str | None) -> list[Finding]:
    """Rules for a step that runs on one of the Windows AMD runners."""
    out: list[Finding] = []

    if shell is None:
        out.append(("E100", where,
                    "no `shell:` on a Windows job. The documented Actions default for "
                    "Windows is `pwsh`, which is NOT installed on these runners; the job "
                    "only works via a fallback. Set `shell: powershell` explicitly"))
    elif shell.strip().lower() != "powershell":
        why = {
            "pwsh": "`pwsh: command not found`; there is no PowerShell 7 on these runners",
            "bash": "`bash: command not found`; one box has a WindowsApps bash.exe that is "
                    "the WSL stub, the others have none",
            "sh": "no POSIX shell on these runners",
        }.get(shell.strip().lower(), "not available on these runners")
        out.append(("E100", where,
                    f"`shell: {shell}` on a Windows job: {why}. The only working value is "
                    f"`shell: powershell` (PowerShell 5.1 Desktop)"))

    if re.search(r"(?<![\w.-])docker(?![\w.-])", body):
        out.append(("E101", where,
                    "runs `docker` in a Windows job. Measured on all four Windows AMD "
                    "runners: not installed (`docker : The term 'docker' is not "
                    "recognized`). Containerised work belongs on the Linux runners"))

    if re.search(r"preamble\.sh", body):
        out.append(("E102", where,
                    "sources `preamble.sh` in a Windows job. It is Linux only and refuses "
                    "to run under a Windows bash; use `lib/preamble.ps1`"))
    elif re.search(r"(?<![\w.-])(bash|sh)\s|\.sh(?![\w.])", body):
        out.append(("E102", where,
                    "invokes a POSIX shell or a `.sh` script in a Windows job; neither "
                    "`bash` nor `sh` is installed on these runners"))

    if re.search(r"RUNNER_OS", body):
        out.append(("E103", where,
                    "reads `RUNNER_OS`, which is EMPTY on these Windows jobs (measured: a "
                    "`${RUNNER_OS:-?}` echo printed blank). A branch on it silently takes "
                    "the wrong arm. Branch on the job, not on the environment"))

    if (shell or "").strip().lower() != "bash":
        for name in GITHUB_ENV_NAMES:
            if re.search(rf"(?<!env:)\${{?{name}\b", body):
                out.append(("E104", where,
                            f"`${name}` is bash syntax. PowerShell reads it as an undefined "
                            f"variable and expands it to the empty string, so the step "
                            f"silently acts on the wrong path and still exits 0. "
                            f"Use `$env:{name}`"))

    if re.search(r"Out-File[^\n]*GITHUB_(ENV|OUTPUT|PATH)", body):
        out.append(("W104", where,
                    "writes GITHUB_ENV with `Out-File`. On PowerShell 5.1 `-Encoding utf8` "
                    "means utf8 WITH BOM (there is no utf8NoBOM before PowerShell 6) and the "
                    "BOM can corrupt the first name written. Use `Add-Content`"))
    return out


def lint_shell_syntax(where: str, body: str) -> list[Finding]:
    """`bash -n` on the block, with GitHub's expression syntax neutralised."""
    scrubbed = re.sub(r"\$\{\{[^}]*\}\}", "PLACEHOLDER", body)
    with tempfile.NamedTemporaryFile("w", suffix = ".sh", delete = False) as fh:
        fh.write(scrubbed)
        p = fh.name
    try:
        r = subprocess.run(["bash", "-n", p], capture_output = True, text = True)
        if r.returncode != 0:
            first = (r.stderr.strip().splitlines() or ["syntax error"])[0]
            return [("E000", where, f"bash syntax error: {first}")]
    finally:
        Path(p).unlink(missing_ok = True)
    return []


def lint_workflow(path: Path) -> list[Finding]:
    doc = yaml.safe_load(path.read_text())
    findings: list[Finding] = []
    gpu_jobs: set[str] = set()

    # E106 first: a selector with no OS label is now ambiguous, because the pool
    # answers with both Linux and Windows machines. Before the Windows boxes were
    # reachable this was merely redundant.
    for job_name, job in (doc.get("jobs") or {}).items():
        labels = [l.lower() for l in _runs_on_labels(job)]
        if "self-hosted" in labels and "strix-halo" in labels:
            if not any(l == "linux" or l == "windows" or l.startswith("windows-")
                       for l in labels):
                findings.append(("E106", job_name,
                                 "selects the strix-halo pool with no OS label. The pool "
                                 "answers with BOTH Linux and Windows machines, so this job "
                                 "lands on whichever replies first. Add `Linux` or `Windows`"))

    for job_name, job, where, body, shell in _steps(doc):
        windows = is_windows_job(job)
        if windows:
            # The bash rules are skipped rather than adapted: `rc=$?`, `pipefail`
            # and `bash -n` describe a shell this step will never run under, and a
            # linter that reports a bash syntax error for a PowerShell block gets
            # ignored wholesale. The rules that are about the JOB, not the shell
            # (W101, W102), still apply and are below.
            findings += lint_windows(where, body, shell)
        else:
            findings += lint_text(where, body)
            if (shell or "bash").strip().lower() in ("bash", "sh"):
                findings += lint_shell_syntax(where, body)
        if re.search(r"mem_get_info|memory_allocated|rocm-smi|nvidia-smi|torch\.cuda", body):
            env = {**(doc.get("env") or {}), **(job.get("env") or {})}
            masked = env.get("HIP_VISIBLE_DEVICES") == "" or \
                env.get("CUDA_VISIBLE_DEVICES") == ""
            if not masked:
                gpu_jobs.add(job_name)
        # Only when a hardware-absence test GUARDS the exit, not merely when the
        # words appear somewhere in a long block. A noisy warning gets ignored,
        # and this one is worth reading.
        lines = body.splitlines()
        for i, line in enumerate(lines):
            if not re.search(r"\bexit\s+0\b", line):
                continue
            window = " ".join(lines[max(0, i - 2):i + 1])
            absent = re.search(
                r"(no|not|missing|absent|cannot see|unavailable)\b[^\n]{0,40}"
                r"\b(gpu|device|cuda|hip|rocm|vulkan|torch)\b", window, re.I)
            guarded = re.search(r"(if\s|\|\||&&|then\b)", window)
            if absent and guarded:
                findings.append(("W102", where,
                                 "exits 0 when the hardware looks absent; a gate should FAIL, "
                                 "since a skip misreports missing hardware as nothing to see"))
                break

    for job_name in sorted(gpu_jobs):
        job = doc["jobs"][job_name]
        if not job.get("concurrency"):
            findings.append(("W101", job_name,
                             "touches GPU memory but declares no concurrency group. The pool is "
                             "many machines, not one, so this is no longer a certainty; but two "
                             "GPU jobs can still land on the SAME machine, and then a co-tenant "
                             "moves the number being measured"))
    return findings


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: lint_workflow.py <workflow.yml> [...]")
        return 2
    total_err = 0
    for p in paths:
        findings = lint_workflow(p)
        print(f"== {p}")
        if not findings:
            print("   clean")
        for code, where, msg in findings:
            print(f"   {code} {where}: {msg}")
            if code.startswith("E"):
                total_err += 1
    print(f"\n{total_err} error-level finding(s)")
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())
