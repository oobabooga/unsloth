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
  W101  A GPU-measuring job without a concurrency group. Slots share one GPU.
  W102  `--fail-under`-style skips: `exit 0` on a missing device, which reports
        "nothing to see" for "we did not get the hardware".

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


def _step_shell(job: dict, step: dict) -> str:
    """The shell a step actually runs under: the step's own `shell:`, else the
    job's `defaults.run.shell`, else GitHub's default of bash."""
    own = step.get("shell")
    if own:
        return str(own).split()[0].lower()
    dflt = ((job.get("defaults") or {}).get("run") or {}).get("shell")
    return str(dflt).split()[0].lower() if dflt else "bash"


def _steps(doc: dict):
    for job_name, job in (doc.get("jobs") or {}).items():
        for i, step in enumerate(job.get("steps") or []):
            if isinstance(step, dict) and "run" in step:
                label = step.get("name") or f"step[{i}]"
                yield job_name, job, f"{job_name}/{label}", step["run"], _step_shell(job, step)


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


def lint_powershell_syntax(where: str, body: str) -> list[Finding]:
    """No PowerShell on this host, so the check is structural: balanced braces
    and parentheses outside strings, which catches a cut-off `function {` or a
    stray `)` without pretending to parse the language."""
    scrubbed = re.sub(r"\$\{\{[^}]*\}\}", "PLACEHOLDER", body)
    scrubbed = re.sub(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'', "", scrubbed)
    scrubbed = re.sub(r"#[^\n]*", "", scrubbed)
    depth = {"{": 0, "(": 0}
    for ch in scrubbed:
        if ch in "{(":
            depth[ch] += 1
        elif ch == "}":
            depth["{"] -= 1
        elif ch == ")":
            depth["("] -= 1
        if depth["{"] < 0 or depth["("] < 0:
            return [("E000", where, "powershell: closing bracket without an opener")]
    if depth["{"] or depth["("]:
        return [("E000", where, f"powershell: unbalanced brackets (braces {depth['{']}, parens {depth['(']})")]
    return []


def lint_workflow(path: Path) -> list[Finding]:
    doc = yaml.safe_load(path.read_text())
    findings: list[Finding] = []
    gpu_jobs: set[str] = set()
    for job_name, job, where, body, shell in _steps(doc):
        findings += lint_text(where, body)
        # `bash -n` only means something for a bash step; a PowerShell body is
        # not bash and would fail on its first `&` or `$env:` for no reason.
        if shell in ("powershell", "pwsh"):
            findings += lint_powershell_syntax(where, body)
        else:
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
                             "touches GPU memory but declares no concurrency group; AMD CI slots "
                             "share one GPU, so a co-tenant can move the number being measured"))
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
