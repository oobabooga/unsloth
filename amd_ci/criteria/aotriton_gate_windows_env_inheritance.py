"""Judge PR 8821's AOTriton gate on Windows: placement, override, spawn inheritance.

The Linux leg of this PR asks whether opening the gate changes ROCm SDPA backend
selection, and whether the experimental kernels are still numerically correct. None of
that is answerable here: whether these Windows boxes carry a ROCm-enabled torch is
unmeasured, and torch is not installed in the test venv. That half is declared as a gap,
not quietly reported as a pass.

What this leg judges instead is the half the Linux leg cannot reach:

  1. `setdefault` semantics on Windows, from three starting states. Opening the gate is
     only correct if an explicit "0" survives the import: `c10::utils::check_env` reads
     "0" as false, so "0" is the documented opt-out and overwriting it would take the
     choice away from the user.
  2. Inheritance into a SPAWNED worker. Windows has no fork, so a value set in a
     parent's `os.environ` reaches a worker only if the process environment block
     `CreateProcess` copies was updated with it. Studio's training jobs run under
     `mp.get_context("spawn")`, so the parent-only reading would be worthless to them.
  3. That the Studio backend entry point is no worse off at the head than at the base.
  4. That the `/etc/profile.d` export the PR deletes was not what supplied the variable
     on this platform. It never applied to Windows, but "never applied" is a claim to
     verify, not to assume, so the base leg reads the machine and user environment
     scopes and the directory itself.

The base must show the defect in the form Windows can show it: after `import unsloth`
the gate is unset, no spawned worker sees it, and nothing else on the box supplies it.
If the base already had the gate set, this run proves nothing and the verdict is VOID.
"""

MODE = "differential"
TITLE = ("PR 8821 on Windows: does the gate land, survive an explicit override, "
         "and reach a spawned worker?")

# Authored for what the CHANGE touches, not for what this host has. The variable is read
# inside `#if USE_ROCM` during a ROCm SDPA capability check, so the kernel-selection and
# correctness half needs a ROCm torch this job does not have; `windows` itself is MET
# here, which is the whole point of running this leg.
NEEDS = ["windows", "rocm", "gpu", "windows_rocm_wddm", "windows_docker", "docker",
         "nvidia", "multi_gpu", "xpu", "mlx"]

ENTRIES = ("unsloth", "studio_main")
STARTS = ("unset", "0", "1")
# What the gate must read after the import, per starting state. `setdefault` writes "1"
# only into the unset case; "0" is the opt-out and has to survive.
EXPECTED = {"unset": "1", "0": "0", "1": "1"}
CHILDREN = ("mp_spawn_child", "popen_inherit_child", "popen_env_copy_child")

# gates() runs before any comparison (differential.py calls it first), so it is also
# where the base reading is stashed for the one judgement that needs both sides: whether
# the Studio entry point got worse. head_is_fixed only receives one state.
_BASELINE: dict = {}


def _run(state, entry, start):
    return ((state or {}).get("runs") or {}).get(f"{entry}|{start}") or {}


def _child_gate(run, child):
    value = run.get(child)
    return value.get("gate") if isinstance(value, dict) else "<missing>"


def gates(obs):
    out = []
    for name in [n for n in obs if not n.startswith("_")]:
        state = obs.get(name) or {}

        missing = [f"{e}|{s}" for e in ENTRIES for s in STARTS if not _run(state, e, s)]
        bad = [f"{e}|{s}" for e in ENTRIES for s in STARTS
               if _run(state, e, s).get("rc") != 0 or _run(state, e, s).get("timeout")]
        out.append((
            f"{name}: all six (entry x starting state) runs completed",
            not missing and not bad,
            f"missing={missing or 'none'} nonzero_or_timeout={bad or 'none'}",
        ))

        # If this landed on Linux the whole leg is about nothing. The probe records the
        # OS it saw and so does each inner process; both have to say Windows.
        inner = {_run(state, e, s).get("platform_system") for e in ENTRIES for s in STARTS}
        out.append((
            f"{name}: this really ran on Windows",
            state.get("probe_platform_system") == "Windows" and inner == {"Windows"},
            f"probe={state.get('probe_platform_system')!r} inner={sorted(x for x in inner if x)} "
            f"host={state.get('probe_computername')}",
        ))

        # The control child is handed an environment with the variable removed. If it
        # ever reports a value, the reader is not reading what it claims to and every
        # inheritance reading below is worthless.
        readings = {f"{e}|{s}": _child_gate(_run(state, e, s), "popen_scrubbed_child")
                    for e in ENTRIES for s in STARTS}
        leaked = [k for k, v in readings.items() if v not in (None, "<missing>")]
        unread = [k for k, v in readings.items() if v == "<missing>"]
        out.append((
            f"{name}: the scrubbed control child saw nothing",
            not leaked and not unread,
            (f"reported a value in {leaked}; " if leaked else "")
            + (f"no reading at all from {unread}" if unread else
               ("None in all six runs" if not leaked else "")),
        ))

    base, head = obs.get("base") or {}, obs.get("head") or {}
    _BASELINE.clear()
    _BASELINE.update(base)

    out.append((
        "only the head carries the gate statement",
        all(base.get(f"{e}_sets_gate") is False and head.get(f"{e}_sets_gate") is True
            for e in ENTRIES),
        "; ".join(f"{e}: base={base.get(f'{e}_sets_gate')} head={head.get(f'{e}_sets_gate')}"
                  for e in ENTRIES),
    ))

    # Non-vacuity for the inheritance question. With the variable set explicitly in the
    # BASE job's environment, a spawned worker must see it. Without this, "the head's
    # worker sees 1" could just mean the probe reads children correctly never.
    proofs = {f"{e}|{c}": _child_gate(_run(base, e, "1"), c)
              for e in ENTRIES for c in CHILDREN}
    out.append((
        "a spawned worker can be observed inheriting a value at all (base, start=1)",
        all(v == "1" for v in proofs.values()),
        "; ".join(f"{k}={v!r}" for k, v in proofs.items()),
    ))

    return out


def _rows(obs, names):
    rows = []
    for entry in ENTRIES:
        for start in STARTS:
            for key, label in (
                ("gate_after_import", "after import"),
                ("mp_spawn_child", "mp spawn worker"),
                ("popen_inherit_child", "subprocess (inherited)"),
                ("popen_env_copy_child", "subprocess (env copy)"),
                ("popen_scrubbed_child", "control (scrubbed)"),
            ):
                cells = []
                for name in names:
                    run = _run(obs.get(name) or {}, entry, start)
                    value = (run.get(key) if key == "gate_after_import"
                             else _child_gate(run, key))
                    cells.append("unset" if value is None else f"`{value}`")
                rows.append(f"| `import {entry}`, start={start} | {label} | "
                            + " | ".join(cells) + " |")
    return rows


def table(obs):
    names = [n for n in obs if not n.startswith("_")]
    lines = [
        "| case | reading | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
        *_rows(obs, names),
    ]

    lines.append("")
    lines.append("| fact | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for key, label in (
        ("probe_computername", "machine"),
        ("probe_platform_system", "OS as the probe saw it"),
        ("probe_python", "python"),
        ("gate_in_job_environment", "gate in the job environment"),
        ("gate_in_machine_scope", "gate in the Machine env scope"),
        ("gate_in_user_scope", "gate in the User env scope"),
        ("etc_profile_d_exists", "/etc/profile.d exists"),
        ("unsloth_sets_gate", "unsloth/__init__.py sets the gate"),
        ("studio_main_sets_gate", "studio/backend/main.py sets the gate"),
    ):
        lines.append(f"| {label} | "
                     + " | ".join(str((obs.get(n) or {}).get(key, "-")) for n in names)
                     + " |")

    # How far each entry point's module body ran. The gate sits near the top of both
    # files, so "the import failed" and "the gate never executed" are different claims.
    lines.append("")
    lines.append("| entry point import (start=unset) | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for entry in ENTRIES:
        for key, label in (("import_ok", "import succeeded"),
                           ("import_error", "error"),
                           ("deepest_line_in_entry_file", "deepest line reached in the file")):
            lines.append(f"| {entry}: {label} | "
                         + " | ".join(str(_run(obs.get(n) or {}, entry, "unset").get(key, "-"))
                                      for n in names) + " |")

    lines.append("")
    lines.append("| the PR's own guard tests, run here | "
                 + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for key, label in (("exists", "tests/python/test_rocm_aotriton_gate.py present"),
                       ("rc", "pytest exit code"),
                       ("summary", "summary")):
        lines.append(f"| {label} | "
                     + " | ".join(str(((obs.get(n) or {}).get("pr_own_tests") or {}).get(key, "-"))
                                  for n in names) + " |")

    lines.append("")
    base = obs.get("base") or {}
    shown, why_base = base_shows_defect(base)
    lines.append(f"- base: {'shows' if shown else 'does NOT show'} the defect - {why_base}")
    for name in names:
        if name == "base":
            continue
        ok, why = head_is_fixed(obs.get(name) or {})
        lines.append(f"- {name}: {'clears' if ok else 'does NOT clear'} it - {why}")
    return "\n".join(lines)


def base_shows_defect(base):
    if not base:
        return False, "no base observation"

    # Something other than the change supplying the variable would make every base
    # reading meaningless, so rule it out before reading the import results.
    for key, label in (("gate_in_job_environment", "the job environment"),
                       ("gate_in_machine_scope", "the Machine environment scope"),
                       ("gate_in_user_scope", "the User environment scope")):
        if base.get(key) is not None:
            return False, (
                f"the gate was already set in {label} ({base.get(key)!r}) before anything "
                f"was imported, so no reading here is attributable to the change")

    still_set = [e for e in ENTRIES if _run(base, e, "unset").get("gate_after_import") is not None]
    if still_set:
        return False, (
            f"at the base, importing {' and '.join(still_set)} already left the gate set, "
            f"so the defect does not reproduce here")

    seen = {f"{e}|{c}": _child_gate(_run(base, e, "unset"), c)
            for e in ENTRIES for c in CHILDREN}
    inherited = {k: v for k, v in seen.items() if v is not None}
    if inherited:
        return False, (
            f"at the base a spawned worker already saw the gate: {inherited}")

    profile_d = base.get("etc_profile_d_exists")
    return True, (
        "at the base, importing unsloth and importing the Studio backend entry point both "
        "leave TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL unset, no spawned worker sees it, "
        "and nothing else on this box supplies it (Machine scope unset, User scope unset, "
        f"/etc/profile.d present={profile_d}) - so on Windows the deleted profile.d export "
        "was never what set it, and without this change the gate is shut")


def head_is_fixed(state):
    if not state:
        return False, "no observation for this state"

    wrong = []
    for entry in ENTRIES:
        for start in STARTS:
            run = _run(state, entry, start)
            want = EXPECTED[start]
            got = run.get("gate_after_import")
            if got != want:
                wrong.append(f"import {entry} with start={start}: gate={got!r}, expected {want!r}")
    if wrong:
        return False, "; ".join(wrong)

    # The Windows-specific half: the value has to reach a spawned worker, and it has to
    # be the value the parent ended up with, not a forced "1". A worker that saw "1"
    # after the user opted out with "0" would be the same bug in the other direction.
    not_inherited = []
    for entry in ENTRIES:
        for start in STARTS:
            run = _run(state, entry, start)
            for child in CHILDREN:
                got = _child_gate(run, child)
                if got != EXPECTED[start]:
                    not_inherited.append(
                        f"{entry}/start={start}/{child}: {got!r}, expected {EXPECTED[start]!r}")
    if not_inherited:
        return False, ("the gate did not survive the spawn: " + "; ".join(not_inherited))

    # The Studio backend entry point must be no worse off than it was at the base. It is
    # imported without the server stack here, so both sides usually stop at the same
    # missing third-party module; a head that stops EARLIER is the finding.
    base_studio = _run(_BASELINE, "studio_main", "unset")
    head_studio = _run(state, "studio_main", "unset")
    if base_studio.get("import_ok") and not head_studio.get("import_ok"):
        return False, (
            f"the Studio backend entry point imported at the base and fails here: "
            f"{head_studio.get('import_error')}")
    base_line = base_studio.get("deepest_line_in_entry_file")
    head_line = head_studio.get("deepest_line_in_entry_file")
    if (isinstance(base_line, int) and isinstance(head_line, int)
            and head_line + 40 < base_line):
        return False, (
            f"studio/backend/main.py stopped at line {head_line} here against {base_line} "
            f"at the base, so the head's module body gets materially less far")

    tests = state.get("pr_own_tests") or {}
    if tests.get("exists") and tests.get("rc") not in (0, None):
        return False, (
            f"the PR's own guard tests fail on Windows: {tests.get('summary') or tests.get('tail', '')[-300:]}")

    note = ""
    if not head_studio.get("import_ok"):
        note = (f" (the Studio entry point still stops at `{head_studio.get('import_error')}`, "
                f"the same place as the base: the server stack is not installed on these "
                f"boxes, so what is shown is that the gate lands above every third-party "
                f"import, not a full backend start)")
    return True, (
        "importing unsloth and importing the Studio backend entry point both set the gate "
        "to \"1\" when it is unset, leave an explicit \"0\" and an explicit \"1\" alone, and "
        "the resulting value reaches a spawn-started multiprocessing worker and both "
        "subprocess shapes" + note)
