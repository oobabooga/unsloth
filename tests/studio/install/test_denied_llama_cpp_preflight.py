# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The installer must answer "can I read ~/.unsloth/llama.cpp?" before it downloads.

An elevated PowerShell installs into the same %USERPROFILE% as the ordinary
account, so a run started with "Run as administrator" can leave an admin-owned
%USERPROFILE%\\.unsloth\\llama.cpp behind that the account cannot open. The tree
outlives an uninstall, so a later normal install finds it and fails.

setup.ps1 reports that properly since #7735 / #7757, but originally only in
PHASE 3.4, after the 2.8 GB PyTorch download and the whole dependency install.
Both install.ps1 and direct setup/update now ask the same question before phase 1.

"the same question" is literal: install.ps1 is fetched standalone (irm | iex) and
bundled as a Tauri resource, so it cannot dot-source studio/setup.ps1, and instead
carries verbatim copies of the probe and the guidance. This module compares them
function by function, the way tests/studio/install/test_rocm_arch_table_parity.py
compares the gfx tables that are hand-copied across the installers.

Behavioural coverage against a real denied tree (icacls /deny on Windows, chmod on
POSIX) lives in tests/studio/test_path_probe_access_denied.ps1, which runs the real
functions and the real preflight.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
INSTALL_PS1 = (ROOT / "install.ps1").read_text(encoding = "utf-8")
SETUP_PS1 = (ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")
SETUP_SH = (ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
ACL_PS1 = (ROOT / "tests" / "studio" / "test_path_probe_access_denied.ps1").read_text(
    encoding = "utf-8"
)

SHARED_BEGIN = "# ── BEGIN SHARED WITH studio/setup.ps1 ──"
SHARED_END = "# ── END SHARED WITH studio/setup.ps1 ──"

# The probe, the denial classifier it rests on, and the guidance. Each one is a
# byte-identical copy in install.ps1; see the module docstring for why.
SHARED_FUNCTIONS = (
    "Test-AccessDeniedError",
    "Get-PathState",
    "Get-LlamaCppInstallReadState",
    "Get-PathDenialDetail",
    "Write-PathAccessDenied",
    "Get-CanonicalDir",
    "Test-StudioHomeIsCustom",
    "Get-ManagedLlamaCppDir",
    "Invoke-ManagedLlamaCppPreflight",
)


def _function_source(text: str, name: str) -> str:
    """The `function NAME { ... }` block, by brace matching.

    A Python port of tests/studio_setup_ps1/Get-FunctionSource.ps1, and naive for
    the same reason: these bodies contain only balanced braces.
    """
    match = re.search(rf"(?im)^[ \t]*function[ \t]+{re.escape(name)}\b", text)
    assert match, f"{name} is not defined"
    start = text.index("{", match.start())
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[match.start():index + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def _normalized(source: str) -> str:
    """Strip the common indentation so a nested copy compares equal to a top-level one.

    install.ps1's helpers live inside `function Install-UnslothStudio`, setup.ps1's
    are at column 0. Nothing else is normalized: wording, ordering and logic must
    match character for character.
    """
    lines = [line.rstrip() for line in source.splitlines()]
    body = [line for line in lines if line.strip()]
    indent = min(len(line) - len(line.lstrip(" ")) for line in body)
    return "\n".join(line[indent:] if line.strip() else "" for line in lines)


@pytest.mark.parametrize("name", SHARED_FUNCTIONS)
def test_installer_copy_matches_setup(name: str) -> None:
    """Edit one file, not the other, and this fails naming the function."""
    assert _normalized(_function_source(INSTALL_PS1, name)) == _normalized(
        _function_source(SETUP_PS1, name)
    ), name


def test_every_function_in_the_shared_block_is_compared() -> None:
    """The parity list cannot silently fall behind the block it guards."""
    assert SHARED_BEGIN in INSTALL_PS1
    assert SHARED_END in INSTALL_PS1
    block = INSTALL_PS1.split(SHARED_BEGIN, 1)[1].split(SHARED_END, 1)[0]
    declared = re.findall(r"(?m)^[ \t]*function[ \t]+([A-Za-z-]+)", block)
    assert sorted(declared) == sorted(SHARED_FUNCTIONS), declared


def test_the_shared_block_carries_the_drift_note() -> None:
    """A reader who does not know these are copies will paraphrase one of them."""
    head = INSTALL_PS1.split(SHARED_BEGIN, 1)[0].rsplit("\n\n", 3)[-1]
    assert "byte-identical copies" in head
    assert "test_denied_llama_cpp_preflight.py" in head
    assert "cannot dot-source" in head
    setup_note = SETUP_PS1.split("function Get-LlamaCppInstallReadState", 1)[0]
    assert "install.ps1 carries a verbatim copy" in setup_note


def test_setup_and_the_installer_use_the_same_probe() -> None:
    """The point of sharing it: a denial that stops the installer also stops setup,
    and one that does not stop setup does not stop the installer."""
    assert "$llamaDirState = Get-LlamaCppInstallReadState -Path $LlamaCppDir" in SETUP_PS1
    assert '$llamaDirState -eq "Denied"' in SETUP_PS1
    assert '$llamaDirState -eq "Readable"' in SETUP_PS1
    assert "(Get-LlamaCppInstallReadState -Path $dir) -ne \"Denied\"" in INSTALL_PS1


def test_the_probe_keeps_all_three_answers() -> None:
    """Collapsing Denied into Absent is the original bug; collapsing Absent into
    Denied would stop a clean install that has no llama.cpp at all."""
    probe = _function_source(SETUP_PS1, "Get-LlamaCppInstallReadState")
    for verdict in ('return "Denied"', 'return "Absent"', 'return "Readable"'):
        assert verdict in probe, verdict
    # Windows reports a MISSING child of a denied directory as absent, so the
    # metadata probe alone answers "no install here" on a tree it cannot look
    # into. The listing is what closes that.
    assert "Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop" in probe
    assert "Test-AccessDeniedError" in probe
    # It runs before anything is installed, so it must not terminate.
    assert probe.count("try {") == 1
    assert "catch" in probe


def test_the_preflight_runs_before_anything_expensive() -> None:
    """The whole fix is the ordering. Every step below costs network, disk or both."""
    call = "$llamaPreflightFailure = Invoke-ManagedLlamaCppPreflight"
    assert INSTALL_PS1.count(call) == 1
    position = INSTALL_PS1.index(call)
    for later in (
        'Write-TauriLog "STEP" "Checking system dependencies"',
        'Write-TauriLog "STEP" "Installing Python"',
        'Write-TauriLog "STEP" "Installing uv package manager"',
        'Write-TauriLog "STEP" "Creating virtual environment"',
        'Write-TauriLog "STEP" "Installing PyTorch"',
        'Write-TauriLog "STEP" "Installing unsloth"',
        'Write-TauriLog "STEP" "Running studio setup"',
    ):
        assert position < INSTALL_PS1.index(later), later
    # And after the System32 relocation, which decides whether this install
    # belongs to a real user profile and therefore which .unsloth is meant.
    relocation = "# ── Leave Windows system directories before installing ──"
    assert INSTALL_PS1.index(relocation) < position


def test_direct_setup_and_update_preflight_before_phase_one() -> None:
    """setup.ps1 is also the complete entrypoint for CLI setup/update and the
    desktop repair update, so install.ps1 ordering alone is insufficient."""
    call = "$llamaPreflightFailure = Invoke-ManagedLlamaCppPreflight"
    assert SETUP_PS1.count(call) == 1
    position = SETUP_PS1.index(call)
    assert SETUP_PS1.index("$LlamaCppDir = Get-ManagedLlamaCppDir") < position
    for later in (
        "PHASE 1: System-level prerequisites",
        "PHASE 2: Frontend build",
        "PHASE 3: Python environment + dependencies",
        "PHASE 3.4: Prefer prebuilt llama.cpp",
    ):
        assert position < SETUP_PS1.index(later), later
    failure = SETUP_PS1[position:SETUP_PS1.index("# Back up User PATH", position)]
    assert "Exit-SetupFailure $llamaPreflightFailure" in failure


def test_acl_suite_runs_every_complete_windows_entrypoint() -> None:
    """Windows CI must exercise actual script invocation and block every escape
    into network or dependency work, not only evaluate extracted functions."""
    assert '& (Join-Path $repoRoot "install.ps1") --tauri' in ACL_PS1
    assert '& (Join-Path $repoRoot "studio/setup.ps1")' in ACL_PS1
    assert 'foreach ($mode in @("install", "setup", "update", "repair"))' in ACL_PS1
    assert "icacls $entryLocked /deny" in ACL_PS1
    assert "else { chmod 000 $entryLocked }" in ACL_PS1
    for trap in (
        "Invoke-WebRequest",
        "Invoke-RestMethod",
        "Start-Process",
        "winget",
        "python",
        "uv",
        "git",
        "npm",
    ):
        assert f'function global:{trap} {{ Stop-EntrypointExpense "{trap}" }}' in ACL_PS1
    for marker in (
        "Checking system dependencies",
        "frontend",
        "Installing Python",
        "Installing uv package manager",
        "Creating virtual environment",
        "Installing PyTorch",
        "Installing unsloth",
        "Unsloth Studio Installed",
    ):
        assert marker in ACL_PS1
    assert ".unsloth-studio-owned" in ACL_PS1
    assert "unsloth_install_manifest.json" in ACL_PS1


def test_the_preflight_fails_the_install_with_the_shared_reason() -> None:
    """A bare exit code is what the reporter saw. The reason has to reach the
    desktop app, which is what Exit-InstallFailure emits."""
    body = _function_source(INSTALL_PS1, "Invoke-ManagedLlamaCppPreflight")
    assert 'Write-PathAccessDenied -Path $dir -Label "llama.cpp install"' in body
    assert "Nothing was installed." in body
    call = INSTALL_PS1.split("$llamaPreflightFailure = Invoke-ManagedLlamaCppPreflight", 1)[1]
    call = call.split("# ── Check winget ──", 1)[0]
    assert "Exit-InstallFailure $llamaPreflightFailure" in call


def test_the_preflight_cannot_be_the_thing_that_breaks_the_run() -> None:
    """It runs before anything is installed, so every failure it can produce is a
    failure it invented. Join-Path throws under "Stop" on an empty USERPROFILE,
    and a profile-less environment has to fail where it actually matters."""
    body = _function_source(INSTALL_PS1, "Invoke-ManagedLlamaCppPreflight")
    guard = 'if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) { return $null }'
    assert guard in body
    assert body.index(guard) < body.index("Get-ManagedLlamaCppDir")
    # Both probes swallow their own failures rather than terminating.
    probe = _function_source(INSTALL_PS1, "Get-LlamaCppInstallReadState")
    assert "-ErrorAction Stop" in probe and "catch" in probe
    assert "Get-PathState" in probe


def test_a_custom_studio_home_is_never_called_a_cache_we_own() -> None:
    """UNSLOTH_STUDIO_HOME points at the user's own directory, and an unreadable
    tree there cannot be proven ours, so "delete it, we reinstall it" is wrong.
    Same rule setup.ps1 applies at the same probe."""
    body = _function_source(INSTALL_PS1, "Invoke-ManagedLlamaCppPreflight")
    # The predicate that chose the directory, not a rule re-derived from the path
    # it returned: those two can disagree and only one of them is the policy.
    assert "$homeIsCustom = Test-StudioHomeIsCustom" in body
    assert "-OwnershipUnverified:$homeIsCustom" in body
    assert (
        'Exit-PathAccessDenied -Path $LlamaCppDir -Label "llama.cpp install"'
        " -OwnershipUnverified:$StudioHomeIsCustom" in SETUP_PS1
    )


def test_a_tree_the_user_pointed_at_is_never_called_a_cache_we_own() -> None:
    """--with-llama-cpp-dir may name the managed location itself: setup.ps1 supports
    that (reuse the build already there) and reports the tree -UserSupplied even
    though the path is the managed one. The preflight sees it first, so it has to
    ask the same question rather than tell the user to delete their own build."""
    body = _function_source(INSTALL_PS1, "Invoke-ManagedLlamaCppPreflight")
    assert "-UserSupplied:$userSupplied" in body
    # Both spellings of the same act: the flag, and the environment variable
    # setup.ps1 reads on its own, which install.ps1 passes through untouched.
    assert (
        "$suppliedDir = if ($WithLlamaCppDir) { $WithLlamaCppDir }"
        " else { $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR }" in body
    )
    # Resolved on both sides, which is how setup.ps1 answers the same question.
    assert (
        "(Get-CanonicalDir -Path $suppliedDir) -eq (Get-CanonicalDir -Path $dir)"
        in body
    )
    assert "$LocalIsCanonical = ($ResolvedLocal -eq $LlamaCppDir)" in SETUP_PS1
    assert (
        'Exit-PathAccessDenied -Path $ResolvedLocal'
        ' -Label "the UNSLOTH_LOCAL_LLAMA_CPP_DIR build" -UserSupplied' in SETUP_PS1
    )


def test_the_managed_path_rule_is_not_duplicated_in_the_installer() -> None:
    """~/.unsloth/llama.cpp unless the studio home is custom. The launcher used to
    carry its own copy of that comparison; the preflight must not add a third."""
    resolver = _function_source(INSTALL_PS1, "Get-ManagedLlamaCppDir")
    assert 'Join-Path $env:USERPROFILE ".unsloth\\llama.cpp"' in resolver
    assert 'Join-Path (Get-CanonicalDir -Path $StudioHome) "llama.cpp"' in resolver
    assert INSTALL_PS1.count('Join-Path $StudioHome "llama.cpp"') == 0
    assert INSTALL_PS1.count("$_llamaPath = Get-ManagedLlamaCppDir") == 1
    # The default-vs-custom question is asked once, by the predicate, and the
    # resolver branches on it rather than carrying its own comparison.
    assert "if (-not (Test-StudioHomeIsCustom)) {" in resolver
    assert "$legacyStudio" not in resolver
    # Resolve both sides before comparing, or an override spelled differently
    # from the default reads as custom and nests llama.cpp in the wrong place.
    predicate = _function_source(INSTALL_PS1, "Test-StudioHomeIsCustom")
    assert predicate.count("Get-CanonicalDir -Path") == 2
    # And canonicalizing is one helper rather than the same block at each site.
    assert "Resolve-Path -LiteralPath $Path" in _function_source(INSTALL_PS1, "Get-CanonicalDir")
    assert INSTALL_PS1.count("Resolve-Path -LiteralPath $Path") == 1


def test_both_entrypoints_resolve_and_reuse_the_same_managed_directory() -> None:
    """Path choice, ownership wording, the early gate, and phase 3.4 all consume
    one shared resolver instead of deriving parallel paths."""
    assert '$LegacyStudioHome = Join-Path $env:USERPROFILE ".unsloth\\studio"' in SETUP_PS1
    assert "$StudioHomeIsCustom = Test-StudioHomeIsCustom" in SETUP_PS1
    assert SETUP_PS1.count("$LlamaCppDir = Get-ManagedLlamaCppDir") == 1
    assert "$UnslothHome = Split-Path -Parent $LlamaCppDir" in SETUP_PS1
    for name in (
        "Get-CanonicalDir",
        "Test-StudioHomeIsCustom",
        "Get-ManagedLlamaCppDir",
    ):
        assert _normalized(_function_source(INSTALL_PS1, name)) == _normalized(
            _function_source(SETUP_PS1, name)
        )
    phase = SETUP_PS1.split("PHASE 3.4", 1)[1].split("$NeedLlamaSourceBuild", 1)[0]
    assert "resolved and preflighted before phase 1" in phase
    assert "$LlamaCppDir =" not in phase


def test_the_installer_never_repairs_permissions_by_itself() -> None:
    """takeown and icacls are printed for the user to run in an elevated shell.
    Running them from the installer would need elevation this process deliberately
    does not have, and would rewrite ACLs nobody asked it to touch."""
    # Every form that runs it: bare, chained, captured ($x = takeown ...), a
    # subexpression, or handed to Start-Process / Invoke-Expression. Quoting it
    # into a string ($x = "takeown ...") is what the guidance below does.
    invocation = re.compile(
        r"(^|[&|;=]\s*|\(\s*|Start-Process\s+|Invoke-Expression\s+)(takeown|icacls)\b"
    )
    for text, label in ((INSTALL_PS1, "install.ps1"), (SETUP_PS1, "setup.ps1")):
        for line in text.splitlines():
            code = line.split("#", 1)[0].strip()
            if "takeown" not in code and "icacls" not in code:
                continue
            assert not invocation.search(code), f"{label}: {line.strip()}"


def test_setup_sh_reports_a_denied_default_home_cache() -> None:
    """POSIX parity for the reporting, not the preflight: see the PR description.
    The ownership guard beside this only covers a custom UNSLOTH_STUDIO_HOME, so
    the default ~/.unsloth/llama.cpp reached install_llama_prebuilt.py, whose
    is_file() raises on EACCES rather than returning false."""
    block = SETUP_SH.split('substep "installing prebuilt llama.cpp..."', 1)[1]
    block = block.split("_PREBUILT_CMD=(", 1)[0]
    assert 'if _studio_dir_unsearchable "$LLAMA_CPP_DIR"; then' in block
    assert '_path_access_denied "$LLAMA_CPP_DIR" "llama.cpp install"' in block
    # After the ownership guard: that one reports a custom home as
    # owner-unverified, which is the more careful wording of the two.
    assert block.index("_assert_studio_owned_or_absent") < block.index(
        "_studio_dir_unsearchable"
    )
