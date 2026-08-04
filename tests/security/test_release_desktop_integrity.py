"""Checks that one desktop version tag can only ever serve one set of binaries."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-desktop.yml"

RELEASE_TAG = "desktop-v0.1.50-beta"
SOURCE_SHA = "1f02275b86f0e0d3a5b1c9f2a4d6e8b0c2a4e6f8"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _steps(workflow, job):
    return workflow["jobs"][job]["steps"]


def _step(workflow, job, name):
    return next(step for step in _steps(workflow, job) if step.get("name") == name)


def _step_names(workflow, job):
    return [step.get("name") for step in _steps(workflow, job)]


def _write_fake_gh(path: Path):
    """Stand in for the gh CLI, recording argv and answering from the environment."""
    path.write_text(
        """#!/bin/sh
set -eu
printf 'gh %s\\n' "$*" >> "$COMMAND_LOG"
if [ "$1" = "api" ]; then
  case "$2" in
    */git/ref/tags/*) exit "$TAG_EXISTS_STATUS" ;;
  esac
  exit 0
fi
if [ "$1 $2" = "release view" ]; then
  exit "$RELEASE_EXISTS_STATUS"
fi
if [ "$1 $2" = "release create" ]; then
  prev=""
  for arg in "$@"; do
    if [ "$prev" = "--notes-file" ]; then
      cp "$arg" "$NOTES_CAPTURE"
    fi
    prev="$arg"
  done
fi
exit 0
""",
        encoding = "utf-8",
    )
    path.chmod(0o755)


def _run_step(
    workflow,
    job: str,
    name: str,
    tmp_path: Path,
    *,
    tag_exists: bool = False,
    release_exists: bool = False,
    extra_env: dict[str, str] | None = None,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok = True)
    _write_fake_gh(fake_bin / "gh")
    log = tmp_path / "commands.log"
    log.write_text("", encoding = "utf-8")

    env = os.environ.copy()
    env.update(
        {
            "COMMAND_LOG": str(log),
            "DESKTOP_RELEASE_TAG": RELEASE_TAG,
            "GH_REPO": "unslothai/unsloth",
            "GH_TOKEN": "masked-token",
            "NOTES_CAPTURE": str(tmp_path / "release-body.md"),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RELEASE_EXISTS_STATUS": "0" if release_exists else "1",
            "RUNNER_TEMP": str(tmp_path),
            "TAG_EXISTS_STATUS": "0" if tag_exists else "1",
        }
    )
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", "-c", _step(workflow, job, name)["run"]],
        cwd = tmp_path,
        env = env,
        text = True,
        capture_output = True,
        check = False,
    )
    return result, log.read_text(encoding = "utf-8").splitlines()


def _stage_assets(tmp_path: Path) -> dict[str, str]:
    """Write stand-in release assets and return their expected digests."""
    asset_dir = tmp_path / "desktop-release-assets"
    asset_dir.mkdir(exist_ok = True)
    digests = {}
    for name, payload in (
        ("Unsloth-Desktop-0_1_50_beta-MacOS.dmg", b"disk image"),
        ("Unsloth-Desktop-0_1_50_beta-Windows.exe", b"installer"),
        ("Unsloth-Desktop-0_1_50_beta-Linux.deb", b"package"),
    ):
        (asset_dir / name).write_bytes(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
    return digests


def _run_create_release(workflow, tmp_path: Path, **kwargs):
    _stage_assets(tmp_path)
    env = {
        "DESKTOP_PRERELEASE": "true",
        "DESKTOP_RELEASE_NOTES": workflow["env"]["DESKTOP_RELEASE_NOTES"],
        "GITHUB_SHA": SOURCE_SHA,
        "RELEASE_DRAFT": "true",
        "STUDIO_VERSION": "v0.1.50-beta",
    }
    env.update(kwargs.pop("extra_env", None) or {})
    return _run_step(
        workflow, "publish-release", "Create versioned release", tmp_path, extra_env = env, **kwargs
    )


def _upload_commands(workflow):
    commands = []
    for step in _steps(workflow, "publish-release"):
        for line in step.get("run", "").splitlines():
            stripped = line.strip()
            if stripped.startswith("gh release upload"):
                commands.append(stripped)
    return commands


def test_an_existing_tag_or_release_fails_before_any_build_work(tmp_path):
    workflow = _workflow()
    names = _step_names(workflow, "prepare-version")
    # The point of guarding here rather than only at publish time is that the
    # three platform builds and the notarization round trip never start.
    assert names.index("Guard against republishing an existing version") < names.index(
        "Verify PyPI package and Unsloth stamp"
    )
    assert workflow["jobs"]["build"]["needs"] == "prepare-version"

    for case in ({"tag_exists": True}, {"release_exists": True}):
        case_dir = tmp_path / "-".join(case)
        case_dir.mkdir()
        result, _ = _run_step(
            workflow,
            "prepare-version",
            "Guard against republishing an existing version",
            case_dir,
            **case,
        )
        assert result.returncode == 1, case
        assert RELEASE_TAG in result.stderr


def test_an_unused_version_passes_the_guard(tmp_path):
    result, _ = _run_step(
        _workflow(),
        "prepare-version",
        "Guard against republishing an existing version",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_publish_refuses_to_reuse_an_existing_release(tmp_path):
    workflow = _workflow()
    # prepare-version runs on the contents:read token, which does not list
    # drafts, so the publish job has to repeat the check for the draft case.
    for case in ({"tag_exists": True}, {"release_exists": True}):
        case_dir = tmp_path / "-".join(case)
        case_dir.mkdir()
        result, commands = _run_create_release(workflow, case_dir, **case)
        assert result.returncode == 1, case
        assert "Refusing to republish" in result.stderr
        assert not [line for line in commands if line.startswith("gh release create")]


def test_release_body_records_the_source_commit_and_asset_digests(tmp_path):
    workflow = _workflow()
    digests = _stage_assets(tmp_path)
    result, commands = _run_create_release(workflow, tmp_path)
    assert result.returncode == 0, result.stderr

    create = next(line for line in commands if line.startswith("gh release create"))
    assert RELEASE_TAG in create
    assert f"--target {SOURCE_SHA}" in create

    body = (tmp_path / "release-body.md").read_text(encoding = "utf-8")
    assert SOURCE_SHA in body
    for name, digest in digests.items():
        assert f"{digest}  {name}" in body


def test_updater_notes_stay_free_of_the_provenance_block(tmp_path):
    # latest.json feeds the in-app update popup from this file, so the digest
    # dump belongs to the release body only.
    workflow = _workflow()
    result, _ = _run_create_release(workflow, tmp_path)
    assert result.returncode == 0, result.stderr

    notes = (tmp_path / "desktop-release-notes.md").read_text(encoding = "utf-8")
    assert "Build provenance" not in notes
    assert "Desktop app for Unsloth." in notes

    metadata_step = _step(
        workflow, "publish-release", "Generate and publish versioned updater metadata"
    )
    assert "'desktop-release-notes.md'" in metadata_step["run"]


def test_versioned_uploads_never_clobber_but_the_channel_pointer_does():
    workflow = _workflow()
    uploads = _upload_commands(workflow)
    versioned = [line for line in uploads if "$DESKTOP_RELEASE_TAG" in line]
    channel = [line for line in uploads if "desktop-latest" in line]
    assert len(versioned) == 2, uploads
    assert len(channel) == 1, uploads

    for line in versioned:
        assert "--clobber" not in line, line
    # The updater channel is the one artifact that is meant to move.
    assert "--clobber" in channel[0]
