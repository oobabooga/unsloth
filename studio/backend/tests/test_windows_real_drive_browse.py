# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real-Windows verification of the drive-root folder browser (PR #7082).

Unlike test_windows_external_drive_paths.py, nothing here is mocked: these
tests run only on a real Windows host and exercise the machine's actual
drives, the real GetLogicalDrives bitmask, real 8.3 short names, and the
real os.path.realpath treatment of \\?\ device-namespace paths, none of
which can be validated off Windows.
"""

from __future__ import annotations

import os
import platform
import re
import sys
import types
from pathlib import Path

import pytest

# Keep runnable in lightweight environments lacking optional logging deps
# (mirrors test_browse_folders_route.py).
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows", reason = "requires a real Windows host"
)

from utils.paths.external_media import (  # noqa: E402
    _active_windows_drive_bitmask,
    is_local_filesystem_root,
    windows_drive_roots,
)


def _system_drive_root() -> str:
    return os.environ.get("SystemDrive", "C:") + "\\"


def _system_root() -> str:
    return os.environ.get("SystemRoot", r"C:\Windows")


def test_bitmask_reports_the_system_drive():
    mask = _active_windows_drive_bitmask()
    assert mask != 0
    letter = _system_drive_root()[0].upper()
    assert mask & (1 << (ord(letter) - ord("A")))


def test_windows_drive_roots_lists_real_drives():
    roots = windows_drive_roots()
    assert Path(_system_drive_root()) in roots
    for root in roots:
        assert re.fullmatch(r"[A-Z]:\\", str(root))
        assert root.is_dir()


def test_is_local_filesystem_root_on_real_paths():
    assert is_local_filesystem_root(_system_drive_root())
    assert not is_local_filesystem_root(_system_root())
    assert not is_local_filesystem_root(str(Path.home()))
    # Device-namespace form of the system drive is still a local root.
    assert is_local_filesystem_root("\\\\?\\" + _system_drive_root())


def test_denylist_matches_real_environment():
    from storage.studio_db import is_denied_system_path

    assert is_denied_system_path(_system_root())
    assert is_denied_system_path(os.path.join(_system_root(), "System32"))
    assert not is_denied_system_path(_system_drive_root())
    assert not is_denied_system_path(str(Path.home()))


def test_realpath_expands_8dot3_short_names_before_denylist():
    """C:\\PROGRA~1 must resolve to C:\\Program Files, so the resolve-then-check
    pattern in the browse listing cannot be bypassed by an 8.3 alias."""
    from storage.studio_db import is_denied_system_path

    short = os.path.join(_system_drive_root(), "PROGRA~1")
    if not os.path.exists(short):
        pytest.skip("8.3 short names disabled on this volume")
    real = os.path.realpath(short)
    expected = os.environ.get("ProgramFiles", r"C:\Program Files")
    assert os.path.normcase(real) == os.path.normcase(expected)
    assert is_denied_system_path(real)


def test_realpath_strips_device_prefix_before_denylist():
    """\\\\?\\C:\\Windows must resolve to the plain form the denylist keys on."""
    from storage.studio_db import is_denied_system_path

    real = os.path.realpath("\\\\?\\" + _system_root())
    assert not real.startswith("\\\\?\\")
    assert is_denied_system_path(real)


def test_allowlist_containment_on_real_drives():
    import routes.models as models_route
    from hub.services.models import folder_browser

    sys_root = Path(_system_drive_root())
    users = Path(_system_drive_root()) / "Users"
    for contains in (
        models_route._is_path_inside_allowlist,
        folder_browser._is_path_inside_allowlist,
    ):
        assert contains(sys_root, [sys_root])
        assert contains(users, [sys_root])
        # A different drive letter is never contained by the system drive.
        other = "Z:\\" if _system_drive_root()[0].upper() != "Z" else "Y:\\"
        assert not contains(Path(other) / "models", [sys_root])


def test_add_scan_folder_rejects_real_drive_root():
    from storage.studio_db import add_scan_folder as legacy_add
    from hub.storage.scan_folders import add_scan_folder as hub_add

    for add in (legacy_add, hub_add):
        with pytest.raises(ValueError):
            add(_system_drive_root())


def test_browse_route_end_to_end_on_real_windows():
    """Drive the real /api/models/browse-folders endpoint against the real C: drive."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import routes.models as models_route
    from auth.authentication import get_current_subject

    app = FastAPI()
    app.include_router(models_route.router, prefix = "/api/models")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    client = TestClient(app)

    # The drive root browses, and denied system dirs are hidden from its listing.
    resp = client.get("/api/models/browse-folders", params = {"path": _system_drive_root()})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    names = {entry["name"] for entry in data["entries"]}
    assert "Windows" not in names
    assert "Program Files" not in names
    # The drive root itself is offered as a suggestion chip.
    suggestions = {os.path.normcase(s) for s in data.get("suggestions", [])}
    assert os.path.normcase(_system_drive_root()) in suggestions

    # Descending into a denied system dir is refused.
    resp = client.get("/api/models/browse-folders", params = {"path": _system_root()})
    assert resp.status_code == 403, resp.text

    # An ordinary drive-root descendant browses fine.
    users = os.path.join(_system_drive_root(), "Users")
    resp = client.get("/api/models/browse-folders", params = {"path": users})
    assert resp.status_code == 200, resp.text
    assert os.path.normcase(resp.json()["current"]) == os.path.normcase(
        os.path.realpath(users)
    )

    # Auth still enforced with no override.
    app.dependency_overrides.clear()
    resp = client.get("/api/models/browse-folders", params = {"path": users})
    assert resp.status_code in (401, 403)
