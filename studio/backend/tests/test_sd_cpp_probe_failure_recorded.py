# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A stable-diffusion.cpp probe that fails keeps its exit status and stderr for the error the user sees."""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger


@pytest.mark.skipif(sys.platform == "win32", reason = "uses /bin/sh scripts")
class TestSdCppProbeFailuresAreRecorded:
    def test_version_keeps_the_exit_status_and_stderr(self, tmp_path):
        from core.inference.sd_cpp_engine import SdCppEngine

        script = tmp_path / "sd-cli"
        script.write_text(
            "#!/bin/sh\necho 'error while loading shared libraries: libgomp.so.1' >&2\nexit 127\n"
        )
        script.chmod(0o755)
        engine = SdCppEngine(str(script))
        assert engine.version() is None
        assert "status 127" in engine.last_failure
        assert "libgomp.so.1" in engine.last_failure

    def test_a_missing_binary_is_named(self, tmp_path):
        from core.inference.sd_cpp_engine import SdCppEngine

        engine = SdCppEngine(str(tmp_path / "absent"))
        assert engine.version() is None
        assert "no stable-diffusion.cpp binary" in engine.last_failure

    def test_a_working_binary_clears_it(self, tmp_path):
        from core.inference.sd_cpp_engine import SdCppEngine

        script = tmp_path / "sd-cli"
        script.write_text("#!/bin/sh\necho 'stable-diffusion.cpp version master-123'\n")
        script.chmod(0o755)
        engine = SdCppEngine(str(script))
        assert engine.version() == "stable-diffusion.cpp version master-123"
        assert engine.last_failure is None

    def test_the_probe_logs_once_per_outcome(self, tmp_path, monkeypatch):
        from core.inference import sd_cpp_backend

        script = tmp_path / "sd-cli"
        script.write_text("#!/bin/sh\necho 'unknown option --list-devices' >&2\nexit 1\n")
        script.chmod(0o755)
        seen = []
        monkeypatch.setattr(sd_cpp_backend.logger, "warning", lambda *a, **k: seen.append(a))
        sd_cpp_backend._LOGGED_PROBE_FAILURES.clear()
        for _ in range(3):
            assert sd_cpp_backend._sd_cpp_probe_output(str(script), "--list-devices") is None
        assert len(seen) == 1
        assert "unknown option --list-devices" in " ".join(str(x) for x in seen[0])
