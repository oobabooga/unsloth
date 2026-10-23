# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The speech-to-text sidecars keep their servers' output and quote it when a server does not start."""

from __future__ import annotations

import subprocess
import sys
import types as _types
from pathlib import Path

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

from core.inference import sidecar_log  # noqa: E402

# A llama-server chat-template abort, as llama-server prints it.
_TEMPLATE_TAIL = (
    "0.06.479.180 I srv init: idle slots will be saved to prompt cache and cleared upon starting a new task\n"
    "0.06.479.282 E common_chat_templates_init: error: lexer: unknown escape character \\x\n"
    '...- if t is string -%}{{- t | replace("\\x00", "") | repl...\n'
    "0.06.479.283 E common_chat_templates_init: failed to initialize chat template\n"
    "0.06.479.283 E common_chat_templates_init: please consider disabling jinja via --no-jinja, or using another chat template\n"
    "0.06.479.295 E srv init: init: chat template parsing error: std::exception\n"
    "0.06.479.295 E srv init: init: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n"
    "0.06.479.298 I srv operator(): operator(): cleaning up before exit...\n"
    "0.06.479.538 E srv llama_server: exiting due to model loading error\n"
)


class TestSidecarLog:
    def test_a_child_writes_its_output_and_the_tail_names_status_and_path(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
        path, handle = sidecar_log.open_child_log("whisper-server", 4242)
        assert path is not None and path.parent == tmp_path / "logs" / "whisper-server"
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('loading model'); sys.stderr.write('fatal: bad header\\n'); sys.exit(3)",
                ],
                stdout = handle,
                stderr = subprocess.STDOUT,
            )
        finally:
            handle.close()
        proc.wait(timeout = 30)
        details = sidecar_log.exit_details(proc.returncode, path)
        assert "status 3" in details
        assert "fatal: bad header" in details
        assert f"Full log: {path}" in details

    def test_an_error_before_a_long_usage_dump_is_kept(self, tmp_path):
        """whisper-server prints the rejected flag, then ~100 lines of usage: the tail alone lost it."""
        path = tmp_path / "w.log"
        path.write_text(
            "load_backend: loaded CUDA backend\n"
            "error: unknown argument: --definitely-not-a-flag\n"
            + "".join(f"  -x{i}, --option-{i}   [0] some option\n" for i in range(120))
        )
        details = sidecar_log.exit_details(0, path)
        assert "Errors reported:\nerror: unknown argument: --definitely-not-a-flag" in details
        assert "--option-119" in details

    def test_the_directory_is_capped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
        directory = tmp_path / "logs" / "stt-llama-server"
        directory.mkdir(parents = True)
        for i in range(30):
            (directory / f"stt-llama-server-{i}-port-1.log").write_text("x")
        path, handle = sidecar_log.open_child_log("stt-llama-server", 1)
        handle.close()
        assert path.exists()
        assert len(list(directory.glob("stt-llama-server-*.log"))) == 20

    def test_an_unwritable_root_falls_back_to_no_log(self, tmp_path, monkeypatch):
        blocker = tmp_path / "file"
        blocker.write_text("")
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(blocker))
        assert sidecar_log.open_child_log("whisper-server", 1) == (None, None)
        assert sidecar_log.exit_details(None, None) == ""

    def test_the_tail_is_bounded(self, tmp_path):
        path = tmp_path / "big.log"
        path.write_text("a" * 100000 + "END")
        tail = sidecar_log.read_tail(path)
        assert tail.endswith("END")
        assert len(tail) <= 1500


class TestMtmdSidecarStartFailure:
    def test_a_template_abort_reads_like_the_chat_load_does(self, tmp_path):
        from core.inference.stt_mtmd_sidecar import MtmdSttSidecar

        log = tmp_path / "stt.log"
        log.write_text(_TEMPLATE_TAIL)
        msg = MtmdSttSidecar._start_failure_message(
            "unsloth/Qwen3-ASR-0.6B-GGUF", 1, log, None, None
        )
        assert "status 1" in msg
        assert "chat template" in msg
        assert str(log) in msg

    def test_a_server_still_starting_at_the_deadline_is_not_called_an_oom_kill(self, tmp_path):
        from core.inference.stt_mtmd_sidecar import MtmdSttSidecar

        log = tmp_path / "stt.log"
        log.write_text("loading tensors\n")
        msg = MtmdSttSidecar._start_failure_message("m", None, log, None, None)
        assert "did not become ready" in msg
        assert "signal 9" not in msg
        assert "loading tensors" in msg

    def test_the_api_key_is_redacted(self, tmp_path):
        from core.inference.stt_mtmd_sidecar import MtmdSttSidecar

        key = "k" * 43
        log = tmp_path / "stt.log"
        log.write_text(f"main: api key {key}\nsomething broke\n")
        msg = MtmdSttSidecar._start_failure_message("m", 1, log, None, key)
        assert key not in msg
