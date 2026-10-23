# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Start failures that used to read as "invalid GGUF or not enough memory" (RD-784, RD-785, RD-788).

A chat-template parse abort and an architecture the installed llama.cpp predates are both a
valid file with enough memory; the message has to say what failed and which build is installed.
Also covers the STT sidecars, which discarded their servers' output entirely.
"""

from __future__ import annotations

import json
import subprocess
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

from core.inference import sidecar_log  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from utils import llama_cpp_freshness  # noqa: E402

_classify = LlamaCppBackend._classify_llama_start_failure

# The tail the RD-785 reporter pasted (Flint 40B, bundled build before b11798), timestamps kept.
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


def _managed_binary(tmp_path: Path, tag: str) -> str:
    """A llama-server path under a prebuilt install carrying a marker with ``tag``."""
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents = True)
    binary = root / "build" / "bin" / "llama-server"
    binary.write_text("")
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps({"release_tag": tag, "tag": "b1"}))
    llama_cpp_freshness._marker_cache.clear()
    return str(binary)


class TestChatTemplateParseFailure:
    def test_it_names_the_template_not_the_file_or_memory(self):
        msg = _classify(_TEMPLATE_TAIL, "/models/flint.gguf", "local/flint", 1)
        assert "chat template" in msg
        assert "not a memory problem" in msg
        assert "Check that the GGUF file is valid" not in msg
        assert "enough memory" not in msg

    def test_it_quotes_the_lexer_error_and_every_remedy(self):
        msg = _classify(_TEMPLATE_TAIL, "/models/flint.gguf", "local/flint", 1)
        assert "lexer: unknown escape character" in msg
        assert "Chat Template" in msg
        assert "load settings" in msg
        assert "Update" in msg
        # Measured on b11798: --no-jinja alone still aborts on the embedded template, so it is
        # named only as a non-remedy.
        assert "--no-jinja alone does not help" in msg

    def test_the_evidence_and_log_path_are_kept(self, tmp_path):
        log = tmp_path / "llama.log"
        msg = _classify(_TEMPLATE_TAIL, "/models/flint.gguf", "local/flint", 1, None, log)
        assert "llama-server output:" in msg
        assert str(log) in msg

    def test_the_srv_init_line_alone_is_enough(self):
        """Builds that skip the common_chat_templates_init lines still print this one."""
        out = "E srv init: init: chat template parsing error: std::exception\n"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "chat template" in msg
        assert "enough memory" not in msg

    def test_it_names_the_installed_build(self, tmp_path):
        binary = _managed_binary(tmp_path, "b11700-mix-abc1234")
        msg = _classify(_TEMPLATE_TAIL, "/models/flint.gguf", "local/flint", 1, binary)
        assert "b11700-mix-abc1234" in msg

    def test_an_unrelated_failure_still_gets_the_fallback(self):
        msg = _classify("something else entirely\n", "/models/x.gguf", "local/x", 1)
        assert "Check that the GGUF file is valid" in msg


class TestUnknownArchitectureNamesTheBuild:
    _OUT = (
        "llama_model_load: error loading model: unknown model architecture: 'deepseek41-dspark'\n"
    )

    def test_the_installed_tag_is_named(self, tmp_path):
        binary = _managed_binary(tmp_path, "b11650-mix-0123abc")
        msg = _classify(self._OUT, "/models/x.gguf", "local/x", 1, binary)
        assert "deepseek41-dspark" in msg
        assert "(b11650-mix-0123abc)" in msg
        assert "updating llama.cpp" in msg.lower()

    def test_no_marker_omits_the_build_quietly(self, tmp_path):
        binary = tmp_path / "bin" / "llama-server"
        binary.parent.mkdir()
        binary.write_text("")
        llama_cpp_freshness._marker_cache.clear()
        msg = _classify(self._OUT, "/models/x.gguf", "local/x", 1, str(binary))
        assert "The installed llama.cpp does not recognise" in msg

    @pytest.mark.parametrize("tag", ["", "x" * 80, "b1 rm -rf", "b1\nnext line", 12345])
    def test_an_odd_marker_value_is_not_quoted(self, tmp_path, tag):
        root = tmp_path / "llama.cpp"
        (root / "bin").mkdir(parents = True)
        binary = root / "bin" / "llama-server"
        binary.write_text("")
        (root / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps({"release_tag": tag}))
        llama_cpp_freshness._marker_cache.clear()
        assert LlamaCppBackend._installed_build_label(str(binary)) is None

    def test_no_binary_means_no_label(self):
        assert LlamaCppBackend._installed_build_label(None) is None


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
