# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Start-failure messages for a chat-template parse error or an unknown architecture name the
cause and the installed llama.cpp build, not the file or memory."""

from __future__ import annotations

import json
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

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from utils import llama_cpp_freshness  # noqa: E402

_classify = LlamaCppBackend._classify_llama_start_failure

# A real llama-server tail from a build older than b11798, timestamps kept.
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
