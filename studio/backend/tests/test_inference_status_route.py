# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the inference status route."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import routes.inference as inference_route  # noqa: E402


class _FakeLlamaBackend:
    is_loaded = False


class _FakeInferenceBackend:
    active_model_name = None
    models: dict = {}
    loading_models: set = set()


def _patch_status_dependencies(monkeypatch):
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", _FakeLlamaBackend)
    monkeypatch.setattr(inference_route, "get_inference_backend", _FakeInferenceBackend)
    monkeypatch.setattr(
        inference_route,
        "_detect_safetensors_features",
        lambda *_args: {
            "supports_reasoning": False,
            "reasoning_style": "enable_thinking",
            "reasoning_effort_levels": [],
            "reasoning_always_on": False,
            "supports_preserve_thinking": False,
            "supports_tools": False,
        },
    )


def test_status_probes_run_off_event_loop(monkeypatch):
    from utils import llama_cpp_freshness

    _patch_status_dependencies(monkeypatch)
    probe_threads = []

    def _find_binary(cls):
        probe_threads.append(threading.current_thread())
        return "/tmp/llama-server"

    def _probe_capabilities(cls, _binary):
        probe_threads.append(threading.current_thread())
        return {"found": True, "supports_mtp": True}

    def _check_freshness(_binary):
        probe_threads.append(threading.current_thread())
        return {
            "stale": True,
            "installed_tag": "b1",
            "latest_tag": "b2",
        }

    monkeypatch.setattr(
        _FakeLlamaBackend,
        "_find_llama_server_binary",
        classmethod(_find_binary),
        raising = False,
    )
    monkeypatch.setattr(
        _FakeLlamaBackend,
        "probe_server_capabilities",
        classmethod(_probe_capabilities),
        raising = False,
    )
    monkeypatch.setattr(llama_cpp_freshness, "check_prebuilt_freshness", _check_freshness)

    async def _run():
        event_loop_thread = threading.current_thread()
        response = await inference_route.get_status(current_subject = "test")
        return event_loop_thread, response

    event_loop_thread, response = asyncio.run(_run())

    assert len(probe_threads) == 3
    assert all(thread is not event_loop_thread for thread in probe_threads)
    assert response.llama_cpp_supports_mtp is True
    assert response.llama_cpp_prebuilt_stale is True
    assert response.llama_cpp_installed_tag == "b1"
    assert response.llama_cpp_latest_tag == "b2"


def test_concurrent_status_probes_use_one_worker(monkeypatch):
    _patch_status_dependencies(monkeypatch)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def _probe(_backend):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return True, {}

    monkeypatch.setattr(inference_route, "_probe_llama_cpp_status", _probe)

    async def _run():
        await asyncio.gather(
            inference_route.get_status(current_subject = "test"),
            inference_route.get_status(current_subject = "test"),
        )

    asyncio.run(_run())

    assert max_active == 1
