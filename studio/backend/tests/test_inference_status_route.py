# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behavioral regression tests for the inference status route.

`/api/inference/status` reports llama.cpp capabilities (which execs
`llama-server --help`) and prebuilt release freshness (which reads
api.github.com). Both are slow on a cold cache, and the UI polls this route
every few seconds.

These tests stub only those two external boundaries and then assert observable
behavior: the event loop keeps running while a probe is in flight, overlapping
polls do not multiply the external work, and a client that disconnects mid-probe
does not spoil another client's response.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import routes.inference as inference_route  # noqa: E402

# Long enough that a blocking probe would be unmistakable in the loop-lag
# measurement below, short enough to keep the suite fast.
_PROBE_DELAY_SECONDS = 0.6
# The loop is polled every 10ms; anything under this bound means the probe never
# owned the loop. A blocking probe parks it for _PROBE_DELAY_SECONDS.
_MAX_ACCEPTABLE_LOOP_LAG_SECONDS = 0.2


class _FakeLlamaBackend:
    is_loaded = False


class _FakeInferenceBackend:
    active_model_name = None
    models: dict = {}
    loading_models: set = set()


def _patch_status_dependencies(monkeypatch):
    """Stub everything the route touches other than the two slow probes."""
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
    # Never inherit an in-flight probe from an earlier test.
    monkeypatch.setattr(inference_route, "_LLAMA_STATUS_PROBE_FUTURE", None)


def _patch_slow_probes(monkeypatch, *, delay: float):
    """Make the capability exec and the GitHub read slow, and count them.

    These are the real external boundaries the route calls, not the coalescing
    machinery under test.
    """
    from utils import llama_cpp_freshness

    capability_calls = []
    freshness_calls = []

    def _find_binary(_cls):
        return "/nonexistent/llama-server"

    def _probe_capabilities(_cls, _binary):
        capability_calls.append(_binary)
        time.sleep(delay)
        return {"found": True, "supports_mtp": True}

    def _check_freshness(_binary):
        freshness_calls.append(_binary)
        return {"stale": True, "installed_tag": "b1", "latest_tag": "b2"}

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
    return capability_calls, freshness_calls


def _assert_probe_reported(response):
    assert response.llama_cpp_supports_mtp is True
    assert response.llama_cpp_prebuilt_stale is True
    assert response.llama_cpp_installed_tag == "b1"
    assert response.llama_cpp_latest_tag == "b2"


def test_status_request_does_not_stall_the_event_loop(monkeypatch):
    """A slow probe must not park the loop; SSE streaming shares it."""
    _patch_status_dependencies(monkeypatch)
    _patch_slow_probes(monkeypatch, delay = _PROBE_DELAY_SECONDS)

    async def _run():
        lags = []
        stop = asyncio.Event()

        async def _measure_loop_lag():
            while not stop.is_set():
                before = time.monotonic()
                await asyncio.sleep(0.01)
                lags.append(time.monotonic() - before - 0.01)

        ticker = asyncio.create_task(_measure_loop_lag())
        # Let the ticker take a few turns first, so a blocking probe shows up as
        # a long lag rather than as an empty measurement.
        await asyncio.sleep(0.05)
        started = time.monotonic()
        response = await inference_route.get_status(current_subject = "test")
        elapsed = time.monotonic() - started
        stop.set()
        await ticker
        return response, elapsed, lags

    response, elapsed, lags = asyncio.run(_run())

    # The probe really was slow, so the measurement window covered it.
    assert elapsed >= _PROBE_DELAY_SECONDS
    assert lags, "the loop never got a turn"
    assert max(lags) < _MAX_ACCEPTABLE_LOOP_LAG_SECONDS, (
        f"event loop stalled for {max(lags):.3f}s during the status probe"
    )
    _assert_probe_reported(response)


def test_overlapping_status_polls_probe_once(monkeypatch):
    """Concurrent pollers share one probe instead of each exec'ing and fetching."""
    _patch_status_dependencies(monkeypatch)
    capability_calls, freshness_calls = _patch_slow_probes(
        monkeypatch, delay = _PROBE_DELAY_SECONDS
    )

    async def _run():
        return await asyncio.gather(
            *(
                inference_route.get_status(current_subject = f"client-{index}")
                for index in range(4)
            )
        )

    responses = asyncio.run(_run())

    assert len(capability_calls) == 1
    assert len(freshness_calls) == 1
    for response in responses:
        _assert_probe_reported(response)


def test_status_probe_reruns_after_the_previous_one_finishes(monkeypatch):
    """Coalescing must not pin a result; a later poll re-probes."""
    _patch_status_dependencies(monkeypatch)
    capability_calls, _freshness_calls = _patch_slow_probes(monkeypatch, delay = 0.0)

    async def _run():
        first = await inference_route.get_status(current_subject = "test")
        second = await inference_route.get_status(current_subject = "test")
        return first, second

    first, second = asyncio.run(_run())

    assert len(capability_calls) == 2
    _assert_probe_reported(first)
    _assert_probe_reported(second)


def test_disconnected_client_does_not_spoil_a_concurrent_status_request(monkeypatch):
    """One waiter giving up must not cancel the probe the others are sharing.

    The probe pool has a single worker, so a probe submitted while that worker is
    busy sits cancellable in the queue. That is the window a disconnecting client
    would otherwise cancel out from under everyone else.
    """
    _patch_status_dependencies(monkeypatch)
    _patch_slow_probes(monkeypatch, delay = 0.0)

    import threading

    occupied = threading.Event()
    release_worker = threading.Event()

    def _occupy_worker():
        occupied.set()
        # Block the only pool worker so the status probe stays queued.
        release_worker.wait(timeout = 5)

    async def _run():
        inference_route._LLAMA_STATUS_PROBE_EXECUTOR.submit(_occupy_worker)
        assert occupied.wait(timeout = 5)

        abandoned = asyncio.create_task(
            inference_route.get_status(current_subject = "disconnecting")
        )
        waiting = asyncio.create_task(
            inference_route.get_status(current_subject = "still-here")
        )
        # Let both reach the shared probe before either can complete.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        abandoned.cancel()
        try:
            await abandoned
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("the abandoned status request was not cancelled")

        release_worker.set()
        return await asyncio.wait_for(waiting, timeout = 5)

    response = asyncio.run(_run())

    _assert_probe_reported(response)
