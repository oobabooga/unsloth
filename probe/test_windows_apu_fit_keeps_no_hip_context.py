# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Windows ROCm APU must size a GGUF without calling ``mem_get_info``.

Windows has no amd-smi, so ``_get_gpu_memory`` always falls through to its torch
branch there, and on a unified-memory APU the amd-smi branch defers even where the
tool exists. ``mem_get_info`` is the last call on that branch that initializes HIP on
the device: ``is_available``, ``device_count`` and ``get_device_properties`` have all
run during hardware detection, so on a backend that has been serving for minutes it
is the one native call still being made for the first time. On a Strix Halo (gfx1151)
with AMD's Windows ROCm wheels it has ended the backend with exit status 3, the MSVC
``abort()`` status -- no exception to catch, the process simply gone, and the desktop
app reporting "Server stopped unexpectedly" the moment the model dropdown asked what
would fit.

It is also the reading that decides nothing there. Windows HIP reports free == total
on a shared pool (#7072), so it describes the pool and not what is left of it, and the
caller already takes the smaller of it and available system RAM -- which is what a
unified-memory APU actually loads into, and the only one of the two that moves when
something else on the host takes memory.

torch, ROCm detection and the platform are all faked: this repository has no AMD GPU,
no ROCm CI and no Windows runner, so none of this is a hardware validation.
"""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.llama_cpp import _IGPU_HOST_RESERVE_MIB, LlamaCppBackend
from utils import hardware
from utils.hardware import hardware as hardware_module

_VISIBLE_DEVICE_MASKS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


class _Host:
    """A single-GPU ROCm host whose torch records every ``mem_get_info`` call."""

    def __init__(self, arch, *, total_mib, free_mib):
        self.mem_get_info_calls: list[int] = []
        self.module = types.ModuleType("torch")
        self.module.version = types.SimpleNamespace(hip = "6.2.0")
        self.module.cuda = types.SimpleNamespace(
            is_available = lambda: True,
            device_count = lambda: 1,
            get_device_properties = lambda i: types.SimpleNamespace(
                gcnArchName = arch, total_memory = total_mib * 1024 * 1024
            ),
            mem_get_info = self._mem_get_info,
            memory_reserved = lambda *a, **k: 0,
        )
        self._free_bytes = free_mib * 1024 * 1024
        self._total_bytes = total_mib * 1024 * 1024

    def _mem_get_info(self, ordinal = 0):
        self.mem_get_info_calls.append(ordinal)
        return self._free_bytes, self._total_bytes


def _probe(monkeypatch, host, *, windows, available_mib):
    """``_get_gpu_memory`` on a host with neither nvidia-smi nor amd-smi."""
    monkeypatch.setitem(sys.modules, "torch", host.module)
    for mask in _VISIBLE_DEVICE_MASKS:
        monkeypatch.delenv(mask, raising = False)
    # Two bindings for one predicate: the package re-exports it by value, and
    # trusted_mem_get_info reads the one in its own module. Both are set so the
    # driver path and the shortcut cannot disagree about the platform.
    for module in (hardware, hardware_module):
        monkeypatch.setattr(
            module, "rocm_windows_free_is_untrusted", staticmethod(lambda: windows)
        )
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: available_mib)
    )
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "llama-server")
    )
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no nvidia-smi")),
    )
    return LlamaCppBackend._get_gpu_memory()


class TestAWindowsApuNeverInitializesTheDevice:
    @pytest.mark.parametrize("arch", ["gfx1150", "gfx1151", "gfx1152"])
    def test_the_budget_comes_from_system_ram_alone(self, monkeypatch, arch):
        host = _Host(arch, total_mib = 65_536, free_mib = 65_536)
        assert _probe(monkeypatch, host, windows = True, available_mib = 49_176) == [
            (0, 49_176 - _IGPU_HOST_RESERVE_MIB, 0)
        ]
        assert host.mem_get_info_calls == []

    def test_it_publishes_what_the_driver_reading_published(self, monkeypatch):
        """Same host, same answer: only the call disappears. Windows HIP reports
        free == total, so the caller's min() already resolved to system RAM."""
        driver_host = _Host("gfx1151", total_mib = 65_536, free_mib = 65_536)
        driver_answer = _probe(
            monkeypatch, driver_host, windows = False, available_mib = 49_176
        )
        assert driver_host.mem_get_info_calls == [0]

        shortcut_host = _Host("gfx1151", total_mib = 65_536, free_mib = 65_536)
        assert (
            _probe(monkeypatch, shortcut_host, windows = True, available_mib = 49_176)
            == driver_answer
        )
        assert shortcut_host.mem_get_info_calls == []


class TestEverythingElseStillAsksTheDriver:
    def test_a_windows_discrete_card_needs_the_drivers_total(self, monkeypatch):
        """System RAM says nothing about a discrete card's capacity, and this probe
        publishes that total for the fit."""
        host = _Host("gfx1100", total_mib = 24_576, free_mib = 20_480)
        assert _probe(monkeypatch, host, windows = True, available_mib = 12_000) == [
            (0, 20_480, 24_576)
        ]
        assert host.mem_get_info_calls == [0]

    def test_an_apu_off_windows_lets_a_smaller_driver_free_win(self, monkeypatch):
        """Off Windows free tracks real residency, so it is a floor the host ceiling
        must not raise."""
        host = _Host("gfx1151", total_mib = 65_536, free_mib = 8_000)
        assert _probe(monkeypatch, host, windows = False, available_mib = 64_000) == [
            (0, 8_000 - _IGPU_HOST_RESERVE_MIB, 0)
        ]
        assert host.mem_get_info_calls == [0]

    def test_unreadable_system_ram_leaves_no_ceiling_to_publish(self, monkeypatch):
        host = _Host("gfx1151", total_mib = 65_536, free_mib = 65_536)
        assert _probe(monkeypatch, host, windows = True, available_mib = None) == [
            (0, 65_536 - _IGPU_HOST_RESERVE_MIB, 0)
        ]
        assert host.mem_get_info_calls == [0]
