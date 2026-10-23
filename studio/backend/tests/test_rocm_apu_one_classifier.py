# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training, llama.cpp, the inventory and diffusion agree on which ROCm devices are APUs."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.rocm_apu import APU_ARCHES, classify_rocm_unified_memory  # noqa: E402

CASES = [
    (SimpleNamespace(gcnArchName = "gfx1151"), True),
    (SimpleNamespace(gcnArchName = "gfx1151:sramecc-:xnack-"), True),
    (SimpleNamespace(gcnArchName = "gfx1103", is_integrated = None), True),
    (SimpleNamespace(gcnArchName = "gfx1036"), True),
    (SimpleNamespace(gcnArchName = "gfx1100"), False),
    (SimpleNamespace(gcnArchName = "gfx1201"), False),
    (SimpleNamespace(gcnArchName = "gfx942", is_integrated = 1), True),
    (SimpleNamespace(gcn_arch_name = "gfx1150"), True),
    (SimpleNamespace(name = "AMD Radeon 8060S Graphics"), True),
    (SimpleNamespace(name = "AMD Radeon 780M"), True),
    (SimpleNamespace(name = "AMD Radeon RX 7600M XT"), False),
    (SimpleNamespace(name = "AMD Radeon RX 7900 XTX"), False),
]


@pytest.mark.parametrize("props,unified", CASES)
def test_training_inventory_and_llama_cpp_agree(props, unified):
    from core.training.worker import _rocm_classify_unified_memory
    from utils.hardware.hardware import _rocm_props_unified_status

    assert classify_rocm_unified_memory(props)[1] is unified
    assert _rocm_classify_unified_memory(props)[1] is unified
    assert _rocm_props_unified_status(props) is unified


def test_the_set_covers_every_arch_either_copy_knew():
    assert {"gfx1150", "gfx1151", "gfx1152", "gfx1103", "gfx1153", "gfx1036"} <= APU_ARCHES


def _fake_rocm_torch(props):
    torch = types.ModuleType("torch")
    torch.version = SimpleNamespace(hip = "6.1.0")
    torch.__version__ = "2.6.0+rocm6.1"
    torch.cuda = SimpleNamespace(
        is_available = lambda: True,
        current_device = lambda: 0,
        get_device_properties = lambda _o: props,
    )
    return torch


@pytest.mark.parametrize(
    "props,kind",
    [
        # HIP before 6.2 leaves is_integrated unset on Strix Halo and Phoenix.
        (SimpleNamespace(gcnArchName = "gfx1151", total_memory = 64 << 30), "unified_memory"),
        (SimpleNamespace(gcnArchName = "gfx1103", total_memory = 16 << 30), "unified_memory"),
        (SimpleNamespace(gcnArchName = "gfx1100", total_memory = 24 << 30), "discrete_vram"),
    ],
)
def test_the_diffusion_probe_reads_the_same_classifier(monkeypatch, props, kind):
    import utils.hardware as hardware
    from core.inference import diffusion_memory

    monkeypatch.setitem(sys.modules, "torch", _fake_rocm_torch(props))
    monkeypatch.setattr(hardware, "trusted_mem_get_info", lambda *a: (20 << 30, 24 << 30))
    monkeypatch.setattr(diffusion_memory, "_torch_is_rocm", lambda _t: True)
    assert diffusion_memory._cuda_memory("cuda")[2] == kind
