# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ROCm must never enter the dense torchao quant ladder, and a crashed probe child is a verdict.

#9396: image generation on a Radeon 780M (gfx1103) took the whole backend down with
``Fatal Python error: Segmentation fault`` in ``_run_smoke_probe``. Two independent defects:

  1. ``torch.cuda.get_device_capability()`` on ROCm returns the AMD gfx version, and the ladder
     compares it against NVIDIA SM floors -- (11, 0) reads as "Blackwell sm_100+", so an
     integrated RDNA 3 GPU was offered fp8 / mxfp8 / int8 and probed with torchao.
  2. The out-of-process probe exists so that a fault in the GPU userspace cannot take the server
     with it. A child killed by SIGSEGV was read as "could not tell", so the caller re-ran the
     identical probe IN-PROCESS and died the same way -- and cached nothing, so every retry
     repeated it.

Hermetic: torch is stubbed via ``sys.modules`` and nothing spawns.
"""

from __future__ import annotations

import sys
import types

import pytest

import core.inference.diffusion_precision as dp
import core.inference.diffusion_transformer_quant as tq


def _target(*, device = "cuda", dtype = "bfloat16"):
    return types.SimpleNamespace(device = device, dtype = dtype)


def _stub_torch(monkeypatch, *, hip = None, version_str = "2.10.0+cu128", cc = (11, 0)):
    """A torch stub that answers like a ROCm build when ``hip`` is set.

    ROCm's real answers: ``cuda.is_available()`` True, device string "cuda", and a capability pair
    that is the gfx version (gfx1103 -> (11, 0), gfx1151 -> (11, 5), gfx942 -> (9, 4))."""
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.float8_e4m3fn = "float8_e4m3fn"
    torch.__version__ = version_str
    torch.version = types.SimpleNamespace(hip = hip, cuda = None if hip else "12.8")
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        get_device_capability = lambda *a: cc,
        get_device_name = lambda *a: "AMD Radeon  780M Graphics" if hip else "NVIDIA B200",
        current_device = lambda: 0,
    )
    torch.nn = types.SimpleNamespace(
        Embedding = type("Embedding", (), {}),
        ModuleList = type("ModuleList", (list,), {}),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(tq, "_SMOKE_CACHE", {}, raising = True)
    return torch


def _forbid_probe(monkeypatch):
    """Fail loudly if anything reaches the smoke probe: on ROCm it is what segfaults."""

    def _boom(*args, **kwargs):
        raise AssertionError("the torchao smoke probe must not run on ROCm")

    monkeypatch.setattr(tq, "_scheme_supported", _boom)
    monkeypatch.setattr(tq, "_run_smoke_probe", _boom)


# ── 1. the arch gate ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hip, version_str",
    [
        ("7.1.25424", "2.10.0+rocm7.1"),  # the build in the report
        (None, "2.10.0+rocm7.1"),  # AMD wheels that tag only __version__
    ],
)
def test_dense_transformer_unsupported_on_rocm(monkeypatch, hip, version_str):
    _stub_torch(monkeypatch, hip = hip, version_str = version_str)
    assert tq.torch_is_rocm() is True
    assert tq.dense_transformer_supported(_target()) is False


def test_dense_transformer_still_supported_on_cuda(monkeypatch):
    _stub_torch(monkeypatch, hip = None, version_str = "2.10.0+cu128", cc = (10, 0))
    assert tq.torch_is_rocm() is False
    assert tq.dense_transformer_supported(_target()) is True


@pytest.mark.parametrize("cc", [(11, 0), (11, 5), (9, 4)])  # gfx1103, gfx1151, gfx942
def test_auto_selects_nothing_and_never_probes_on_rocm(monkeypatch, cc):
    """The regression: (11, 0) is gfx1103, not sm_110, and must not clear the sm_100 tier."""
    _stub_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1", cc = cc)
    _forbid_probe(monkeypatch)
    assert tq.select_transformer_quant_scheme(_target(), "auto") is None
    assert tq.auto_scheme_candidates(_target()) == ()


@pytest.mark.parametrize("scheme", list(tq.TQ_SCHEMES))
def test_explicit_scheme_refused_without_probing_on_rocm(monkeypatch, scheme):
    _stub_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    _forbid_probe(monkeypatch)
    assert tq.select_transformer_quant_scheme(_target(), scheme) is None


def test_refusal_reason_names_rocm(monkeypatch):
    _stub_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    reason = tq.dense_transformer_unsupported_reason(_target())
    assert "ROCm" in reason and "AMD" in reason


def test_refusal_reason_unchanged_off_cuda(monkeypatch):
    _stub_torch(monkeypatch, hip = None)
    assert "CUDA GPU in bf16" in tq.dense_transformer_unsupported_reason(_target(device = "cpu"))


# ── 2. the text-encoder gate has the same capability misread ──────────────────


@pytest.mark.parametrize(
    "mode", [dp.TE_QUANT_INT8, dp.TE_QUANT_FP8_DYNAMIC, dp.TE_QUANT_NVFP4]
)
def test_torchao_text_encoder_modes_unsupported_on_rocm(monkeypatch, mode):
    _stub_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    assert dp.te_quant_supported(_target(), mode) is False


def test_layerwise_fp8_text_encoder_still_supported_on_rocm(monkeypatch):
    """Plain fp8 is a torch dtype cast with no torchao in it, so ROCm keeps it."""
    _stub_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    assert dp.te_quant_supported(_target(), dp.TE_QUANT_FP8) is True


@pytest.mark.parametrize(
    "mode", [dp.TE_QUANT_INT8, dp.TE_QUANT_FP8_DYNAMIC, dp.TE_QUANT_NVFP4]
)
def test_torchao_text_encoder_modes_still_supported_on_cuda(monkeypatch, mode):
    _stub_torch(monkeypatch, hip = None, version_str = "2.10.0+cu128", cc = (10, 0))
    monkeypatch.setattr(dp, "is_stubbed", lambda name: False)
    assert dp.te_quant_supported(_target(), mode) is True


# ── 3. a crashed probe child is a verdict, not a reason to retry in-process ───


class _Proc:
    def __init__(self, exitcode):
        self.exitcode = exitcode


@pytest.mark.parametrize("signal_number", sorted(tq._PROBE_CRASH_SIGNALS))
def test_crashed_child_marks_every_scheme_unusable(signal_number):
    verdict = tq._crashed_child_verdict(_Proc(-signal_number), "cuda")
    assert verdict == {scheme: False for scheme in tq.TQ_SCHEMES}


@pytest.mark.parametrize("exitcode", [0, 1, -9, -15, None])
def test_non_crash_exits_still_fall_back_in_process(exitcode):
    """SIGKILL is the OOM killer, SIGTERM is our own timeout teardown; neither is a verdict."""
    assert tq._crashed_child_verdict(_Proc(exitcode), "cuda") is None
    assert tq._crashed_child_verdict(None, "cuda") is None


def test_crash_verdict_stops_the_in_process_probe(monkeypatch):
    """End of the #9396 chain: the child died, so the parent must not run the same probe."""
    _stub_torch(monkeypatch, hip = "7.1.25424", version_str = "2.10.0+rocm7.1")
    monkeypatch.setattr(
        tq, "_child_probe_table", lambda device: tq._crashed_child_verdict(_Proc(-11), device)
    )

    def _boom(*args, **kwargs):
        raise AssertionError("a crashed probe child must not be retried in this process")

    monkeypatch.setattr(tq, "_run_smoke_probe", _boom)
    assert tq._scheme_supported(tq.TQ_INT8, "cuda") is False
    # Cached, so a second load answers from memory instead of spawning another child to die.
    assert tq._SMOKE_CACHE[(tq.TQ_INT8, "cuda:0")] is False


def test_child_verdict_is_used_instead_of_a_second_in_process_probe(monkeypatch):
    """The child and the in-process probe wrote/read DIFFERENT cache keys ("cuda" vs "cuda:0"),
    so a child that answered was ignored and the parent probed again -- which is what carried the
    fatal probe into the server process. One key, and the child's answer is the answer."""
    _stub_torch(monkeypatch, hip = None, version_str = "2.10.0+cu128", cc = (10, 0))
    monkeypatch.setattr(
        tq, "_child_probe_table", lambda device: {s: (s == tq.TQ_INT8) for s in tq.TQ_SCHEMES}
    )

    def _boom(*args, **kwargs):
        raise AssertionError("the child already answered; the parent must not probe again")

    monkeypatch.setattr(tq, "_run_smoke_probe", _boom)
    assert tq._scheme_supported(tq.TQ_INT8, "cuda") is True
    assert tq._scheme_supported(tq.TQ_FP8, "cuda") is False
    assert tq._SMOKE_CACHE[(tq.TQ_INT8, "cuda:0")] is True


@pytest.mark.parametrize("unproven_ok", [True, False])
def test_child_out_of_memory_is_still_not_a_verdict(monkeypatch, unproven_ok):
    """None is the child's allocator failure: not cached, and not turned into "unsupported"."""
    _stub_torch(monkeypatch, hip = None, version_str = "2.10.0+cu128", cc = (10, 0))
    monkeypatch.setattr(tq, "_child_probe_table", lambda device: {s: None for s in tq.TQ_SCHEMES})
    monkeypatch.setattr(tq, "_run_smoke_probe", lambda *a, **k: pytest.fail("no re-probe"))
    assert tq._scheme_supported(tq.TQ_INT8, "cuda", unproven_ok = unproven_ok) is unproven_ok
    assert (tq.TQ_INT8, "cuda:0") not in tq._SMOKE_CACHE


def test_a_scheme_missing_from_the_table_is_still_not_an_answer(monkeypatch):
    """Only a present key is the child's verdict; an absent one falls through as it always did."""
    _stub_torch(monkeypatch, hip = None, version_str = "2.10.0+cu128", cc = (10, 0))
    monkeypatch.setattr(tq, "_child_probe_table", lambda device: {tq.TQ_FP8: True})
    probed: list = []
    monkeypatch.setattr(
        tq, "_run_smoke_probe", lambda scheme, device: probed.append(scheme) or True
    )
    assert tq._scheme_supported(tq.TQ_INT8, "cuda") is True
    assert probed == [tq.TQ_INT8]
