# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether a ROCm device is an APU, whose "VRAM" is the same physical RAM the host uses.

Standard library only, so every layer can import it without another layer's dependencies.
"""

from __future__ import annotations

import re
from typing import Any

# Named unified-memory APUs with a shared GPU/system-RAM pool.
_NAMED_APU_ARCHES = frozenset({"gfx1150", "gfx1151", "gfx1152"})

# Arches LLVM's AMDGPUUsage table types as APU and that never shipped as a discrete board, so the
# arch alone settles it where the driver leaves hipDeviceProp_t::integrated unset (clr before
# rocm-6.2.0, and the legacy R0000 property ABI).
_UNNAMED_APU_ARCHES = frozenset(
    {
        "gfx801",
        "gfx810",
        "gfx902",
        "gfx909",
        "gfx90c",
        "gfx1013",
        "gfx1033",
        "gfx1035",
        "gfx1036",
        "gfx1103",
        "gfx1153",
    }
)

APU_ARCHES = _NAMED_APU_ARCHES | _UNNAMED_APU_ARCHES

# Integrated-only Radeon names, matched as whole tokens so "RX 7600M" does not match "600M".
_APU_NAME_RE = re.compile(r"(?<![\w-])(?:820m|8040s|780m|760m|740m|680m|660m|610m)(?!\w)")
_APU_NAME_MARKERS = ("890m", "880m", "8065s", "8060s", "8050s", "860m", "840m")


def rocm_arch(props: Any) -> str:
    """The device's gfx arch without ``:xnack`` features, or "" when no attribute carries one."""
    for attr in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
        value = (getattr(props, attr, "") or "").split(":")[0].strip()
        if value:
            return value
    return ""


def classify_rocm_unified_memory(props: Any) -> tuple[str, bool]:
    """``(gcn_arch, is_unified)`` for a ROCm device's torch properties.

    ``is_integrated`` can only upgrade to unified. The name is read only when no arch attribute
    exists, which happens on AMD SDK / Radeon wheels. Call this only on ROCm: the NVIDIA GeForce
    840M shares a name marker.
    """
    gcn_arch = rocm_arch(props)
    if getattr(props, "is_integrated", 0):
        return gcn_arch, True
    if gcn_arch:
        return gcn_arch, gcn_arch.lower() in APU_ARCHES
    name = (getattr(props, "name", "") or "").lower()
    return gcn_arch, any(marker in name for marker in _APU_NAME_MARKERS) or (
        _APU_NAME_RE.search(name) is not None
    )
