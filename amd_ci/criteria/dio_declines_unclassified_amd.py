#!/usr/bin/env python3
"""Criteria: DirectIO must not be chosen for an AMD GPU nothing classified.

Judges only. Pairs with probes/dio_apu_probe.py.

The defect: `_SELF_EVIDENTLY_DISCRETE` held `rocm` and `hip`, on the stated
grounds that `_weights_in_host_memory` classifies AMD upstream. That classifier
is `_rocm_unified_memory_gpu_ids`, which needs a ROCm-enabled torch to read the
driver and returns the same empty set for "no APU here" as for "there is no ROCm
torch to ask". PyTorch publishes no Windows ROCm wheel, so on Windows -- the only
platform this policy fires on -- the classifier never answers, and the empty set
was read as "discrete".

This host is a Strix Halo: a unified-memory APU. So the question the differential
asks is a real one about real hardware, not a constructed case.

Runs in differential mode between the PR head before this fix (base) and after
it (head), which is the pair that brackets the change being judged.
"""

from __future__ import annotations

TITLE = "DirectIO on a unified-memory AMD APU (gfx1151)"
MODE = "differential"
NEEDS = ["gpu", "windows"]


def _is_strix_halo(o: dict) -> bool:
    """Corroborate the host from anything that names the part.

    Three sources, because on the host that matters two of them are silent:
    torch's device name exists only where a ROCm torch does, which on Windows is
    nowhere, and amd-smi is installed on the Windows boxes but answers
    `Error LoadLibraryA`. The Windows video controller name is what is left, and
    it is read through the toolkit's own mapping rather than a second copy.
    """
    blob = str(o.get("amd_smi", {}).get("stdout_head", "")).lower()
    names = " ".join(o.get("torch", {}).get("device_names", []) or []).lower()
    vc = o.get("video_controllers") or {}
    controllers = " ".join(vc.get("names") or []).lower()
    archs = [a.lower() for a in (vc.get("archs") or [])]
    markers = ("gfx1151", "strix", "8060s", "8050s")
    if "gfx1151" in archs:
        return True
    return any(m in blob or m in names or m in controllers for m in markers)


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    """Non-vacuity. Each of these, failing, would make the verdict meaningless."""
    out = []
    base, head = obs.get("base") or {}, obs.get("head") or {}

    for name, o in (("base", base), ("head", head)):
        out.append((
            f"{name} revision has the DirectIO gate",
            bool(o.get("feature_present")),
            "the revision defines _gpu_offload_confirmed"
            if o.get("feature_present")
            else "no _gpu_offload_confirmed; this state predates the feature, so the "
                 "pair does not bracket the change",
        ))
        out.append((
            f"{name} probe reached a verdict",
            o.get("gpu_offload_confirmed") is not None,
            str(o.get("gpu_offload_confirmed_error")
                or o.get("import_error")
                or o.get("error")
                or f"confirmed={o.get('gpu_offload_confirmed')}"),
        ))

    out.append((
        "the host is a unified-memory AMD APU",
        _is_strix_halo(head) or _is_strix_halo(base),
        "amd-smi, torch or the Windows video controller names a gfx1151 / Strix "
        "Halo / Radeon 8060S part"
        if (_is_strix_halo(head) or _is_strix_halo(base))
        else "could not corroborate the part from amd-smi, torch or the video "
             "controller; a verdict about APU handling on a host that may not be "
             "an APU says nothing",
    ))

    # The whole point is what happens when the classifier CANNOT answer. If it
    # can, this host does not exercise the defect and the result is not the one
    # being claimed -- so say so rather than passing quietly.
    answered = head.get("rocm_classification_answered")
    out.append((
        "the ROCm classifier cannot answer on this host",
        answered is False,
        "no ROCm torch, so `_rocm_unified_memory_gpu_ids` returns the empty set "
        "for lack of an answer rather than for lack of an APU"
        if answered is False
        else f"classifier answered={answered}; on a host where it CAN answer the "
             "APU is caught upstream and this differential is not exercising the gate",
    ))
    return out


def table(obs: dict) -> str:
    rows = [
        "| observation | base (pre-fix) | head (post-fix) |",
        "|---|---|---|",
    ]
    fields = [
        ("platform", "platform"),
        ("torch importable", "torch_importable"),
        ("torch is ROCm", "torch_is_rocm"),
        ("Windows video controller", "video_controller_names"),
        ("`_rocm_unified_memory_gpu_ids()`", "rocm_unified_memory_gpu_ids"),
        ("`_rocm_classification_answered()`", "rocm_classification_answered"),
        ("`_amd_apu_wants_unified_memory([0])`", "amd_apu_wants_unified_memory"),
        ("**`_gpu_offload_confirmed()`**", "gpu_offload_confirmed"),
        ("**policy emitted `--load-mode dio`**", "policy_emitted_dio"),
        ("**effective DirectIO**", "effective_dio"),
    ]
    for label, key in fields:
        cells = []
        for name in ("base", "head"):
            o = obs.get(name) or {}
            if key == "torch_importable":
                v = (o.get("torch") or {}).get("importable")
            elif key == "torch_is_rocm":
                v = bool((o.get("torch") or {}).get("hip_version"))
            elif key == "video_controller_names":
                v = ", ".join((o.get("video_controllers") or {}).get("names") or []) or "none"
            else:
                v = o.get(key, "absent")
            cells.append(f"`{v}`")
        rows.append(f"| {label} | {cells[0]} | {cells[1]} |")

    head = obs.get("head") or {}
    note = (
        "\n\nThe device list `--list-devices` would print is stubbed as "
        f"`{head.get('assumed_device_ids')}`: the runner carries no llama.cpp install, "
        "and the id format is fixed by ggml. That stub is the INPUT to the decision, "
        "not the decision. Everything else in the table was read from this machine."
    )
    if head.get("decision_platform_is_native") is False:
        note += (
            "\n\n**This leg is not Windows.** The loader decision was evaluated with "
            "`sys.platform` set to `win32`, because the policy is Windows-only and "
            "would otherwise decline for the platform before reaching the AMD gate. "
            "The classifier answers above are still this host's own."
        )
    return "\n".join(rows) + note


def base_shows_defect(base: dict) -> tuple[bool, str]:
    if base.get("gpu_offload_confirmed"):
        return True, (
            "the pre-fix revision confirms a discrete full offload on this APU and "
            f"emits `--load-mode dio` (emitted={base.get('policy_emitted_dio')}, "
            f"effective={base.get('effective_dio')}), having classified nothing: "
            f"`_rocm_unified_memory_gpu_ids()` returned "
            f"`{base.get('rocm_unified_memory_gpu_ids')}` for want of a ROCm torch"
        )
    return False, (
        "the pre-fix revision already declines on this host "
        f"(confirmed={base.get('gpu_offload_confirmed')}), so there is no defect here "
        "to fix and the verdict is VOID rather than a pass"
    )


def head_is_fixed(head: dict) -> tuple[bool, str]:
    if head.get("gpu_offload_confirmed"):
        return False, (
            "the fixed revision still confirms the offload on this APU "
            f"(emitted={head.get('policy_emitted_dio')}, "
            f"effective={head.get('effective_dio')})"
        )
    return True, (
        "the fixed revision declines: an AMD id now counts only once "
        "`_rocm_classification_answered()` says something actually looked at the "
        f"device, and here it returns `{head.get('rocm_classification_answered')}`. "
        f"No `--load-mode dio` is emitted (emitted={head.get('policy_emitted_dio')})"
    )
