#!/usr/bin/env python3
"""Criteria: unsloth#7449 defect 2, judged against measured memory.

The defect is "the weights land in system RAM instead of the GPU pool". #9884
narrowed the thing that causes it, ``GGML_CUDA_ENABLE_UNIFIED_MEMORY``, to the
case where host RAM is the larger pool and the weights outgrow the carve-out.

    base -> the commit before #9884, where any AMD APU got the flag
    head -> current main

The DECIDING question is the decision, because that is the only thing the two
states differ in: the llama.cpp binary, the model and the host are identical.
The residency legs are not the verdict; they are what makes the verdict mean
something. Their job is to establish, on this hardware, that the flag is in fact
the mechanism -- that turning it on moves the weights somewhere measurably
different. If it does not, then the decision the head makes is not the thing that
fixes #7449, and the gates below say so instead of quietly passing.

That is why `flag_on` and `flag_off` are gated for having RUN in both states but
their numbers are not compared between states. Comparing them across states would
be comparing a binary with itself.

Pairs with probes/studio_apu_residency_probe.py.
"""

from __future__ import annotations

TITLE = "unsloth#7449 defect 2: where a gfx1151 APU puts the weights"
MODE = "differential"

# Authored, not derived. The change is a ROCm APU memory policy; the report is a
# WINDOWS Strix Halo; and the same launch path has a discrete-GPU branch and a
# multi-GPU branch that decide differently and that this host cannot reach.
NEEDS = ["rocm", "gpu", "integrated_gpu", "windows", "windows_rocm_wddm",
         "discrete_gpu", "multi_gpu"]

# A leg whose generation produced no decode rate measured nothing: llama.cpp can
# print a load and then fail before the first token, and the memory deltas alone
# would still look like a successful residency reading.
MIN_TOKENS_PER_SEC = 0.01


def _model_decision(state: dict) -> dict | None:
    """The decision for the file actually on disk, not for a round number.

    The probe asks about the real model size first and a ladder afterwards, so
    this matches on bytes rather than taking element zero, which would silently
    follow a reordering of that list.
    """
    want = state.get("model_bytes")
    for d in (state.get("decision") or {}).get("decisions") or []:
        if want is not None and d.get("need_bytes") == want:
            return d
    return None


def _leg(state: dict, name: str) -> dict:
    return (state.get("legs") or {}).get(name) or {}


def _gpu_mib(leg: dict) -> float | None:
    """MiB of model weights llama.cpp itself says went to a GPU device.

    Keyed on the device NAME rather than on a total, because the whole question
    is which buffer they landed in. `system_info` is no substitute: it names ROCm
    for a run that touched no GPU.
    """
    bufs = leg.get("model_buffer_mib_by_device")
    if not bufs:
        return None
    return sum(v for k, v in bufs.items()
               if any(tag in k for tag in ("ROCm", "Vulkan", "HIP", "CUDA")))


def _host_mib(leg: dict) -> float | None:
    bufs = leg.get("model_buffer_mib_by_device")
    if not bufs:
        return None
    return sum(v for k, v in bufs.items()
               if any(tag in k for tag in ("CPU", "Mapped", "Host")))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    for name in ("base", "head"):
        st = obs.get(name)
        if not st:
            continue
        dec = st.get("decision") or {}
        out.append((f"{name} the decision child returned a reading",
                    not dec.get("child_failed") and not dec.get("backend_error"),
                    f"rc={dec.get('rc')}; "
                    + str(dec.get("torch_error") or dec.get("backend_error")
                          or dec.get("stderr_tail", ""))[-180:]))
        out.append((f"{name} torch is a ROCm build", bool(dec.get("torch_hip")),
                    f"torch={dec.get('torch_version')} hip={dec.get('torch_hip')} "
                    f"cuda={dec.get('torch_cuda')}"))
        devs = dec.get("devices") or []
        dev = devs[0] if devs else {}
        arch = str(dev.get("gcnArchName") or "")
        out.append((f"{name} the device is the gfx1151 APU", "gfx1151" in arch,
                    f"{dev.get('name')} arch={arch or 'unreadable'} "
                    f"is_integrated={dev.get('is_integrated')} "
                    f"total={dev.get('total_mib')} MiB"))
        out.append((f"{name} the decision answered for the model on disk",
                    bool(_model_decision(st)),
                    f"model = {st.get('model_gib')} GiB "
                    f"({st.get('model_bytes')} bytes); "
                    f"decided_by={(_model_decision(st) or {}).get('decided_by')}"))
        # Residency. These make the verdict mean something; they are not the verdict.
        for leg_name in ("flag_off", "flag_on"):
            leg = _leg(st, leg_name)
            out.append((f"{name} {leg_name}: llama.cpp ran and generated",
                        leg.get("rc") == 0 and not leg.get("timed_out")
                        and isinstance(leg.get("tg_ts"), (int, float))
                        and leg["tg_ts"] > MIN_TOKENS_PER_SEC,
                        f"rc={leg.get('rc')} timed_out={leg.get('timed_out')} "
                        f"tg={leg.get('tg_ts')} t/s pp={leg.get('pp_ts')} t/s "
                        f"in {leg.get('seconds')}s"))
            out.append((f"{name} {leg_name}: llama.cpp reported where the weights went",
                        bool(leg.get("model_buffer_mib_by_device")),
                        f"buffers={leg.get('model_buffer_mib_by_device')} "
                        f"layers={leg.get('layer_devices')}"))
            out.append((f"{name} {leg_name}: a memory counter answered",
                        any(isinstance(leg.get("mem_peak", {}).get(k), (int, float))
                            for k in ("vram_used_mib", "gtt_used_mib")),
                        f"peak={leg.get('mem_peak')} samples={leg.get('samples')} "
                        f"errors={leg.get('sample_errors')}"))
    return out


def table(obs: dict) -> str:
    rows = ["### The decision (this is what the two states differ in)", "",
            "| state | carve-out MiB | host RAM MiB | model | decided by | flag set? |",
            "|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        st = obs.get(name)
        if not st:
            continue
        dec = st.get("decision") or {}
        devs = dec.get("devices") or []
        pool = ((dec.get("rocm_selected_pool_mib") or {}).get("value")
                or (devs[0].get("total_mib") if devs else None))
        host = (dec.get("available_system_memory_mib") or {}).get("value")
        d = _model_decision(st) or {}
        rows.append(f"| {name} | {pool} | {host} | {st.get('model_gib')} GiB | "
                    f"`{d.get('decided_by')}` | **{d.get('unified_memory')}** |")

    rows += ["", "### The measurement (identical binary and model in both states,",
             "so these numbers are a property of the FLAG and of this hardware)", "",
             "| state | leg | GPU buffer MiB | host buffer MiB | VRAM used delta MiB | "
             "GTT/shared used delta MiB | host avail drop MiB | prefill t/s | decode t/s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        st = obs.get(name)
        if not st:
            continue
        for leg_name in ("flag_off", "flag_on"):
            leg = _leg(st, leg_name)
            if not leg:
                continue
            rows.append(
                f"| {name} | `{leg.get('flag')}` | {_gpu_mib(leg)} | {_host_mib(leg)} | "
                f"{leg.get('vram_used_delta_mib')} | {leg.get('gtt_used_delta_mib')} | "
                f"{leg.get('host_available_drop_mib')} | {leg.get('pp_ts')} | "
                f"**{leg.get('tg_ts')}** |")

    rows += ["", "Read the GPU/host buffer columns first: they are llama.cpp's own "
             "`load_tensors: <device> model buffer size` lines, which is the most "
             "direct statement of where the weights went that exists. The counter "
             "deltas are the driver's independent second opinion.", ""]

    # Whether the flag is even the mechanism, per state, from the state's own legs.
    for name in ("base", "head"):
        st = obs.get(name)
        if not st:
            continue
        off, on = _leg(st, "flag_off"), _leg(st, "flag_on")
        off_gpu, on_gpu = _gpu_mib(off), _gpu_mib(on)
        off_tg, on_tg = off.get("tg_ts"), on.get("tg_ts")
        if off_gpu is None or on_gpu is None:
            rows.append(f"- `{name}` could not compare the two legs: "
                        f"flag_off GPU buffer {off_gpu}, flag_on {on_gpu}")
            continue
        moved = abs(on_gpu - off_gpu) > 64  # MiB; below this is allocator noise
        rows.append(
            f"- `{name}` setting the flag {'MOVES' if moved else 'does NOT move'} the "
            f"weights: GPU buffer {off_gpu} MiB -> {on_gpu} MiB"
            + (f", decode {off_tg} -> {on_tg} t/s"
               if isinstance(off_tg, (int, float)) and isinstance(on_tg, (int, float))
               else ""))
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    d = _model_decision(base)
    if not d or "unified_memory" not in d:
        return False, "the base state produced no decision for the model on disk"
    pool = ((base.get("decision", {}).get("rocm_selected_pool_mib") or {}).get("value"))
    return bool(d["unified_memory"]), (
        f"a {base.get('model_gib')} GiB model {'DOES' if d['unified_memory'] else 'does not'} "
        f"get GGML_CUDA_ENABLE_UNIFIED_MEMORY at the base "
        f"(carve-out {pool} MiB, decided by `{d.get('decided_by')}`)")


def head_is_fixed(head: dict) -> tuple[bool, str]:
    d = _model_decision(head)
    if not d or "unified_memory" not in d:
        return False, "the head state produced no decision for the model on disk"
    return not bool(d["unified_memory"]), (
        f"on current main the same model "
        f"{'still gets' if d['unified_memory'] else 'does not get'} managed allocation "
        f"(decided by `{d.get('decided_by')}`)")
