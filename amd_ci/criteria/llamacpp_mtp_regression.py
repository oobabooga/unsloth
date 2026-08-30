#!/usr/bin/env python3
"""Criteria: does the head model break what the base model does, on this host?

Judges only; it never observes. Pairs with probes/llamacpp_mtp_probe.py.

Used for two state sets, both of which put the WORKING model at `base`:
  V2 revision vs V3 revision of the same quant, and
  a quant without IQ4_NL vs one with it, at the same revision.

MODE is "regression", not "differential", and that polarity is the reason.
Differential mode VOIDs on "the base state did not exhibit the defect", which is
exactly what base is supposed to do here, so every honest run would read as a
non-result. Regression mode asks the question that was actually posed: is head
worse than base?

The gates exist because this comparison has several ways of looking decisive
while measuring nothing: two states pointed at the same file, a model that
quietly fell back to CPU, MTP that never engaged in a cell whose whole point is
MTP, an allocation failure on a shared box, or two different llama.cpp builds.
Each becomes INCONCLUSIVE rather than a confident NO_REGRESSION.
"""

from __future__ import annotations

TITLE = "Qwen3.8-27B under llama.cpp on gfx1151 (pass --title per cell)"
MODE = "regression"

# What the QUESTION touches, not what the host has. The reporters are on
# Windows, on discrete RDNA3/RDNA4, and several are on two GPUs at once; none of
# that is reachable from one integrated gfx1151 under RADV on Linux, and a
# report that stays quiet about it overstates its reach.
NEEDS = [
    "gpu", "rocm", "vulkan", "integrated_gpu",
    "discrete_gpu", "multi_gpu", "multi_gpu_amd",
    "windows", "windows_rocm_wddm", "nvidia", "amdvlk",
]

EXPECT_ARCH = "gfx1151"

# gates() sees every state; base_shows_defect/head_is_worse are handed one at a
# time, so anything cross-state is stashed here. Same pattern as
# criteria/gpu_summary_sees_others.py.
_CTX: dict = {"mtp_requested": False}


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _on_gpu(state: dict) -> bool:
    """Did the weights land on an accelerator rather than the CPU?

    Counted from `load_tensors: layer N assigned to device <dev>`, which the
    probe gets by running the server with -v. system_info is the wrong source:
    it names ROCm but never Vulkan, so a detector keyed on it calls a working
    Vulkan run cpu-only. A handful of layers on an accelerator would be a
    partial offload, so this asks for a clear majority rather than one hit.
    """
    by_dev = state.get("layers_by_device") or {}
    if by_dev:
        acc = sum(n for d, n in by_dev.items() if not d.upper().startswith("CPU"))
        return acc > sum(by_dev.values()) / 2
    devs = [d for d in (state.get("devices") or []) if not d.upper().startswith("CPU")]
    return bool(devs)


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    _CTX["mtp_requested"] = any(
        "mtp" in str(v.get("spec_type", "")).lower() for v in states.values())

    out: list[tuple[str, bool, str]] = []

    missing = [n for n, v in states.items() if v.get("setup_error") or not v.get("cmd")]
    out.append(("every state launched the binary", not missing,
                "; ".join(f"{n}: {states[n].get('setup_error', 'never ran')}" for n in missing)
                or ", ".join(sorted(states))))

    builds = {n: v.get("build") for n, v in states.items()}
    same_build = len(set(builds.values())) == 1 and all(builds.values())
    out.append(("all states used the same llama.cpp build", same_build,
                ", ".join(f"{n}={b}" for n, b in builds.items())))

    on_gpu = {n: _on_gpu(v) for n, v in states.items()}
    out.append(("every state put the model on the GPU", all(on_gpu.values()),
                "; ".join(f"{n}={states[n].get('layers_by_device') or states[n].get('devices')}"
                          for n in states)))

    archs = {n: (v.get("gfx") or "?") for n, v in states.items()}
    right_arch = all(EXPECT_ARCH in a for a in archs.values())
    out.append((f"host is {EXPECT_ARCH}", right_arch,
                ", ".join(f"{n}={a}" for n, a in archs.items())))

    # A cell that asked for MTP and never got it compares nothing. Only one
    # state needs to prove it, because the other may have died before it could.
    if _CTX["mtp_requested"]:
        engaged = {n: bool(v.get("mtp_engaged")) for n, v in states.items()}
        out.append(("MTP actually engaged in at least one state", any(engaged.values()),
                    ", ".join(f"{n}={e}" for n, e in engaged.items())))

    # On a 62 GiB shared box an allocation failure looks exactly like the defect
    # from the outside, and would be reported as one. It is not.
    oom = {n: "out_of_memory" in (v.get("markers") or []) for n, v in states.items()}
    out.append(("no state ran out of memory", not any(oom.values()),
                ", ".join(f"{n}={o}" for n, o in oom.items())))

    # The single easiest way to fake a confident NO_REGRESSION.
    ggufs = {n: v.get("gguf") for n, v in states.items()}
    distinct = len(set(ggufs.values())) == len(ggufs)
    out.append(("the states used different model files", distinct,
                "; ".join(f"{n}={g}" for n, g in ggufs.items())))

    return out


def table(obs: dict) -> str:
    rows = ["| state | model | ready | tokens | draft_n | mtp | rc/sig | zombie | markers |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        gguf = (v.get("gguf") or "").rsplit("/", 1)[-1]
        rc = v.get("rc")
        sig = v.get("signal")
        rows.append(
            f"| {name} | {gguf} | {v.get('server_ready')} | {v.get('tokens_generated')} | "
            f"{v.get('draft_n')} | {v.get('mtp_engaged')} | "
            f"{rc}{'/SIG' + str(sig) if sig else ''} | {v.get('zombie')} | "
            f"{', '.join(v.get('markers') or []) or '-'} |")
    rows.append("")
    rows.append("`tokens` is from a real POST /completion. /health is not used as a success "
               "signal: llama.cpp#27306 keeps answering it after the Vulkan device is lost.")
    return "\n".join(rows)


def _describe(state: dict) -> str:
    markers = state.get("markers") or []
    if markers:
        return "markers " + ", ".join(markers)
    if state.get("zombie"):
        return "process alive but /completion produced nothing"
    sig = state.get("signal")
    if sig:
        return f"killed by signal {sig}" + (" (SIGABRT)" if sig == 6 else "")
    if not state.get("server_ready"):
        return "server never became ready"
    return f"exited rc={state.get('rc')} with no diagnostic in the log"


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    b_ok, h_ok = bool(base.get("succeeded")), bool(head.get("succeeded"))

    if b_ok and not h_ok:
        return True, (f"base generated {base.get('tokens_generated')} tokens; "
                      f"head produced none - {_describe(head)}")
    if not b_ok and not h_ok:
        return False, (f"neither state generated a token, so what differs between the states "
                       f"is not the variable: base {_describe(base)}; head {_describe(head)}. "
                       f"If MTP was requested here, that points at MTP rather than the quant")
    if not b_ok and h_ok:
        return False, (f"head worked and base did not - {_describe(base)}. "
                       f"The reported direction is not what this host shows")
    return False, (f"both states generated tokens (base {base.get('tokens_generated')}, "
                   f"head {head.get('tokens_generated')}); the reported crash did not "
                   f"reproduce on this host with this configuration")
