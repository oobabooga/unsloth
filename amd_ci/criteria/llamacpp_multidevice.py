#!/usr/bin/env python3
"""Criteria: does the head model break when the graph is split across devices?

Judges only; it never observes. Pairs with probes/llamacpp_mtp_probe.py and
probes/rpc_servers_fixture.py.

Same shape as llamacpp_mtp_regression, plus the gate that this whole branch
lives or dies on: **the model must actually have been split across two or more
non-CPU devices**. A multi-GPU spoof that quietly ran on one device is a
single-device run wearing a costume, and it would produce a confident
NO_REGRESSION about a configuration that was never tested. That gate is the
difference between a result and a decoration.

The spoof is two `ggml-rpc-server` processes sharing ONE physical GPU. It
reproduces multi-backend SCHEDULING - `ggml_backend_sched` with several
backends, `-sm layer` / `-sm row`, `-ts`, cross-device KV placement - which is
what llama.cpp#24492 blames. It does NOT reproduce a second card: no PCIe hop,
no second driver context, no per-card VRAM ceiling. NEEDS says so, and the
fixture's own READY payload carries `spoofed: true` into the observations.
"""

from __future__ import annotations

TITLE = "Qwen3.8-27B split across devices on gfx1151 (RPC-spoofed)"
MODE = "regression"

# multi_gpu and multi_gpu_amd stay in NEEDS even though this branch is the whole
# point: the host still has one card, and the report must keep saying so.
NEEDS = [
    "gpu", "rocm", "vulkan", "integrated_gpu",
    "discrete_gpu", "multi_gpu", "multi_gpu_amd",
    "windows", "windows_rocm_wddm", "nvidia", "amdvlk",
]

EXPECT_ARCH = "gfx1151"
MIN_DEVICES = 2

_CTX: dict = {"mtp_requested": False}


def _states(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _accel_devices(state: dict) -> dict[str, int]:
    """Layer counts per non-CPU device.

    From `load_tensors: layer N assigned to device <dev>`, which the probe gets
    by running the server with -v. `Vulkan_Host` and `*_Mapped` are host-side
    staging buffers, not places a layer computes, so they do not count towards
    the split.
    """
    by_dev = state.get("layers_by_device") or {}
    return {d: n for d, n in by_dev.items()
            if not d.upper().startswith("CPU") and "_HOST" not in d.upper()}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    states = _states(obs)
    _CTX["mtp_requested"] = any(
        "mtp" in str(v.get("spec_type", "")).lower() for v in states.values())
    fixture = obs.get("_fixture") or {}

    out: list[tuple[str, bool, str]] = []

    missing = [n for n, v in states.items() if v.get("setup_error") or not v.get("cmd")]
    out.append(("every state launched the binary", not missing,
                "; ".join(f"{n}: {states[n].get('setup_error', 'never ran')}" for n in missing)
                or ", ".join(sorted(states))))

    builds = {n: v.get("build") for n, v in states.items()}
    same_build = len(set(builds.values())) == 1 and all(builds.values())
    out.append(("all states used the same llama.cpp build", same_build,
                ", ".join(f"{n}={b}" for n, b in builds.items())))

    # THE gate for this branch.
    split = {n: _accel_devices(v) for n, v in states.items()}
    really_split = all(len(d) >= MIN_DEVICES for d in split.values())
    out.append((f"every state split across >= {MIN_DEVICES} non-CPU devices", really_split,
                "; ".join(f"{n}={d}" for n, d in split.items())))

    # A split that put one layer on the second device is technically two devices
    # and practically one. Ask for a real share.
    balanced = {}
    for n, d in split.items():
        total = sum(d.values()) or 1
        balanced[n] = min(d.values()) / total if d else 0.0
    out.append(("the smaller share is at least 10% of layers",
                all(v >= 0.10 for v in balanced.values()),
                ", ".join(f"{n}={v:.0%}" for n, v in balanced.items())))

    # Only asserted for the RPC cells. The dual-backend cells reach two devices
    # by merging libggml-vulkan.so into a ROCm build instead, and have no
    # fixture at all; demanding one there would fail a cell for not using a
    # mechanism it was never meant to use.
    if fixture:
        out.append(("the RPC fixture came up", bool(fixture.get("ports")),
                    str(fixture.get("rpc_arg") or fixture.get("error") or "no fixture")))

    archs = {n: (v.get("gfx") or "?") for n, v in states.items()}
    out.append((f"host is {EXPECT_ARCH}", all(EXPECT_ARCH in a for a in archs.values()),
                ", ".join(f"{n}={a}" for n, a in archs.items())))

    if _CTX["mtp_requested"]:
        engaged = {n: bool(v.get("mtp_engaged")) for n, v in states.items()}
        out.append(("MTP actually engaged in at least one state", any(engaged.values()),
                    ", ".join(f"{n}={e}" for n, e in engaged.items())))

    # Two RPC servers share one pool, so an allocation failure is likelier here
    # than anywhere else in this investigation, and it looks exactly like the
    # defect from the outside. It is not.
    oom = {n: "out_of_memory" in (v.get("markers") or []) for n, v in states.items()}
    out.append(("no state ran out of memory", not any(oom.values()),
                ", ".join(f"{n}={o}" for n, o in oom.items())))

    ggufs = {n: v.get("gguf") for n, v in states.items()}
    out.append(("the states used different model files",
                len(set(ggufs.values())) == len(ggufs),
                "; ".join(f"{n}={g}" for n, g in ggufs.items())))

    return out


def table(obs: dict) -> str:
    rows = ["| state | model | split | tokens | mtp | rc/sig | zombie | markers |",
            "|---|---|---|---|---|---|---|---|"]
    for name, v in _states(obs).items():
        sig = v.get("signal")
        rows.append(
            f"| {name} | {(v.get('gguf') or '').rsplit('/', 1)[-1]} | "
            f"{_accel_devices(v)} | {v.get('tokens_generated')} | {v.get('mtp_engaged')} | "
            f"{v.get('rc')}{'/SIG' + str(sig) if sig else ''} | {v.get('zombie')} | "
            f"{', '.join(v.get('markers') or []) or '-'} |")
    fixture = obs.get("_fixture") or {}
    how = (f"{len(fixture.get('ports') or [])} ggml-rpc-server processes on one physical "
           f"{fixture.get('device', '?')}" if fixture
           else "two ggml backends (Vulkan and HIP) over one physical GPU")
    rows += ["", f"Devices are SPOOFED: {how}. This exercises multi-backend scheduling on a "
                 f"host with ONE card, not two cards."]
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
        return True, (f"split across devices, base generated {base.get('tokens_generated')} "
                      f"tokens and head produced none - {_describe(head)}")
    if not b_ok and not h_ok:
        return False, (f"neither state generated a token once the graph was split, so what "
                       f"differs between the states is not the variable: base "
                       f"{_describe(base)}; head {_describe(head)}. Compare against the "
                       f"single-device cell before blaming the split")
    if not b_ok and h_ok:
        return False, (f"head worked and base did not - {_describe(base)}. The reported "
                       f"direction is not what this host shows")
    return False, (f"both states generated tokens across the split (base "
                   f"{base.get('tokens_generated')}, head {head.get('tokens_generated')}); "
                   f"the reported crash did not reproduce on spoofed multi-device")
