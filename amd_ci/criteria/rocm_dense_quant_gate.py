#!/usr/bin/env python3
"""Criteria: does the dense torchao ladder still classify a ROCm GPU as NVIDIA? (PR 9828)

The defect is a MISREAD, not a crash. ROCm returns a gfx version through
`get_device_capability()`, so gfx1151 answers (11, 5) and clears every NVIDIA SM
floor in the ladder, and the loader then probes schemes on a card that has no such
kernels. On the reporter's gfx1103 that probe is what takes the backend down; on
this runner, with a wheel that matches the host, the same probe returns cleanly.

So the defect predicate is reachability, not a signal: does a bf16 "cuda" target on
a machine the gates have proven is real ROCm gfx1151 enter the ladder at all. That
is what reproduces here, and demanding a SIGSEGV instead would make every healthy
gfx1151 run VOID while proving nothing about the code that was changed.

Pairs with probes/rocm_dense_quant_probe.py.
"""

from __future__ import annotations

TITLE = "Dense torchao quant gate on ROCm gfx1151, base versus head"
MODE = "differential"

# Declared for what the CHANGE touches, not for what this host happens to be. The
# card in the bug report is a gfx1103 APU; the cache-key correction is about telling
# two cards apart; the ladder it gates is an NVIDIA path; the stub refactor it grew
# out of is Windows ROCm. None of that is answerable on one integrated Linux AMD GPU,
# and an under-declared NEEDS would leave the report bounding nothing.
NEEDS = [
    "linux", "gpu", "rocm", "integrated_gpu",
    "discrete_gpu", "multi_gpu", "multi_gpu_amd",
    "nvidia", "windows", "windows_rocm_wddm",
]

_TORCHAO_TE_MODES = ("int8", "fp8_dynamic", "nvfp4")


def _env(o: dict) -> dict:
    return o.get("env") or {}


def _sec(o: dict, name: str) -> dict:
    return o.get(name) or {}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = [n for n in obs if not n.startswith("_")]

    for name in ("base", "head"):
        o = obs.get(name) or {}
        ok = bool(_sec(o, "gate")) and "dense_transformer_supported" in _sec(o, "gate")
        detail = "observed" if ok else str(o.get("_missing_output") or o.get("error") or
                                            _sec(o, "gate").get("worker_error") or "no gate reading")
        out.append((f"{name} produced a gate reading", ok, detail))

    rocm = all(bool(_env(obs[n]).get("hip")) for n in states)
    out.append(("every state ran on a HIP/ROCm torch", rocm,
                ", ".join(f"{n}={_env(obs[n]).get('hip')}" for n in states)))

    archs = {n: str(_env(obs[n]).get("arch", "")) for n in states}
    arch_ok = all(a.startswith("gfx1151") for a in archs.values())
    out.append(("the card is gfx1151", arch_ok, ", ".join(f"{n}={a or '?'}" for n, a in archs.items())))

    # The trap the PR documents itself: under a ROCm wheel that does not match the host
    # EVERY kernel faults, and a differential measured in that state would read a broken
    # wheel as the torchao defect.
    health = all(_env(obs[n]).get("alloc_ok") and _env(obs[n]).get("bf16_matmul_ok")
                 for n in states)
    out.append(("a trivial allocation and a bf16 matmul both succeeded", health,
                ", ".join(f"{n}: alloc={_env(obs[n]).get('alloc_ok')} "
                          f"matmul={_env(obs[n]).get('bf16_matmul_ok')}" for n in states)))

    # A stubbed or absent torchao makes the base refuse for a reason that has nothing to
    # do with this change, and the differential would be measuring the stub.
    stubbed = [n for n in states if _env(obs[n]).get("torchao_stubbed")]
    missing = [n for n in states if _env(obs[n]).get("torchao") is None]
    out.append(("torchao is real, not the stub and not absent", not stubbed and not missing,
                f"stubbed={stubbed or 'none'}, absent={missing or 'none'}, "
                + ", ".join(f"{n}={_env(obs[n]).get('torchao')}" for n in states)))

    # The base leg runs code that can fault by construction; if it did, say so instead of
    # reading its absence as a result.
    crashed = [f"{n}.{w}" for n in states for w in ("env", "gate", "select", "te", "cache")
               if (obs[n].get(w) or {}).get("killed_by_signal")]
    out.append(("no observation child was killed by a signal", not crashed,
                ", ".join(crashed) if crashed else "none"))
    return out


def table(obs: dict) -> str:
    rows = ["| observation | " + " | ".join(n for n in obs if not n.startswith("_")) + " |"]
    rows.append("|---" * (1 + len([n for n in obs if not n.startswith("_")])) + "|")
    states = [n for n in obs if not n.startswith("_")]

    def row(label, fn):
        rows.append(f"| {label} | " + " | ".join(_fmt(fn(obs[n])) for n in states) + " |")

    row("capability seen", lambda o: _sec(o, "gate").get("capability_seen_by_module"))
    row("`torch_is_rocm` exists", lambda o: _sec(o, "gate").get("has_torch_is_rocm"))
    row("`torch_is_rocm()`", lambda o: _sec(o, "gate").get("torch_is_rocm"))
    row("`dense_transformer_supported`", lambda o: _sec(o, "gate").get("dense_transformer_supported"))
    row("refusal names ROCm/AMD", lambda o: _names_rocm(_sec(o, "gate").get("unsupported_reason")))
    row("`auto` selects", lambda o: _sec(o, "select").get("selected", {}).get("auto"))
    row("auto candidates", lambda o: _sec(o, "select").get("auto_candidates"))
    row("explicit int8 / fp8 / nvfp4 / mxfp8", lambda o: [
        _sec(o, "select").get("selected", {}).get(s) for s in ("int8", "fp8", "nvfp4", "mxfp8")])
    row("scheme-probe tripwire calls", lambda o: _sec(o, "select").get("tripwire_call_count"))
    row("TE plain fp8", lambda o: _sec(o, "te").get("supported", {}).get("fp8"))
    row("TE torchao modes", lambda o: [
        _sec(o, "te").get("supported", {}).get(m) for m in _TORCHAO_TE_MODES])
    row("child asked about", lambda o: _sec(o, "cache").get("child_asked_about"))
    row("cache keys after int8", lambda o: _sec(o, "cache").get("cache_keys"))
    row("in-process re-probes", lambda o: _sec(o, "cache").get("in_process_probe_count"))
    row("`_crashed_child_verdict` exists", lambda o: _sec(o, "cache").get("has_crashed_child_verdict"))
    return "\n".join(rows)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, list):
        return ", ".join("-" if x is None else f"`{x}`" for x in v) if v else "(empty)"
    return f"`{v}`"


def _names_rocm(reason) -> bool | None:
    if reason is None:
        return None
    low = str(reason).lower()
    return "rocm" in low or "amd" in low


def base_shows_defect(base: dict) -> bool:
    """The ladder is entered on an AMD card.

    Two halves, and both are needed. `dense_transformer_supported` True says the gate let
    a ROCm GPU through; a non-zero tripwire count says selection really walked the ladder
    and asked whether an NVIDIA-only scheme was supported on it. Either alone could be
    argued with."""
    gate = _sec(base, "gate")
    select = _sec(base, "select")
    return (gate.get("dense_transformer_supported") is True
            and (select.get("tripwire_call_count") or 0) > 0)


def head_is_fixed(head: dict) -> bool:
    gate, select, te, cache = (_sec(head, s) for s in ("gate", "select", "te", "cache"))

    refused = gate.get("dense_transformer_supported") is False
    reason_ok = _names_rocm(gate.get("unsupported_reason")) is True
    # Nothing reached the code that faults.
    untouched = (select.get("tripwire_call_count") or 0) == 0
    no_scheme = all(select.get("selected", {}).get(m) is None
                    for m in ("auto", "int8", "fp8", "nvfp4", "mxfp8"))
    no_candidates = not (select.get("auto_candidates") or [])
    # The gate must remove the torchao paths without taking the plain dtype cast with it.
    te_ok = (te.get("supported", {}).get("fp8") is True
             and all(te.get("supported", {}).get(m) is False for m in _TORCHAO_TE_MODES))
    # A completed child verdict is consumed, under a key qualified by the card, and the
    # child was asked about that same card rather than a bare "cuda".
    cache_ok = (cache.get("in_process_probe_count") == 0
                and cache.get("int8_supported") is True
                and all(str(d).startswith("cuda:") for d in (cache.get("child_asked_about") or ["x"]))
                and all("@cuda:" in k for k in (cache.get("cache_keys") or ["x"])))
    return all([refused, reason_ok, untouched, no_scheme, no_candidates, te_ok, cache_ok])
