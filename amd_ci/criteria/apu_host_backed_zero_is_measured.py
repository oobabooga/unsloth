#!/usr/bin/env python3
"""Criteria: on a Linux ROCm APU whose sysfs total matches torch's, does the
Settings > System tile say the machine has no VRAM?

unsloth#11451, against unsloth#7449 defect 1.

The defect is a two-hop one and neither hop looks wrong on its own.

  hop 1  `_rocm_linux_shared_pool_host_gb_by_index` OMITS an index whose sysfs
         total is readable and whose torch total does not exceed it by more than
         10%. Omission is the function's way of saying "unknown".

  hop 2  `gpu-vram.ts::gpuMemoryTotalsGb` reads an absent (or null)
         `shared_memory_host_backed_gb` as `hostBackedKnown = false` and then
         takes `hostBacked = total`. The reserved bucket stays empty, the
         dedicated bucket collapses to zero, and the whole carve-out is printed
         as shared host memory.

So a gfx1151 whose `mem_info_vram_total` IS the full 64 GiB window -- torch and
sysfs agreeing exactly, which is the healthiest possible reading -- renders as
`0 GiB VRAM + 64 GiB shared`. The head publishes the figure as a measured 0.0
instead, which flips `hostBackedKnown` to true, moves all 64 GiB into reserved,
and the tile reads `64 GiB`.

What makes this criteria module non-vacuous, and what it therefore gates on:

  REACHED   the branch the PR changed has to be the branch this machine takes.
            That is a statement about the INPUTS -- sysfs total readable and
            positive, torch total not exceeding it by more than 10% -- and it is
            checked against the raw sysfs bytes and torch `total_memory` the
            probe read for itself, not against the function's output. A host in
            the excess regime (a real GTT-backed overhang, WSL, a discrete card)
            never reaches the changed line, and a differential run there would
            be measuring nothing while looking identical.

  THE PART  gfx1151, unified memory, one visible GPU. `gfx_target_version`
            110501 off KFD.

  THE RULE  the frontend files the tile rendering is derived from must be
            byte-identical at base and head. This PR touches no frontend file,
            so any difference means the rendering is not a constant across the
            comparison and the two tile strings are not comparable.

The verdict is then read off the RENDERED TOTALS, not off the device dict: the
device dict differing is what the PR does, and grading a PR on whether it did
what it says it does is circular. The question is whether the number a human
reads changes, and in which direction.

Pairs with probes/studio_apu_host_backed_probe.py.
"""

from __future__ import annotations

TITLE = ("unsloth#11451: a Linux ROCm APU whose sysfs total matches torch reads "
         "`0 GiB VRAM + 64 GiB shared` in Settings > System")
MODE = "differential"

# Authored, and deliberately wider than this host. The changed function is gated
# on `_rocm_known_unified`, so a DISCRETE ROCm card reaching it is a case this
# host cannot produce; the sibling branch three lines down is the Windows ROCm
# WDDM one, which this PR leaves alone and which a reader will ask about; and the
# `shared_memory` narrowing is argued in the diff on a two-socket MI300A, which
# is multi-GPU and partitionable. None of those are reachable on one Strix Halo.
NEEDS = ["rocm", "gpu", "integrated_gpu", "linux",
         "windows", "windows_rocm_wddm", "discrete_gpu",
         "multi_gpu", "gpu_partitions"]

# Fraction of the torch total by which sysfs may be exceeded before the change
# stops applying. Not tunable here for a reason: it is the literal constant in
# `_rocm_linux_shared_pool_host_gb_by_index`, and a criteria module that picked
# its own would be gating on a different predicate than the one under test.
EXCESS_BAND = 0.1

# GiB. Below this the device is not a plausible APU window and the reading is
# vacuous: 0 dedicated out of 0 total is not the defect, it is an empty read.
MIN_TOTAL_GIB = 1.0

GFX1151_TARGET_VERSION = "110501"


# ── helpers: all pure reads of the observation, no judgement ─────────────────

def _device(obs: dict) -> dict:
    devices = obs.get("devices") or []
    return devices[0] if devices else {}


def _totals(obs: dict) -> dict:
    """Prefer the checkout's own TypeScript when it ran; fall back to the port.

    Recorded separately in the probe precisely so this choice is visible. The
    gate below fails if both ran and disagreed, so the fallback is never a way
    of preferring the answer that suits."""
    render = obs.get("render") or {}
    ts = (render.get("typescript") or {}).get("totals")
    if isinstance(ts, dict):
        return ts
    port = render.get("totals_python_port") or {}
    return port.get("value") if isinstance(port, dict) and "value" in port else {}


def _labels(obs: dict) -> dict:
    render = obs.get("render") or {}
    return render.get("labels_typescript") or render.get("labels_python_port") or {}


def _sysfs_total_bytes(obs: dict):
    """The largest `mem_info_vram_total` any DRM card reported. One visible GPU is
    gated below, so "largest" and "the card's" coincide; taking a max rather than
    an index avoids inventing a join this probe does not need."""
    raw = ((obs.get("raw") or {}).get("sysfs_cards") or {}).get("value") or []
    totals = [c.get("mem_info_vram_total") for c in raw
              if isinstance(c.get("mem_info_vram_total"), int)
              and c.get("mem_info_vram_total") > 0]
    return max(totals) if totals else None


def _torch_total_bytes(obs: dict):
    devs = (obs.get("torch") or {}).get("devices") or []
    if not devs:
        return None
    props = (devs[0].get("properties") or {})
    if not props.get("ok"):
        return None
    return (props.get("value") or {}).get("total_memory_bytes")


def _gfx_target_versions(obs: dict) -> list:
    nodes = ((obs.get("raw") or {}).get("kfd_nodes") or {}).get("value") or []
    return [n.get("gfx_target_version") for n in nodes if n.get("gfx_target_version")]


def _frontend_hashes(obs: dict) -> dict:
    return {k: (v or {}).get("sha256")
            for k, v in (obs.get("frontend_sources") or {}).items()}


def _fmt_gib(value) -> str:
    return f"{value / (1024 ** 3):.3f}" if isinstance(value, (int, float)) else "n/a"


# ── gates: is there anything here to compare? ────────────────────────────────

def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = [s for s in ("base", "head", "merge") if obs.get(s)]

    for name in states:
        o = obs[name]
        dev = _device(o)

        # The probe ran at all.
        rc = o.get("_probe_rc")
        backend = o.get("backend") or {}
        err = backend.get("import_error") or (
            backend.get("get_backend_visible_gpu_info") or {}).get("error")
        out.append((f"{name}: backend answered",
                    bool(o.get("devices")) and not err,
                    f"probe rc={rc}, {len(o.get('devices') or [])} device(s)"
                    + (f", error: {str(err)[:200]}" if err else "")))

        # Exactly one visible GPU, and it is the unified-memory part. Two devices
        # would make "the card's sysfs total" an unjoined guess.
        n = len(o.get("devices") or [])
        out.append((f"{name}: one visible unified-memory GPU",
                    n == 1 and dev.get("unified_memory") is True,
                    f"{n} device(s); unified_memory={dev.get('unified_memory')!r}; "
                    f"name={dev.get('name')!r}"))

        total = dev.get("memory_total_gb")
        out.append((f"{name}: the device reports a real budget",
                    isinstance(total, (int, float)) and total >= MIN_TOTAL_GIB,
                    f"memory_total_gb={total!r}"))

        # The part. A gfx1151 is what the defect was reported on and what the
        # comment in the diff cites; another APU would be a different machine.
        versions = _gfx_target_versions(o)
        out.append((f"{name}: KFD says gfx1151",
                    GFX1151_TARGET_VERSION in versions,
                    f"gfx_target_version(s) {versions or 'none read'}; expected "
                    f"{GFX1151_TARGET_VERSION}"))

        # REACHED. Computed from the raw bytes the probe read itself.
        sysfs_b = _sysfs_total_bytes(o)
        torch_b = _torch_total_bytes(o)
        if sysfs_b and torch_b:
            excess = torch_b - sysfs_b
            reached = sysfs_b > 0 and excess <= EXCESS_BAND * torch_b
            detail = (f"torch total_memory {torch_b} B ({_fmt_gib(torch_b)} GiB), "
                      f"sysfs mem_info_vram_total {sysfs_b} B ({_fmt_gib(sysfs_b)} GiB), "
                      f"excess {excess} B = {100.0 * excess / torch_b:.3f}% of torch "
                      f"(band {EXCESS_BAND:.0%})")
        else:
            reached, detail = False, (
                f"could not read both inputs: sysfs mem_info_vram_total={sysfs_b!r}, "
                f"torch total_memory={torch_b!r}")
        out.append((f"{name}: the changed branch is the one this machine takes",
                    reached, detail))

        # The port is only faithful to a revision. If node ran, it must agree.
        render = o.get("render") or {}
        ts = render.get("typescript") or {}
        matched = render.get("port_matches_typescript")
        if ts.get("ok"):
            # A real disagreement means one of the two renderings is wrong and the
            # quoted tile string cannot be trusted. That FAILS.
            out.append((f"{name}: the port agrees with the real gpu-vram.ts",
                        matched is True,
                        f"node {ts.get('node_version')}: port={render.get('totals_python_port')}, "
                        f"typescript={ts.get('totals')}"))
        else:
            # node absent is NOT a skip of the thing under test: the defect is
            # fully determined by the device dict, and the rendering is a
            # derivation of it. It IS an unverified quote, so it is surfaced here
            # and in the table rather than passing quietly.
            out.append((f"{name}: tile rendering cross-checked against gpu-vram.ts",
                        True,
                        "NOT CROSS-CHECKED ON THIS HOST: "
                        + str(ts.get("error") or f"node rc={ts.get('rc')}")
                        + ". The quoted tile strings come from the Python port of "
                          "gpuMemoryTotalsGb; the real TypeScript did not run here"))

    # The rendering rule must be a CONSTANT across the comparison.
    if obs.get("base") and obs.get("head"):
        hb, hh = _frontend_hashes(obs["base"]), _frontend_hashes(obs["head"])
        same = hb == hh and all(v for v in hb.values())
        differing = sorted(k for k in hb if hb.get(k) != hh.get(k))
        out.append(("the frontend rendering rule is identical at base and head",
                    same,
                    "all four frontend sources byte-identical" if same else
                    f"differs or unread: {differing or list(hb)}"))
    return out


# ── the comparison ───────────────────────────────────────────────────────────

def _row(name: str, obs: dict) -> str:
    o = obs.get(name)
    if not o:
        return ""
    dev = _device(o)
    totals = _totals(o)
    labels = _labels(o)
    host_backed = dev.get("shared_memory_host_backed_gb")
    present = dev.get("shared_memory_host_backed_gb_key_present")
    is_null = dev.get("shared_memory_host_backed_gb_is_null")
    if not present:
        shown = "ABSENT (key not published)"
    elif is_null:
        shown = "`null` (unknown)"
    else:
        shown = f"`{host_backed}` (measured)"
    internals = ((o.get("backend") or {}).get("internals") or {})
    indices = internals.get("shared_pool_indices_present")
    return (f"| {name} "
            f"| {dev.get('memory_total_gb')} "
            f"| {dev.get('shared_memory')!r} "
            f"| {shown} "
            f"| {dev.get('unified_memory')!r} "
            f"| {indices if indices is not None else 'n/a'} "
            f"| {totals.get('dedicated')} "
            f"| {totals.get('shared')} "
            f"| `{labels.get('resources_tab', '?')}` "
            f"| `{labels.get('two_dp', '?')}` |")


def table(obs: dict) -> str:
    rows = [
        "| state | memory_total_gb | shared_memory | shared_memory_host_backed_gb "
        "| unified_memory | host-gb map indices | tile dedicated | tile shared "
        "| Settings tile | same, 2dp |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ("base", "head", "merge"):
        row = _row(name, obs)
        if row:
            rows.append(row)

    notes = []
    base = obs.get("base") or {}
    sysfs_b, torch_b = _sysfs_total_bytes(base), _torch_total_bytes(base)
    if sysfs_b and torch_b:
        notes.append(
            f"Raw inputs at the base: torch `total_memory` {torch_b} B "
            f"({_fmt_gib(torch_b)} GiB), sysfs `mem_info_vram_total` {sysfs_b} B "
            f"({_fmt_gib(sysfs_b)} GiB), excess {torch_b - sysfs_b} B. The changed "
            f"predicate publishes 0.0 exactly when that excess is within "
            f"{EXCESS_BAND:.0%} of the torch total.")
    cards = ((base.get("raw") or {}).get("sysfs_cards") or {}).get("value") or []
    for c in cards:
        if c.get("mem_info_vram_total"):
            notes.append(
                f"DRM sysfs `{c.get('pci_id')}`: vram_total {c.get('mem_info_vram_total')} B, "
                f"vram_used {c.get('mem_info_vram_used')} B, "
                f"gtt_total {c.get('mem_info_gtt_total')} B, "
                f"gtt_used {c.get('mem_info_gtt_used')} B.")
    # Is the carve-out BESIDE host RAM or INSIDE it? Reported, not gated. The
    # verdict answers "did the tile stop saying the machine has no VRAM"; this
    # answers "and is the new reading true of this hardware", which is a
    # different question and must not be smuggled into the first one by a gate.
    mem = ((base.get("raw") or {}).get("proc_meminfo") or {}).get("value") or {}
    mem_total = mem.get("MemTotal_bytes")
    phys = ((base.get("raw") or {}).get("physical_memory") or {}).get("value") or {}
    installed = phys.get("sysfs_total_bytes")
    if mem_total and sysfs_b:
        combined = mem_total + sysfs_b
        notes.append(
            f"Carve-out shape. `/proc/meminfo` `{mem.get('raw_lines', {}).get('MemTotal', '?')}` "
            f"= {mem_total} B ({_fmt_gib(mem_total)} GiB); sysfs `mem_info_vram_total` "
            f"{sysfs_b} B ({_fmt_gib(sysfs_b)} GiB); the two sum to {combined} B "
            f"({_fmt_gib(combined)} GiB). Installed capacity from "
            f"`/sys/devices/system/memory` = {installed!r} B "
            f"({_fmt_gib(installed)} GiB). If the sum lands at the installed capacity "
            f"the VRAM bar is DISJOINT from host RAM and 'the whole torch budget is "
            f"the dedicated heap' is true of this machine; if `MemTotal` alone is "
            f"already the installed capacity, the two OVERLAP and the head's reading "
            f"is a different wrong number rather than a right one. This row does not "
            f"enter the verdict.")
    render = (obs.get("head") or {}).get("render") or {}
    ts = render.get("typescript") or {}
    notes.append(
        "Tile strings rendered by the checkout's own `gpu-vram.ts` under node "
        f"{ts.get('node_version')}."
        if ts.get("ok") else
        "Tile strings rendered by the probe's port of `gpuMemoryTotalsGb`; node was "
        "not available on this runner, so the real TypeScript did not execute here. "
        "The device dicts in the table are raw backend output and are unaffected.")
    notes.append(
        "`Settings tile` is resources-tab.tsx's own `formatGiB`, which trims "
        "trailing zeros, so an exact 64 prints as `64 GiB`. The 2dp column is the "
        "same totals without the trim.")
    return "\n".join(rows) + "\n\n" + "\n\n".join(notes)


def base_shows_defect(base: dict):
    """The tile claims the machine has no dedicated VRAM and the whole window is
    host memory."""
    dev = _device(base)
    totals = _totals(base)
    total = dev.get("memory_total_gb")
    if not isinstance(total, (int, float)) or total < MIN_TOTAL_GIB:
        return False, "no usable device total at the base"
    dedicated, shared = totals.get("dedicated"), totals.get("shared")
    unknown = (not dev.get("shared_memory_host_backed_gb_key_present")
               or dev.get("shared_memory_host_backed_gb_is_null"))
    shows = (unknown
             and dedicated == 0
             and abs((shared or 0) - total) < 0.011)
    return shows, (f"host-backed figure unknown={unknown}, tile dedicated={dedicated}, "
                   f"tile shared={shared}, device total={total}")


def head_is_fixed(state: dict):
    """The figure is published as a measured number, `shared_memory` is not
    asserted for a pool with no host-backed part, and the tile shows the window
    as dedicated."""
    dev = _device(state)
    totals = _totals(state)
    total = dev.get("memory_total_gb")
    host_backed = dev.get("shared_memory_host_backed_gb")
    if not isinstance(total, (int, float)) or total < MIN_TOTAL_GIB:
        return False, "no usable device total"
    published = (dev.get("shared_memory_host_backed_gb_key_present")
                 and not dev.get("shared_memory_host_backed_gb_is_null")
                 and isinstance(host_backed, (int, float))
                 and host_backed == 0)
    # The narrowing the diff argues for: nothing host-backed, so nothing shared.
    not_shared = dev.get("shared_memory") is not True
    dedicated, shared = totals.get("dedicated"), totals.get("shared")
    fixed = (published and not_shared
             and shared == 0
             and abs((dedicated or 0) - total) < 0.011)
    return fixed, (f"host-backed published as {host_backed!r}, "
                   f"shared_memory={dev.get('shared_memory')!r}, tile dedicated={dedicated}, "
                   f"tile shared={shared}, device total={total}")
