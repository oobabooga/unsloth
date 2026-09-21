#!/usr/bin/env python3
"""Criteria: does the Settings > System VRAM readout describe this APU?

unsloth#7449 defect 1, "VRAM readout wrong on an APU". The reporter is on
Windows 11 with a Strix Halo (Radeon 8060S, gfx1151, 128 GB unified memory).
Three PRs have been merged at it: #8863, #9314 and #11366.

#11366's evidence table carries the row "Windows APU (flag already set):
unchanged". Read against the code that row is true and narrow: `shared_memory`
is set at `hardware.py:2393` as `known_unified and platform.system() ==
"Windows"`, that line came from #9242, and #11366 changed four FRONTEND files
and no backend file at all. So the boolean was indeed already set on Windows.

The row says nothing about the numbers, and the numbers are the defect. Two
things have to be true for the readout to describe the machine, and neither is
implied by the flag:

  TRACKS   the `used` the tile shows has to MOVE when the GPU is loaded. On
           Windows the numerator is a WDDM cooked counter, and `Dedicated Usage`
           saturates at the BIOS carve-out while the overflow lands in `Shared`
           (measured on a gfx1151: holding 64 GiB reads 31.637 GB dedicated and
           33.887 GB shared). Paired with a pool-scoped total, dedicated alone
           reports a loaded 89 GB device as 57 GB free, which is the direction
           that OOMs. Idle, the correct and the broken readings are identical,
           which is why the probe holds memory and this criterion is written on
           the DELTA rather than on an absolute.

  SPLIT    `shared_memory_host_backed_gb` has to be a number. `gpu-vram.ts`
           splits the dedicated and shared pools on it, and when it is None the
           frontend takes the whole pool as host-backed, so the dedicated bucket
           collapses to zero and a 512 MiB-carve-out box and a 32 GiB-carve-out
           box print the identical "0 GiB VRAM + <pool> shared". On Windows that
           figure comes from the DirectX registry via
           `_windows_rocm_shared_pool_host_gb_by_index`, whose map is
           all-or-nothing: one unreadable adapter subkey empties it.

A state is faithful only if BOTH hold. Either one alone is the defect.

Pairs with probes/studio_settings_vram_probe.py.
"""

from __future__ import annotations

GIB = 1024 ** 3

TITLE = "unsloth#7449 defect 1: the Settings > System VRAM readout on a gfx1151 APU"
MODE = "differential"

# Authored, and deliberately wider than this host. The change is an APU VRAM
# readout on ROCm; the report is Windows, so the WDDM counter path matters; and
# the same functions carry a discrete-GPU branch (`_rocm_props_total_is_carve_out`
# fails OPEN, so an unclassified discrete card reaches them) and a multi-GPU
# branch (`_match_adapter_used_to_devices` only emits when capacity FORCES a
# pairing, which one visible GPU never does). Under-declaring here would render
# a report that bounds nothing.
NEEDS = ["rocm", "gpu", "integrated_gpu", "windows", "windows_rocm_wddm",
         "discrete_gpu", "multi_gpu"]

# GiB. The reported `used` is a SYSTEM-WIDE figure -- on Windows the WDDM
# Dedicated + Shared cooked counters, on Linux DRM sysfs vram_used + gtt_used --
# so it also carries the desktop compositor, the driver's own allocations and
# any co-tenant, while the hold is this process's touched bytes alone. The two
# never agree exactly. The spread is bounded by the measured gfx1151 case that
# motivated the Dedicated+Shared pairing: holding 64 GiB (68.72 GB) the counters
# summed to 65.52 GB, about 3.0 GiB of mixed GB/GiB and compositor accounting.
# 4.0 sits above that spread and far below the 8 GiB default hold, so the defect
# this is written against -- `used` plateauing at the carve-out while the
# allocation lands in the shared segment -- cannot squeeze underneath it.
# Widening this number is the easiest way to fake the result, which is why it is
# argued here rather than tuned at the call site.
USED_TRACKS_TOLERANCE_GIB = 4.0

# GiB. Below this the `used` reading is vacuous: an idle APU and a plateaued one
# report the same thing, so a hold that did not take leaves nothing to compare
# and the gate FAILS rather than skipping.
MIN_HOLD_GIB = 1.0

STATES = ("base", "head", "merge")


# ── accessors ─────────────────────────────────────────────────────────────────

def _env(st: dict) -> dict:
    return st.get("env") or {}


def _cls(st: dict) -> dict:
    return st.get("classify") or {}


def _readout(st: dict, which: str) -> dict:
    return st.get("readout_" + which) or {}


def _settings(st: dict, which: str) -> dict:
    return _readout(st, which).get("settings") or {}


def _gt(st: dict, which: str) -> dict:
    return st.get("ground_truth_" + which) or {}


def _wrapped(bucket: dict, name: str):
    """Unwrap the probe's call() envelope; None when it recorded an error."""
    entry = bucket.get(name)
    if isinstance(entry, dict) and "ok" in entry:
        return entry.get("value")
    return entry


def _arch(st: dict) -> str:
    arch = _cls(st).get("props_gcnArchName") or ""
    if arch:
        return str(arch)
    classified = _wrapped(_cls(st), "classify_unified_memory") or []
    return str(classified[0]) if classified else ""


def _hold_gib(st: dict):
    held = (st.get("hold") or {}).get("held_bytes")
    return None if held is None else held / GIB


def _gt_used_bytes(gt: dict):
    """Ground-truth occupancy: dedicated plus shared, or None when neither
    counter answered. A counter that declined is never read as zero -- "the
    counter said nothing" and "the GPU holds nothing" are the two answers this
    whole measurement exists to separate."""
    parts = [gt.get(k) for k in ("dedicated_used_bytes", "shared_used_bytes")
             if gt.get(k) is not None]
    return sum(parts) if parts else None


def _gt_delta_gib(st: dict):
    idle = _gt_used_bytes(_gt(st, "idle"))
    held = _gt_used_bytes(_gt(st, "held"))
    if idle is None or held is None:
        return None
    return (held - idle) / GIB


def _reported_delta_gib(st: dict):
    idle = _settings(st, "idle").get("used_gb")
    held = _settings(st, "held").get("used_gb")
    if idle is None or held is None:
        return None
    return float(held) - float(idle)


def _gib(value):
    return None if value is None else round(value / GIB, 2)


def _fmt(value, suffix: str = ""):
    if value is None:
        return "None"
    if isinstance(value, float):
        return "{0:.2f}{1}".format(value, suffix)
    return "{0}{1}".format(value, suffix)


# ── the two questions ─────────────────────────────────────────────────────────

def _tracks(st: dict):
    """(True / False / None, evidence). None means it could not be asked."""
    reported = _reported_delta_gib(st)
    truth = _gt_delta_gib(st)
    if reported is None or truth is None:
        return None, ("reported used delta = {0}, ground-truth delta = {1} "
                      "(one of them was unreadable, so tracking cannot be asked)"
                      .format(_fmt(reported, " GiB"), _fmt(truth, " GiB")))
    ok = abs(reported - truth) <= USED_TRACKS_TOLERANCE_GIB
    return ok, ("reported used moved {0} while the machine moved {1} across a "
                "{2} hold; |difference| {3} the {4:.1f} GiB tolerance"
                .format(_fmt(reported, " GiB"), _fmt(truth, " GiB"),
                        _fmt(_hold_gib(st), " GiB"), "within" if ok else "EXCEEDS",
                        USED_TRACKS_TOLERANCE_GIB))


def _split_is_real(st: dict):
    """(True / False, evidence). A None host-backed figure collapses the whole
    pool into 'shared' and prints the same label whatever the carve-out is."""
    held = _settings(st, "held")
    known = bool(held.get("host_backed_known"))
    return known, ("shared_memory_host_backed_gb = {0}, so the label splits "
                   "{1} GiB dedicated + {2} GiB shared out of {3} GiB"
                   .format(_fmt(held.get("host_backed_gb")),
                           _fmt(held.get("dedicated_gb")),
                           _fmt(held.get("shared_gb")),
                           _fmt(held.get("total_gb"))))


def _faithful(st: dict):
    tracks, tracks_why = _tracks(st)
    split, split_why = _split_is_real(st)
    return tracks, split, tracks_why + "; " + split_why


# ── gates: these FAIL, they never skip ────────────────────────────────────────

def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    for name in STATES:
        st = obs.get(name)
        if not st:
            continue
        env, cls = _env(st), _cls(st)

        out.append((
            "{0} probe returned a reading".format(name),
            not st.get("child_failed") and not st.get("timed_out"),
            "rc={0}; {1}".format(
                st.get("rc"),
                str(env.get("torch_error") or st.get("hardware_import_error")
                    or st.get("stderr_tail", ""))[-180:])))

        out.append((
            "{0} torch is a ROCm build".format(name),
            bool(env.get("torch_version_hip")),
            "torch={0} hip={1} cuda={2}".format(env.get("torch_version"),
                                                env.get("torch_version_hip"),
                                                env.get("torch_version_cuda"))))

        arch = _arch(st)
        out.append((
            "{0} the device is the gfx1151 APU".format(name),
            "gfx1151" in arch,
            "{0} arch={1} is_integrated={2} positively_unified={3}".format(
                cls.get("props_name"), arch or "unreadable",
                cls.get("props_is_integrated"),
                _wrapped(cls, "positively_unified"))))

        # Without a hold the used reading is the same number whether the counter
        # tracks the allocation or plateaus at the carve-out, so a comparison
        # made against an idle GPU proves nothing at all.
        hold = st.get("hold") or {}
        held_gib = _hold_gib(st)
        out.append((
            "{0} the hold actually held >= {1:.0f} GiB".format(name, MIN_HOLD_GIB),
            bool(hold.get("held")) and held_gib is not None and held_gib >= MIN_HOLD_GIB,
            "requested {0} GiB, held {1}; {2}".format(
                hold.get("requested_gib"), _fmt(held_gib, " GiB"),
                hold.get("error") or hold.get("skipped") or "no error")))

        # The comparison has a machine-side arm only if this OS answered.
        gt_held = _gt(st, "held")
        out.append((
            "{0} ground truth was readable on this OS".format(name),
            _gt_used_bytes(gt_held) is not None and _gt_delta_gib(st) is not None,
            "source={0} dedicated={1} shared={2} delta={3}; errors={4}".format(
                gt_held.get("source"),
                _fmt(_gib(gt_held.get("dedicated_used_bytes")), " GiB"),
                _fmt(_gib(gt_held.get("shared_used_bytes")), " GiB"),
                _fmt(_gt_delta_gib(st), " GiB"),
                str(gt_held.get("errors") or [])[:160])))

        # A label at BOTH instants, since the criterion is a delta.
        out.append((
            "{0} the readout produced a Settings label idle and held".format(name),
            bool(_settings(st, "idle").get("label")) and bool(_settings(st, "held").get("label")),
            "idle={0!r} held={1!r}; {2}".format(
                _settings(st, "idle").get("label"), _settings(st, "held").get("label"),
                str(_settings(st, "held").get("error") or "")[:160])))
    return out


# ── the report table ──────────────────────────────────────────────────────────

def table(obs: dict) -> str:
    rows = [
        "| state | device | arch | IS_ROCM | props.total_memory | mem_get_info total | "
        "memory_total_gb | shared_memory | unified_memory | host_backed_gb | used | free | "
        "Settings label | GT dedicated | GT shared | GT host avail |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in STATES:
        st = obs.get(name)
        if not st:
            continue
        env, cls = _env(st), _cls(st)
        rd, sett, gt = _readout(st, "held"), _settings(st, "held"), _gt(st, "held")
        mgi = _wrapped(rd, "mem_get_info") or []
        rows.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} | "
            "`{12}` | {13} | {14} | {15} |".format(
                name,
                cls.get("props_name"),
                _arch(st) or None,
                _wrapped(env, "hw_IS_ROCM"),
                _fmt(_gib(cls.get("props_total_memory")), " GiB"),
                _fmt(_gib(mgi[1]) if len(mgi) > 1 else None, " GiB"),
                _fmt(sett.get("memory_total_gb")),
                sett.get("shared_memory"),
                sett.get("unified_memory"),
                _fmt(sett.get("host_backed_gb")),
                sett.get("used_label") or _fmt(sett.get("used_gb")),
                _fmt(sett.get("free_gb")),
                sett.get("label"),
                _fmt(_gib(gt.get("dedicated_used_bytes")), " GiB"),
                _fmt(_gib(gt.get("shared_used_bytes")), " GiB"),
                _fmt(_gib(gt.get("host_available_bytes")), " GiB")))

    rows += ["",
             "Held-minus-idle, which is what the verdict is measured on:",
             "",
             "| state | hold | reported used moved | the machine moved | tracks |",
             "|---|---|---|---|---|"]
    for name in STATES:
        st = obs.get(name)
        if not st:
            continue
        tracks, _why = _tracks(st)
        rows.append("| {0} | {1} | {2} | {3} | {4} |".format(
            name, _fmt(_hold_gib(st), " GiB"),
            _fmt(_reported_delta_gib(st), " GiB"), _fmt(_gt_delta_gib(st), " GiB"),
            "yes" if tracks else ("NO" if tracks is False else "unaskable")))

    rows += ["",
             "`/api/system/hardware` walks a SEPARATE path (`get_gpu_summary` -> "
             "`get_gpu_memory_info`) and disagrees with the tile above on an APU by "
             "construction, so both are quoted:"]
    for name in STATES:
        st = obs.get(name)
        if not st:
            continue
        rd = _readout(st, "held")
        rows.append("- `{0}` About tab: `{1}`".format(
            name, str(_wrapped(rd, "gpu_summary") or (rd.get("gpu_summary") or {}))[:400]))
    return "\n".join(rows)


# ── the verdict ───────────────────────────────────────────────────────────────

def base_shows_defect(base: dict) -> tuple[bool, str]:
    tracks, split, why = _faithful(base)
    if tracks is None:
        # Not a reproduction. differential.py turns this into VOID, which is the
        # right answer: an unreadable base arm proves nothing either way.
        return False, "the base state could not be asked whether used tracks occupancy -- " + why
    defect = (tracks is False) or (split is False)
    return defect, ("the base readout {0} describe the machine: {1}"
                    .format("does NOT" if defect else "does", why))


def head_is_fixed(head: dict) -> tuple[bool, str]:
    tracks, split, why = _faithful(head)
    fixed = (tracks is True) and (split is True)
    return fixed, ("at the head the readout {0} both arms: {1}"
                   .format("satisfies" if fixed else "still fails", why))
