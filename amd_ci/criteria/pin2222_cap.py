#!/usr/bin/env python3
"""Criteria: unsloth-zoo PR 2222, the counted pinned cap bounds every pinned byte the host allocator holds.

Defect: with torch 2.13+ host allocator settings (pinned_max_cached_size_mb / pinned_max_round_threshold_mb),
the offload pool's pinned reservations let torch.cuda.host_memory_stats() allocated_bytes.peak go over
UNSLOTH_GC_PINNED_MAX_GB in one of the scenarios (foreign pin freed, reused larger cached block, two
threads). Fixed: no scenario's peak exceeds the cap, and the head still pinned at least one of its own
buffers in each scenario (refusing every pin would be a vacuous pass).

Pairs with probes/pin2222_probe.py. Gates make the run INCONCLUSIVE when the host cannot answer: ROCm torch
older than 2.13, host_memory_stats missing, or the knobs not honoured by this ROCm build.
"""

from __future__ import annotations

TITLE = "unsloth-zoo PR 2222: real pinned host memory vs the counted cap on gfx1151 (ROCm)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "nvidia", "discrete_gpu"]
MIB = 1 << 20
SCEN = ("1", "1b", "2", "3")
EXPECTED_SCEN = ("1", "2", "3")


def _tv(v: str | None) -> tuple[int, int]:
    try:
        a, b = str(v).split("+")[0].split(".")[:2]
        return int(a), int(b)
    except Exception:  # noqa: BLE001
        return (0, 0)


def _knob_facts(o: dict) -> list[tuple[str, bool, str]]:
    k = o.get("knobs") or {}
    rd, r1 = k.get("round_default") or {}, k.get("round_1") or {}
    cd, c4 = k.get("cached_default") or {}, k.get("cached_4") or {}
    out = []
    out.append(("torch >= 2.13 ROCm build on gfx1151",
                _tv(rd.get("torch")) >= (2, 13) and bool(rd.get("hip")) and str(rd.get("arch", "")).startswith("gfx1151"),
                f"torch={rd.get('torch')} hip={rd.get('hip')} arch={rd.get('arch')} err={rd.get('error')}"))
    out.append(("_accelerator_getAllocatorSettings present and reflects PYTORCH_ALLOC_CONF",
                bool(r1.get("has_getter")) and "pinned_max_round_threshold_mb" in str(r1.get("settings"))
                and "pinned_max_cached_size_mb" in str(c4.get("settings")),
                f"round_1 settings={r1.get('settings')!r}; cached_4 settings={c4.get('settings')!r}"))
    out.append(("host_memory_stats counts a pinned block",
                (rd.get("held_after_3mib") or 0) >= 3 * MIB and bool(rd.get("pinned")),
                f"default 3 MiB pin -> held {rd.get('held_after_3mib')} B, keys {rd.get('stats_keys')}"))
    out.append(("pinned_max_round_threshold_mb honoured (3 MiB pins 3 MiB, default rounds to 4)",
                r1.get("held_after_3mib") == 3 * MIB and rd.get("held_after_3mib") == 4 * MIB,
                f"threshold 1: {r1.get('held_after_3mib')} B; default: {rd.get('held_after_3mib')} B"))
    out.append(("pinned_max_cached_size_mb honoured (freed 6 MiB block returns to the OS)",
                (c4.get("held_after_free_6mib") or 1 << 40) < 2 * MIB and (cd.get("held_after_free_6mib") or 0) >= 6 * MIB,
                f"cached 4: {c4.get('held_after_free_6mib')} B after free; default: {cd.get('held_after_free_6mib')} B"))
    return out


def _by(o: dict) -> dict:
    return {(r.get("arm"), r.get("scenario")): r for r in o.get("results") or []}


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    head = obs.get("head") or {}
    for name in ("base", "head"):
        o = obs.get(name) or {}
        out.append((f"{name} probe set up its venv", not o.get("setup_error") and bool(o.get("knobs")),
                    f"venv={o.get('venv')} err={str(o.get('setup_error', ''))[:200]}"))
    out += _knob_facts(head)
    for name in [n for n in obs if not n.startswith("_")]:
        o = obs[name]
        for (arm, s), r in _by(o).items():
            ok = "error" not in r and str(r.get("zoo_file", "")).startswith(str(r.get("arm_dir", "?")))
            if not ok:
                out.append((f"{arm} scenario {s} ran its own checkout", False,
                            f"{str(r.get('error', ''))[-300:]} file={r.get('zoo_file')}"))
        missing = [f"{a}:{s}" for a in (o.get("arms") or {}) for s in SCEN if (a, s) not in _by(o)]
        out.append((f"{name}: every arm x scenario produced a result", not missing and bool(o.get("results")),
                    f"missing={missing} arm_errors={o.get('arm_errors')}"))
    out.append(("intermediate arms ran at the head state", len(head.get("arms") or {}) >= 3,
                f"arms={list((head.get('arms') or {}).keys())}"))
    return out


def table(obs: dict) -> str:
    rows = []
    arms = []
    res = {}
    for name in [n for n in obs if not n.startswith("_")]:
        for (arm, s), r in _by(obs[name]).items():
            if arm not in arms: arms.append(arm)
            res[(arm, s)] = r
    rows.append("| scenario | cap MiB | " + " | ".join(arms) + " |")
    rows.append("|---|---|" + "---|" * len(arms))
    for s in SCEN:
        cap = next((r.get("cap_mib") for (a, ss), r in res.items() if ss == s and r.get("cap_mib")), "?")
        cells = []
        for a in arms:
            r = res.get((a, s))
            if not r: cells.append("-"); continue
            if "error" in r: cells.append("ERROR"); continue
            cells.append(f"{'OVER' if r['over_cap'] else 'ok'} peak {r['peak_mib']} dec {r['decisions']}")
        rows.append(f"| {s} | {cap} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _over(o: dict) -> list[str]:
    st = o.get("state")
    return [s for (a, s), r in _by(o).items() if a == st and s in EXPECTED_SCEN and r.get("over_cap")]


def base_shows_defect(base: dict):
    over = _over(base)
    return (bool(over), f"over cap in scenarios {over}")


def head_is_fixed(head: dict):
    st = head.get("state")
    mine = {s: r for (a, s), r in _by(head).items() if a == st}
    over = [s for s, r in mine.items() if r.get("over_cap") or "error" in r]
    vacuous = [s for s, r in mine.items() if not any(r.get("decisions") or [])]
    ok = not over and not vacuous and all(s in mine for s in SCEN)
    return (ok, f"over/error {over}, pinned nothing {vacuous}")
