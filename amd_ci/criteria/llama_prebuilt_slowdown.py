#!/usr/bin/env python3
"""Criteria: does unsloth#7371's throughput collapse reproduce on this gfx1151?

Differential, with the states being RELEASE TAGS:

    base -> the prebuilt #7371 calls slow (b10079)
    head -> the prebuilt a user gets today

The defect is "this build is much slower than the reference build the report
calls fast", where the reference is measured by the SAME probe, on the SAME
host, inside the SAME state. Each state therefore carries its own control, and
the two states' control readings are compared to each other before any verdict
is drawn: if the reference moves between states the host was not quiet and the
whole comparison is discarded as INCONCLUSIVE rather than rescued.

The VOID rule does the real work here. #7371 has a comment claiming the problem
is already gone; if b10079 is NOT slow on this host, that is the answer, and it
arrives as VOID rather than as a green tick on the head.

Pairs with probes/llama_prebuilt_throughput_probe.py.
"""

from __future__ import annotations

TITLE = "unsloth#7371: llama.cpp prebuilt throughput on gfx1151"
MODE = "differential"

# Authored, not derived. The change under examination is a ROCm llama.cpp
# prebuilt on an AMD APU, and the report is from Windows on the same chip.
NEEDS = ["rocm", "gpu", "integrated_gpu", "windows", "discrete_gpu", "multi_gpu"]

# #7371 reports 39 -> 11 tok/s, a 72% loss. Anything at or past a third of the
# reference is far outside run-to-run noise on this box and is what "severe
# regression" has to mean to be worth acting on.
DEFECT_RATIO = 0.67
# The head clears it when it is within 10% of the reference.
FIXED_RATIO = 0.90
# The negative control: the reference build's own reading must agree between
# states this closely, or nothing measured here is comparable.
CONTROL_TOLERANCE_PCT = 12.0
# Within one state, repeated generations must not scatter more than this.
SPREAD_TOLERANCE_PCT = 25.0


def _ref(o: dict) -> dict:
    return o.get("reference") or {}


def _own(o: dict) -> dict:
    return o.get("measured") or {}


def _ratio(state: dict, key: str):
    ref = _ref(state).get(key)
    own = _own(state).get(key)
    if not ref or not own:
        return None
    return own / ref


def _gen_spread(build: dict, label: str = "topk40"):
    return ((build.get("generation") or {}).get(label) or {}).get("tg_ts_spread_pct")


def _on_gpu(build: dict) -> tuple[bool, str]:
    """The weights have to be on the AMD device, or this measures the CPU backend.

    `system_info` names ROCm even on a CPU-only run, so read where the layers
    were actually assigned instead. That is the same signal #9792 counted its
    device splits from.
    """
    devs = dict(build.get("layer_devices") or {})
    if not devs:
        return False, "no `load_tensors: layer N assigned to device ...` line was emitted"
    accel = {d: n for d, n in devs.items() if not d.upper().startswith("CPU")}
    if not accel:
        return False, f"every layer was assigned to the CPU backend: {devs}"
    return True, ", ".join(f"{d}: {n} layers" for d, n in sorted(devs.items()))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    states = [n for n in ("base", "head") if obs.get(n)]

    for name in states:
        st = obs[name] or {}
        if st.get("error"):
            out.append((f"{name} probe completed", False, str(st["error"])[:200]))
            continue
        for role, build in (("reference", _ref(st)), ("measured", _own(st))):
            tag = build.get("tag") or "?"
            if build.get("error"):
                out.append((f"{name}/{role} bundle usable", False,
                            f"{tag}: {str(build['error'])[:160]}"))
                continue
            ok, ev = _on_gpu(build)
            out.append((f"{name}/{role} ran on the GPU", ok, f"{tag}: {ev}"))
            have = build.get("bench_tg_ts") and build.get("gen_tg_ts")
            out.append((f"{name}/{role} produced both readings", bool(have),
                        f"{tag}: llama-bench tg={build.get('bench_tg_ts')} t/s, "
                        f"generation tg={build.get('gen_tg_ts')} t/s"))
            spread = _gen_spread(build)
            out.append((f"{name}/{role} generation was repeatable",
                        spread is not None and spread <= SPREAD_TOLERANCE_PCT,
                        f"{tag}: spread {spread}% across repeats "
                        f"(tolerance {SPREAD_TOLERANCE_PCT}%)"))

    # The negative control, and the reason this run can make a claim about a
    # number at all: the same build measured in both states must agree.
    if len(states) >= 2:
        tags = {n: _ref(obs[n]).get("tag") for n in states}
        same_tag = len(set(t for t in tags.values() if t)) == 1
        out.append(("control is the same build in every state", same_tag,
                    ", ".join(f"{n}: {t}" for n, t in tags.items())))
        # BOTH readings, not just the generation one. base_shows_defect takes the
        # worse of the two ratios, so a control that moved on either reading can
        # manufacture a regression on that reading alone. Measured on a busy
        # shared box during development: the generation control held to 5% while
        # the llama-bench control moved 28%, which on its own would have read as
        # the head being 140% of the reference.
        for key, what in (("gen_tg_ts", "generation"), ("bench_tg_ts", "llama-bench decode")):
            vals = {n: _ref(obs[n]).get(key) for n in states}
            got = [v for v in vals.values() if v]
            if len(got) == len(states) and max(got) > 0:
                drift = 100.0 * (max(got) - min(got)) / max(got)
                out.append((
                    f"control (base against base) held still, {what}",
                    drift <= CONTROL_TOLERANCE_PCT,
                    "reference build read "
                    + ", ".join(f"{n} {v:.2f} t/s" for n, v in vals.items() if v)
                    + f"; drift {drift:.1f}% (tolerance {CONTROL_TOLERANCE_PCT}%)"))
            else:
                out.append((f"control (base against base) held still, {what}", False,
                            f"the reference build did not produce a {what} reading in "
                            f"every state: {vals}"))
    return out


def table(obs: dict) -> str:
    rows = ["| state | build | role | llama-bench pp t/s | llama-bench tg t/s | "
            "generation tg t/s (top-k 40) | generation tg t/s (top-k 0) | "
            "sampler ops without a device kernel |",
            "|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        st = obs.get(name)
        if not st:
            continue
        for role, build in (("reference (control)", _ref(st)), ("measured", _own(st))):
            if not build:
                continue
            ops = ", ".join(f"`{o}`" for o in build.get("unsupported_sampler_ops", [])) or "none"
            rows.append(
                f"| {name} | `{build.get('tag')}` | {role} | "
                f"{build.get('bench_pp_ts')} | {build.get('bench_tg_ts')} | "
                f"{build.get('gen_tg_ts')} | {build.get('gen_tg_ts_topk0')} | {ops} |")
    extra = []
    for name in ("base", "head"):
        st = obs.get(name)
        if not st:
            continue
        for key, what in (("bench_tg_ts", "llama-bench decode"),
                          ("gen_tg_ts", "real generation")):
            r = _ratio(st, key)
            if r is not None:
                extra.append(f"- `{name}` {what}: {100 * r:.0f}% of the reference build")
    if extra:
        rows.append("")
        rows.append("Measured against the reference build read in the same state:")
        rows += extra

    sweep = _sweep_rows(obs)
    if sweep:
        rows += ["", "Build sweep, all measured in the `base` state on this host in one "
                     "sitting, so they are comparable to each other and to the control "
                     "above. Observations beside the differential, not part of it:", "",
                 "| build | llama-bench tg t/s | generation tg t/s (top-k 40) | "
                 "generation tg t/s (top-k 0) | sampler ops without a device kernel |",
                 "|---|---|---|---|---|"]
        rows += sweep
    return "\n".join(rows)


def _sweep_rows(obs: dict) -> list[str]:
    base = obs.get("base") or {}
    builds = base.get("builds") or {}
    out = []
    for tag in sorted(builds, key = _build_order):
        b = builds[tag]
        ops = ", ".join(f"`{o}`" for o in b.get("unsupported_sampler_ops", [])) or "none"
        marker = {"reference": " (control)", "state": " (base state)"}.get(b.get("role"), "")
        out.append(f"| `{tag}`{marker} | {b.get('bench_tg_ts')} | {b.get('gen_tg_ts')} "
                   f"| {b.get('gen_tg_ts_topk0')} | {ops} |")
    return out if len(out) > 2 else []


def _build_order(tag: str):
    """`b10079-mix-fb3d4ca` sorts after `b10069-...`; a plain string sort puts
    b10107 before b1069 and makes the sweep unreadable."""
    digits = ""
    for ch in tag.lstrip("b"):
        if not ch.isdigit():
            break
        digits += ch
    return (int(digits) if digits else 0, tag)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    """Is the build #7371 calls slow actually slow on this card, right now?

    Slow in EITHER reading counts, because the two readings are how the cause is
    told apart: llama-bench runs no sampler, a real generation does.
    """
    bench, gen = _ratio(base, "bench_tg_ts"), _ratio(base, "gen_tg_ts")
    if bench is None and gen is None:
        return False, "no comparable reading at the base"
    worst = min(r for r in (bench, gen) if r is not None)
    return worst <= DEFECT_RATIO, (
        f"worst reading is {100 * worst:.0f}% of the reference "
        f"(defect threshold {100 * DEFECT_RATIO:.0f}%)")


def head_is_fixed(head: dict) -> tuple[bool, str]:
    bench, gen = _ratio(head, "bench_tg_ts"), _ratio(head, "gen_tg_ts")
    have = [r for r in (bench, gen) if r is not None]
    if not have:
        return False, "no comparable reading at the head"
    return min(have) >= FIXED_RATIO, (
        f"worst reading is {100 * min(have):.0f}% of the reference "
        f"(fixed threshold {100 * FIXED_RATIO:.0f}%)")
