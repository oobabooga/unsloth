#!/usr/bin/env python3
"""Criteria: MiniMax-H3 streamed conditioner encode (judges probes/h3_te_encode_probe.py).

Every state measures four modes in fresh processes: lazy / lazy_off (pageable host copy, head kill switch on / off)
and pinned / pinned_off. The base ignores the switch, so its lazy vs lazy_off pair is the A/A noise arm:
``noise = max(2 * |median(lazy_off) / median(lazy) - 1|, 5%)``.

base_shows_defect: the base's pageable encode is >= 1.5x slower than its pinned encode (same bytes, same kernels, only
the host copy path differs).
head_is_fixed: in the head, the pageable encode with the layer streamer is <= 0.5x the kill-switch encode (no sample
overlap), the pinned encode is no slower than its kill-switch arm beyond noise, the streamer engaged on every layer in
the streamer arms and not in the kill-switch arms, every arm's hidden_states[50] has the same sha256, and the encode's
peak allocated device bytes stay within the 3.0 GB streamed-conditioner budget (H3_TE_STREAMED_GB).
"""

from __future__ import annotations

import statistics

TITLE = "gfx1151: MiniMax-H3 streamed conditioner encode, layer transfers vs leaf groups (timing, exclusive GPU)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "discrete_gpu", "windows", "multi_gpu"]
BUDGET_BYTES = 3.0e9
LAYERS = 50


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _med(mode: dict) -> float:
    return statistics.median(mode["encode_s"]) if mode.get("encode_s") else float("nan")


def _ok(mode: dict) -> bool:
    return bool(mode) and "error" not in mode and bool(mode.get("encode_s"))


def _noise(state: dict) -> float:
    m = state.get("modes", {})
    if not (_ok(m.get("lazy", {})) and _ok(m.get("lazy_off", {}))):
        return 0.05
    return max(2 * abs(_med(m["lazy_off"]) / _med(m["lazy"]) - 1), 0.05)


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = []
    archs = {k: {m.get("arch") for m in v.get("modes", {}).values()} for k, v in st.items()}
    out.append(("runner is gfx1151 in every state", all(a == {"gfx1151"} for a in archs.values()), str(archs)))
    errs = {k: {n: (m.get("error") or "")[:120] for n, m in v.get("modes", {}).items() if not _ok(m)}
            for k, v in st.items()}
    out.append(("every mode ran in every state", all(len(v.get("modes", {})) == 4 for v in st.values())
                and not any(errs.values()), str(errs)))
    modes = {k: {n: m.get("te_streamed") for n, m in v.get("modes", {}).items()} for k, v in st.items()}
    want = {"lazy": "stream_lazy", "lazy_off": "stream_lazy", "pinned": "stream", "pinned_off": "stream"}
    out.append(("Studio streamed the conditioner in the requested mode", all(
        all(mm.get(n) == w for n, w in want.items()) for mm in modes.values()), str(modes)))
    out.append(("hidden states finite", all(m.get("finite") for v in st.values() for m in v.get("modes", {}).values()),
                ""))
    if "base" in st and "head" in st:
        shas = {f"{k}.{n}": set(m.get("sha256") or []) for k, v in st.items() for n, m in v.get("modes", {}).items()}
        allsha = set().union(*shas.values()) if shas else set()
        out.append(("hidden_states[50] bit-identical across every state and mode", len(allsha) == 1,
                    f"{len(allsha)} distinct sha256"))
    return out


def table(obs: dict) -> str:
    st = _states(obs)
    rows = ["| state | mode | te_streamed | layer streamer | encode s (each) | median s | peak alloc GB | staged / direct GB |"
            " first encode s | sync / memcpy calls (profiled encode) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in st.items():
        for mode, m in v.get("modes", {}).items():
            if not _ok(m):
                rows.append(f"| {name} | {mode} | ERROR {str(m.get('error'))[:150]} |" + " |" * 7)
                continue
            s = m.get("streamer") or {}
            calls = m.get("profile_calls") or {}
            sync = sum(c["count"] for k, c in calls.items() if "ynchronize" in k)
            mem = sum(c["count"] for k, c in calls.items() if "emcpy" in k)
            rows.append(
                f"| {name} | {mode} | {m.get('te_streamed')} | {m.get('layer_stream_installed')} "
                f"{s.get('layers_streamed', '')} | {', '.join(f'{x:.2f}' for x in m['encode_s'])} | {_med(m):.2f} | "
                f"{max(m.get('peak_alloc_bytes') or [0]) / 1e9:.2f} | "
                f"{s.get('bytes_staged', 0) / 1e9:.1f} / {s.get('bytes_direct', 0) / 1e9:.1f} | "
                f"{m.get('first_encode_s')} | {sync} / {mem} |")
    if "base" in st and "head" in st:
        b, h = st["base"].get("modes", {}), st["head"].get("modes", {})
        rows.append("")
        for mode in ("lazy", "pinned"):
            if _ok(b.get(mode, {})) and _ok(h.get(mode, {})):
                rows.append(f"{mode}: head / base median = {_med(h[mode]) / _med(b[mode]):.3f} "
                            f"({_med(b[mode]):.2f} s -> {_med(h[mode]):.2f} s); base A/A noise {_noise(st['base']):.3f}")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    m = base.get("modes", {})
    if not (_ok(m.get("lazy", {})) and _ok(m.get("pinned", {}))):
        return False
    return _med(m["lazy"]) >= 1.5 * _med(m["pinned"])


def head_is_fixed(head: dict) -> bool:
    m = head.get("modes", {})
    if not all(_ok(m.get(n, {})) for n in ("lazy", "lazy_off", "pinned", "pinned_off")):
        return False
    noise = 0.05
    if not (_med(m["lazy"]) <= 0.5 * _med(m["lazy_off"]) and max(m["lazy"]["encode_s"]) < min(m["lazy_off"]["encode_s"])):
        return False
    if _med(m["pinned"]) > _med(m["pinned_off"]) * (1 + noise):
        return False
    for n in ("lazy", "pinned"):
        s = m[n].get("streamer") or {}
        if not m[n].get("layer_stream_installed") or s.get("layers_streamed", 0) < LAYERS * len(m[n]["encode_s"]):
            return False
        if s.get("fallbacks", 0):
            return False
    if any(m[n].get("layer_stream_installed") for n in ("lazy_off", "pinned_off")):
        return False
    shas = set()
    for n in m:
        shas |= set(m[n].get("sha256") or [])
    if len(shas) != 1:
        return False
    return all(max(m[n].get("peak_alloc_bytes") or [0]) <= BUDGET_BYTES for n in ("lazy", "pinned"))
