#!/usr/bin/env python3
"""Criteria: on gfx1151, does contiguous k/v beat the strided k/v diffusers' native backend hands SDPA, and does the
head's Studio attention path collect that win without changing outputs? Judges probes/sdpa_contiguous_kv_probe.py.

Noise per cell is the A/A arm: ``noise = max(2 * |median(strided_aa) / median(strided) - 1|, 3%)``. A gain counts only
when the median ratio clears the noise AND every contiguous-arm sample beats every strided sample (no overlap).

base_shows_defect: at some cell contiguous k/v is a real gain over the Studio path (Studio leaves it on the table).
head_is_fixed: at every cell where contiguous k/v is a real gain, the Studio path is within noise of it and SDPA saw
contiguous k/v; at every cell, the Studio path is no slower than strided beyond noise; Studio outputs are bitwise equal
to the strided reference. The wan_attn arm (a real diffusers processor) is compared base vs head in the table.
"""

from __future__ import annotations

TITLE = "gfx1151: contiguous k/v before SDPA (timing, exclusive GPU)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "discrete_gpu", "windows", "windows_rocm_wddm", "multi_gpu"]


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _cells(state: dict) -> list[dict]:
    return [c for c in state.get("cells", []) if "median_ms" in c]


def _noise(c: dict) -> float:
    m = c["median_ms"]
    return max(2 * abs(m["strided_aa"] / m["strided"] - 1), 0.03)


def _real_gain(c: dict, fast: str, slow: str) -> bool:
    m, t = c["median_ms"], c["times_ms"]
    if fast not in m or slow not in m:
        return False
    slow_samples = t[slow] + (t["strided_aa"] if slow == "strided" else [])
    return m[fast] < m[slow] * (1 - _noise(c)) and max(t[fast]) < min(slow_samples)


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    out = []
    out.append(("runner is gfx1151 in every state", all(v.get("arch") == "gfx1151" for v in st.values()),
                ", ".join(f"{k}={v.get('arch')} L2={v.get('l2_cache_size')}" for k, v in st.items())))
    full = {k: (len(_cells(v)), len(v.get("cells", []))) for k, v in st.items()}
    out.append(("every state measured at least 4 cells without error",
                all(a >= 4 and a == b for a, b in full.values()),
                ", ".join(f"{k}={a}/{b}" for k, (a, b) in full.items())))
    out.append(("Studio path configured in every state",
                all("studio_error" not in v and all("studio" in c["median_ms"] for c in _cells(v))
                    for v in st.values()),
                ", ".join(f"{k}={v.get('studio_error', 'ok')[:80]}" for k, v in st.items())))
    base = st.get("base", {})
    out.append(("base carries no contiguous-k/v wrapper", base.get("studio_native_wrapped") is False,
                f"base wrapped={base.get('studio_native_wrapped')}"))
    out.append(("A/A arm bitwise equal to strided everywhere",
                all(c["numerics_vs_strided"]["strided_aa"]["bitwise_equal"] for v in st.values() for c in _cells(v)),
                "SDPA is deterministic run to run, so a non-equal A/A means the comparison is meaningless"))
    return out


def table(obs: dict) -> str:
    st = _states(obs)
    rows = ["| state | H | S | k+v head bytes | reps | strided ms | A/A ms | noise | kv_contig ms (copy incl.) | "
            "kv/strided | studio ms | studio/strided | studio SDPA k,v contig | kv bitwise | studio bitwise | "
            "wan_attn ms | kernels strided / kv_contig |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in st.items():
        for c in v.get("cells", []):
            if "median_ms" not in c:
                rows.append(f"| {name} | {c.get('heads')} | {c.get('tokens')} | | | ERROR {str(c.get('error'))[:120]} |"
                            + " |" * 11)
                continue
            m = c["median_ms"]
            saw = c.get("studio_sdpa_saw_contiguous") or [[None, None, None]]
            num = c["numerics_vs_strided"]
            k = c.get("kernels", {})
            rows.append(
                f"| {name} | {c['heads']} | {c['tokens']} | {c['kv_strided_bytes_per_head']} | {c['reps']} | "
                f"{m['strided']:.2f} | {m['strided_aa']:.2f} | {_noise(c):.3f} | {m['kv_contig']:.2f} | "
                f"{m['kv_contig'] / m['strided']:.3f}{' *' if _real_gain(c, 'kv_contig', 'strided') else ''} | "
                f"{m.get('studio', float('nan')):.2f} | {m.get('studio', float('nan')) / m['strided']:.3f} | "
                f"{saw[0][1]},{saw[0][2]} | {num['kv_contig']['bitwise_equal']} ({num['kv_contig']['max_abs']:.2g}) | "
                f"{num.get('studio', {}).get('bitwise_equal')} | "
                f"{m.get('wan_attn', float('nan')):.2f} | "
                f"{(k.get('strided') or ['?'])[0][:50]} / {(k.get('kv_contig') or ['?'])[0][:50]} |")
    if "base" in st and "head" in st:
        bw = {c["tokens"]: c.get("wan_out_sha256") for c in _cells(st["base"]) if c.get("wan_out_sha256")}
        hw = {c["tokens"]: c.get("wan_out_sha256") for c in _cells(st["head"]) if c.get("wan_out_sha256")}
        same = {s: bw[s] == hw.get(s) for s in bw}
        rows.append("")
        rows.append(f"wan_attn output sha256 equal base vs head per S: {same}")
        bm = {c["tokens"]: c["median_ms"].get("wan_attn") for c in _cells(st["base"]) if "wan_attn" in c["median_ms"]}
        hm = {c["tokens"]: c["median_ms"].get("wan_attn") for c in _cells(st["head"]) if "wan_attn" in c["median_ms"]}
        rows.append("wan_attn (Wan 14B self-attention module, 40 heads) head/base median: "
                    + ", ".join(f"S={s}: {hm[s] / bm[s]:.3f}" for s in bm if s in hm and bm[s]))
    rows.append("")
    rows.append("`*` = real gain: median ratio clears the A/A noise and the samples do not overlap.")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    return any(_real_gain(c, "kv_contig", "studio") for c in _cells(base))


def head_is_fixed(head: dict) -> bool:
    cells = _cells(head)
    if not cells:
        return False
    for c in cells:
        m = c["median_ms"]
        if "studio" not in m:
            return False
        if not c["numerics_vs_strided"].get("studio", {}).get("bitwise_equal"):
            return False
        if m["studio"] > m["strided"] * (1 + _noise(c)):
            return False
        if _real_gain(c, "kv_contig", "strided"):
            saw = c.get("studio_sdpa_saw_contiguous") or []
            if not saw or not all(s[1] and s[2] for s in saw):
                return False
            if m["studio"] > m["kv_contig"] * (1 + _noise(c)):
                return False
    return True
