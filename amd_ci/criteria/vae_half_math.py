#!/usr/bin/env python3
"""Criteria: does the op-local ROCm VAE half-reduction math keep #17252's win without flipping the process-global
``allow_fp16_bf16_reduction_math_sdp`` flag? Judges probes/vae_half_math_probe.py (base = main with #17252, head = fix).

Noise per arm pair is the A/A arm: ``noise = max(2 * |median(half_aa) / median(half) - 1|, 3%)``.

base_shows_defect: the global flag reads True at some aten op of a half-arm call (the merged scope flips it).
head_is_fixed: the flag never reads True during or after a half-arm call; the half path engaged (and the kill switch did
not); the half arm beats the kill-switch (fp32 upcast) arm beyond noise with no sample overlap, for every attention cell
and the VAE decode; the head half arm is within noise of the base half arm; head half outputs are bitwise equal to the
base half outputs (same numerics as the merged PR).
"""

from __future__ import annotations

TITLE = "gfx1151: VAE attention half-reduction math without the global SDPA flag (timing, exclusive GPU)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "windows", "windows_rocm_wddm", "nvidia", "discrete_gpu"]

_SEEN: dict = {}


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _units(state: dict) -> list[tuple[str, dict]]:
    """(label, unit) for every attention cell and the VAE decode that measured without error."""
    out = [(f"attn S={c.get('tokens')}", c) for c in state.get("cells", []) if "timing" in c]
    if "timing" in state.get("vae", {}):
        out.append(("vae decode 1024px", state["vae"]))
    return out


def _noise(u: dict) -> float:
    m = u["timing"]
    return max(2 * abs(m["half_aa"]["median_ms"] / m["half"]["median_ms"] - 1), 0.03)


def _half_beats_kill(u: dict) -> bool:
    t = u["timing"]
    half = t["half"]["times_ms"] + t["half_aa"]["times_ms"]
    return (t["half"]["median_ms"] < t["kill"]["median_ms"] * (1 - _noise(u))
            and max(half) < min(t["kill"]["times_ms"]))


def _engaged(u: dict) -> bool:
    e = u.get("engagement", {})
    return all(e.get(n, {}).get("half", 0) >= 1 for n in ("half", "half_aa")) and e.get("kill", {}).get("half") == 0


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    _SEEN.clear()
    _SEEN.update(st)
    out = [("runner is gfx1151 with ROCm torch in every state",
            all(v.get("arch") == "gfx1151" and v.get("hip") for v in st.values()),
            ", ".join(f"{k}={v.get('arch')} hip={v.get('hip')} torch={v.get('torch')}" for k, v in st.items()))]
    out.append(("no setup error; every attention cell and the VAE decode measured in every state",
                all("setup_error" not in v and len(_units(v)) == len(v.get("cells", [])) + 1 and v.get("cells")
                    for v in st.values()),
                "; ".join(f"{k}: {len(_units(v))} units, setup={str(v.get('setup_error', 'ok'))[:80]}, "
                          f"vae={str(v.get('vae', {}).get('error', 'ok'))[:120]}" for k, v in st.items())))
    out.append(("no fused SDPA kernel takes head dim 512 (the math path #17252 targets)",
                all(c.get("fused_sdpa_available") is False for v in st.values() for c in v.get("cells", [])),
                ", ".join(f"{k}={[c.get('fused_sdpa_available') for c in v.get('cells', [])]}" for k, v in st.items())))
    out.append(("each state runs its own tree (base has the global-flag scope, head the op-local function)",
                st.get("base", {}).get("has_half_reduction_math") is True
                and st.get("head", {}).get("has_half_math_sdpa") is True
                and st.get("head", {}).get("has_half_reduction_math") is False,
                ", ".join(f"{k}: {v.get('module_file')}" for k, v in st.items())))
    out.append(("the half path engaged in every unit and every state (kill switch: not)",
                all(_engaged(u) for v in st.values() for _, u in _units(v)),
                "; ".join(f"{k} {lbl}: {u.get('engagement')}" for k, v in st.items() for lbl, u in _units(v))))
    out.append(("A/A half arms bitwise equal (deterministic, so cross-state equality means something)",
                all(u.get("half_aa_bitwise") for v in st.values() for _, u in _units(v)), ""))
    return out


def table(obs: dict) -> str:
    st = _states(obs)
    rows = ["| state | unit | reps | half ms | A/A ms | kill (fp32 upcast) ms | half/kill | noise | half peak MiB | "
            "kill peak MiB | max abs half vs fp32 | max abs kill vs fp32 | PSNR half / kill | LPIPS half / kill | "
            "flag True during half (of n) | flag after |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in st.items():
        for lbl, u in _units(v):
            t, n = u["timing"], u.get("vs_fp32", {})
            f = u.get("flag_samples_during_half", {})
            rows.append(
                f"| {name} | {lbl} | {len(t['half']['times_ms'])} | {t['half']['median_ms']:.2f} | "
                f"{t['half_aa']['median_ms']:.2f} | {t['kill']['median_ms']:.2f} | "
                f"{t['half']['median_ms'] / t['kill']['median_ms']:.3f}{' *' if _half_beats_kill(u) else ''} | "
                f"{_noise(u):.3f} | {t['half']['peak_mib']} | {t['kill']['peak_mib']} | "
                f"{n.get('half', {}).get('max_abs', float('nan')):.4g} | {n.get('kill', {}).get('max_abs', float('nan')):.4g} | "
                f"{n.get('half', {}).get('psnr_db', '-')} / {n.get('kill', {}).get('psnr_db', '-')} | "
                f"{n.get('half', {}).get('lpips', '-')} / {n.get('kill', {}).get('lpips', '-')} | "
                f"{f.get('true')} of {f.get('n')} | {u.get('flag_after_half')} |")
    if "base" in st and "head" in st:
        rows.append("")
        bu, hu = dict(_units(st["base"])), dict(_units(st["head"]))
        for lbl in bu:
            if lbl not in hu:
                continue
            b, h = bu[lbl], hu[lbl]
            rows.append(f"- {lbl}: head/base half median {h['timing']['half']['median_ms'] / b['timing']['half']['median_ms']:.3f}"
                        f" (noise {max(_noise(b), _noise(h)):.3f}); half output sha256 equal base vs head: "
                        f"{b.get('sha256', {}).get('half') == h.get('sha256', {}).get('half')}; kill output equal: "
                        f"{b.get('sha256', {}).get('kill') == h.get('sha256', {}).get('kill')}")
        rows.append(f"- lpips: {st['head'].get('vae', {}).get('lpips_error', 'ok')}")
    rows.append("")
    rows.append("`*` = half beats the kill switch: median ratio clears the A/A noise and the samples do not overlap.")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    units = _units(base)
    return bool(units) and any(u.get("flag_samples_during_half", {}).get("true", 0) > 0 for _, u in units)


def head_is_fixed(head: dict) -> bool:
    units = _units(head)
    base = dict(_units(_SEEN.get("base", {})))
    if not units or not base:
        return False
    for lbl, u in units:
        f = u.get("flag_samples_during_half", {})
        if f.get("n", 0) == 0 or f.get("true", 1) != 0 or u.get("flag_after_half") is not False:
            return False
        if not _engaged(u) or not _half_beats_kill(u):
            return False
        b = base.get(lbl)
        if b is None or b.get("sha256", {}).get("half") != u.get("sha256", {}).get("half"):
            return False
        noise = max(_noise(u), _noise(b))
        if u["timing"]["half"]["median_ms"] > b["timing"]["half"]["median_ms"] * (1 + noise):
            return False
    return head.get("flag_at_end") is False
