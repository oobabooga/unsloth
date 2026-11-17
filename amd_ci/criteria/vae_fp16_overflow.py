#!/usr/bin/env python3
"""Criteria: unsloth #18016. Judges probes/vae_fp16_overflow_probe.py (base = merge base, head = the fix).

base_shows_defect: the real Z-Image VAE decoded in fp16 (the gfx103x dtype) with Studio's attention patch and default
settings returns NaN at some size, with the half path engaged.
head_is_fixed: every fp16 default decode is finite, does not engage half math, and is bitwise equal to the kill-switch
arm (UNSLOTH_VAE_ATTN_HALF_MATH=0, the reporter's workaround); the bf16 decode still engages half math, is bitwise equal
to base's bf16 half output, its median is within noise of base's, and it beats the kill-switch arm beyond noise with no
sample overlap (the gfx1151 gain is kept).

Noise per unit: ``max(2 * |median(half_aa) / median(half) - 1|, 3%)``.
"""

from __future__ import annotations

TITLE = "gfx1151: Z-Image VAE fp16 decode NaN (#18016) fixed, bf16 half-math VAE attention gain kept (timing)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "windows", "windows_rocm_wddm", "nvidia", "discrete_gpu"]

_SEEN: dict = {}


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _fp16(state: dict, arm: str) -> list:
    return [c for c in state.get("fp16", []) if c.get("arm") == arm and "vs_fp32" in c]


def _units(state: dict) -> list:
    out = []
    if "timing" in state.get("bf16", {}):
        out.append(("bf16 decode 1024px", state["bf16"]))
    if "timing" in state.get("attn", {}):
        out.append(("bf16 attn S=16384 d=512", state["attn"]))
    return out


def _noise(u: dict) -> float:
    t = u["timing"]
    return max(2 * abs(t["half_aa"]["median_ms"] / t["half"]["median_ms"] - 1), 0.03)


def _beats(u: dict, other: str) -> bool:
    t = u["timing"]
    half = t["half"]["times_ms"] + t["half_aa"]["times_ms"]
    return t["half"]["median_ms"] < t[other]["median_ms"] * (1 - _noise(u)) and max(half) < min(t[other]["times_ms"])


def gates(obs: dict) -> list:
    st = _states(obs)
    _SEEN.clear()
    _SEEN.update(st)
    out = [("runner is gfx1151 with ROCm torch in every state",
            all(v.get("arch") == "gfx1151" and v.get("hip") for v in st.values()),
            ", ".join(f"{k}={v.get('arch')} hip={v.get('hip')} torch={v.get('torch')} {v.get('platform')}"
                      for k, v in st.items()))]
    out.append(("no setup error; 4 fp16 decode cells, the bf16 decode and the attention op measured in every state",
                all("setup_error" not in v and len([c for c in v.get("fp16", []) if "vs_fp32" in c]) == 4
                    and len(_units(v)) == 2 for v in st.values()),
                "; ".join(f"{k}: setup={str(v.get('setup_error', 'ok'))[:200]} fp16={str(v.get('fp16_error', 'ok'))[:120]}"
                          f" bf16={str(v.get('bf16', {}).get('error', 'ok'))[:120]}"
                          f" attn={str(v.get('attn', {}).get('error', 'ok'))[:120]}" for k, v in st.items())))
    out.append(("no fused SDPA kernel takes head dim 512 (the math path the half-math default targets)",
                all(v.get("attn", {}).get("fused_sdpa_available") is False for v in st.values()),
                ", ".join(f"{k}={v.get('attn', {}).get('fused_sdpa_available')}" for k, v in st.items())))
    out.append(("each state runs its own tree",
                len({v.get("module_sha256") for v in st.values()}) == len(st),
                ", ".join(f"{k}: {v.get('module_file')} {str(v.get('module_sha256'))[:12]}" for k, v in st.items())))
    out.append(("the reporter's workaround (kill switch) decodes fp16 finite in every state",
                all(c["vs_fp32"]["finite"] for v in st.values() for c in _fp16(v, "kill")) and
                all(_fp16(v, "kill") for v in st.values()),
                "; ".join(f"{k} {c['size']}: nan={c['vs_fp32']['nan']}" for k, v in st.items() for c in _fp16(v, "kill"))))
    out.append(("bf16 A/A half arms bitwise equal (deterministic, so cross-state equality means something)",
                all(v.get("bf16", {}).get("half_aa_bitwise") for v in st.values()), ""))
    out.append(("the bf16 decode engages half math in every state (kill switch: not)",
                all(v.get("bf16", {}).get("engagement", {}).get("half", {}).get("half", 0) >= 1
                    and v.get("bf16", {}).get("engagement", {}).get("kill", {}).get("half") == 0 for v in st.values()),
                "; ".join(f"{k}: {v.get('bf16', {}).get('engagement')}" for k, v in st.items())))
    return out


def table(obs: dict) -> str:
    st = _states(obs)
    rows = ["### fp16 decode, real Z-Image-Turbo VAE, real latents (gfx103x dtype)", "",
            "| state | size | arm | NaN values | half-math calls | max abs vs fp32 | PSNR dB | LPIPS |",
            "|---|---|---|---|---|---|---|---|"]
    for name, v in st.items():
        for c in v.get("fp16", []):
            n = c.get("vs_fp32", {})
            rows.append(f"| {name} | {c.get('size')} | {c.get('arm')} | {n.get('nan', c.get('error', '')[:60])} | "
                        f"{c.get('engagement', {}).get('half')} | {n.get('max_abs')} | {n.get('psnr_db', '-')} | "
                        f"{n.get('lpips', '-')} |")
    rows += ["", "### bf16 (gfx1151 default)", "",
             "| state | unit | reps | half ms | A/A ms | kill (fp32) ms | PR as submitted ms | half/kill | "
             "half/PR-orig | noise | half peak MiB | kill peak MiB | PR-orig peak MiB | PSNR half / kill / PR-orig | "
             "LPIPS half / kill / PR-orig |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in st.items():
        for lbl, u in _units(v):
            t, n = u["timing"], u.get("vs_fp32", {})
            rows.append(
                f"| {name} | {lbl} | {len(t['half']['times_ms'])} | {t['half']['median_ms']:.2f} | "
                f"{t['half_aa']['median_ms']:.2f} | {t['kill']['median_ms']:.2f} | {t['pr_orig']['median_ms']:.2f} | "
                f"{t['half']['median_ms'] / t['kill']['median_ms']:.3f}{' *' if _beats(u, 'kill') else ''} | "
                f"{t['half']['median_ms'] / t['pr_orig']['median_ms']:.3f}{' *' if _beats(u, 'pr_orig') else ''} | "
                f"{_noise(u):.3f} | {t['half']['peak_mib']} | {t['kill']['peak_mib']} | {t['pr_orig']['peak_mib']} | "
                + " / ".join(str(n.get(a, {}).get("psnr_db", "-")) for a in ("half", "kill", "pr_orig")) + " | "
                + " / ".join(str(n.get(a, {}).get("lpips", "-")) for a in ("half", "kill", "pr_orig")) + " |")
    if "base" in st and "head" in st:
        rows.append("")
        bu, hu = dict(_units(st["base"])), dict(_units(st["head"]))
        for lbl in bu:
            if lbl in hu:
                b, h = bu[lbl], hu[lbl]
                rows.append(f"- {lbl}: head/base half median "
                            f"{h['timing']['half']['median_ms'] / b['timing']['half']['median_ms']:.3f} "
                            f"(noise {max(_noise(b), _noise(h)):.3f}); bf16 half sha256 equal base vs head: "
                            f"{b.get('sha256', {}).get('half') == h.get('sha256', {}).get('half') if 'sha256' in b else '-'}")
        rows.append(f"- bf16 decode engagement head: {st['head'].get('bf16', {}).get('engagement')}")
        rows.append(f"- lpips: {st['head'].get('lpips_error', 'ok')}")
    rows += ["", "`*` = half beats that arm: median ratio clears the A/A noise and the samples do not overlap."]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    cells = _fp16(base, "default")
    return any(not c["vs_fp32"]["finite"] and c.get("engagement", {}).get("half", 0) >= 1 for c in cells)


def head_is_fixed(head: dict) -> bool:
    cells, kills = _fp16(head, "default"), {c["size"]: c for c in _fp16(head, "kill")}
    if len(cells) != 2 or len(kills) != 2:
        return False
    for c in cells:
        if not c["vs_fp32"]["finite"] or c.get("engagement", {}).get("half") != 0:
            return False
        if c.get("sha256") != kills[c["size"]].get("sha256"):
            return False
    base = dict(_units(_SEEN.get("base", {})))
    units = _units(head)
    if len(units) != 2 or len(base) != 2:
        return False
    hb, bb = head["bf16"], _SEEN["base"]["bf16"]
    if hb.get("engagement", {}).get("half", {}).get("half", 0) < 1:
        return False
    if hb.get("sha256", {}).get("half") != bb.get("sha256", {}).get("half"):
        return False
    for lbl, u in units:
        b = base[lbl]
        if not _beats(u, "kill"):
            return False
        if u["timing"]["half"]["median_ms"] > b["timing"]["half"]["median_ms"] * (1 + max(_noise(u), _noise(b))):
            return False
    return True
