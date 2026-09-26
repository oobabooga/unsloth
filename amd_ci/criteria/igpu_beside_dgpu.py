#!/usr/bin/env python3
"""Criteria: does a real Windows ROCm iGPU keep a used figure when a discrete card sits beside it (#8942, PR 11871)?

Defect: with two visible devices the unified Dedicated + Shared sum was computed only for a lone
device, so the iGPU read None. Base must show that on the spoofed pair (VOID otherwise); head must
report the iGPU inside the bracket of Dedicated + Shared for its REAL LUID sampled before and after,
keep the phantom card at exactly its Dedicated row, and leave the real lone-device reading working.
"""

from __future__ import annotations

TITLE = "Windows ROCm iGPU used VRAM beside a (spoofed) discrete card"
MODE = "differential"
NEEDS = ["windows", "windows_rocm_wddm", "gpu", "discrete_gpu", "multi_gpu"]
TOL_GIB = 0.25
PHANTOM_USED_GIB = 4.28


def _states(obs):
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _bracket_ok(used, refs, total):
    refs = [r for r in refs if r is not None]
    if used is None or not refs:
        return False
    lo, hi = min(refs), min(max(refs), total if total else max(refs))
    return lo - TOL_GIB <= used <= hi + TOL_GIB


def gates(obs):
    st = _states(obs)
    out = []
    out.append(("no probe crashed", all("error" not in v for v in st.values()),
                "; ".join(f"{n}: {v.get('error')}" for n, v in st.items() if "error" in v) or "ok"))
    out.append(("ROCm torch on Windows", all(v.get("system") == "Windows" and v.get("hip") for v in st.values()),
                ", ".join(f"{n}: {v.get('system')} hip={v.get('hip')}" for n, v in st.items())))
    out.append(("detect_hardware ran and set IS_ROCM", all(v.get("is_rocm") for v in st.values()),
                ", ".join(f"{n}: {v.get('device')} is_rocm={v.get('is_rocm')}" for n, v in st.items())))
    out.append(("HIP reported a LUID for the real iGPU", all(v.get("luid") for v in st.values()),
                ", ".join(f"{n}: {hex(v['luid']) if v.get('luid') else None}" for n, v in st.items())))
    out.append(("unified-memory classifier importable", all("classification_error" not in v for v in st.values()),
                "; ".join(f"{n}: {v.get('classification_error') or v.get('classification')}" for n, v in st.items())))
    out.append(("real iGPU is positively unified", all(v.get("positively_unified") for v in st.values()),
                ", ".join(f"{n}: {(v.get('props') or {}).get('arch')}" for n, v in st.items())))
    out.append(("spoofed pair measured", all(len(v.get("spoof") or []) >= 3 for v in st.values()),
                ", ".join(f"{n}: {len(v.get('spoof') or [])}" for n, v in st.items())))
    return out


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def table(obs):
    rows = ["| state | lone iGPU used / total (GiB) | spoof iGPU used | spoof dGPU used | spoof aggregate | real Dedicated+Shared ref | Ded query ms | Shared query ms | lone call ms (med) | spoof call ms (med) | visible poll ms |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, v in _states(obs).items():
        lone = (v.get("lone") or [{}])[-1]
        sp = (v.get("spoof") or [{}])[-1]
        devs = sp.get("devices") or [{}, {}]
        refs = sp.get("ref_gib") or []
        rows.append(
            f"| {n} | {lone.get('used_gb')} / {lone.get('total_gb')} | {devs[0].get('used_gb')} | "
            f"{devs[1].get('used_gb') if len(devs) > 1 else None} | {sp.get('agg')} | "
            f"{', '.join(f'{r:.2f}' for r in refs if r is not None)} | "
            f"{v.get('ms_dedicated_query', 0):.0f} | {v.get('ms_shared_query', 0):.0f} | "
            f"{(_med([x.get('ms') for x in v.get('lone') or []]) or 0):.0f} | "
            f"{(_med([x.get('ms') for x in v.get('spoof') or []]) or 0):.0f} | {v.get('visible_poll_ms', 0):.0f} |")
    rows.append("")
    rows.append(f"Tolerance {TOL_GIB} GiB around the before/after Dedicated + Shared bracket for the real LUID; "
                f"phantom card must read exactly {PHANTOM_USED_GIB} GiB (its Dedicated row).")
    return "\n".join(rows)


def base_shows_defect(base):
    sp = base.get("spoof") or []
    return bool(sp) and all((s.get("devices") or [{}])[0].get("used_gb") is None for s in sp)


def head_is_fixed(head):
    sp = head.get("spoof") or []
    if not sp:
        return False
    for s in sp:
        devs = s.get("devices") or []
        if len(devs) != 2:
            return False
        igpu, dgpu = devs
        if not _bracket_ok(igpu.get("used_gb"), s.get("ref_gib") or [], igpu.get("total_gb")):
            return False
        if dgpu.get("used_gb") is None or abs(dgpu["used_gb"] - PHANTOM_USED_GIB) > 0.01:
            return False
        if s.get("agg") is None or abs(s["agg"] - (igpu["used_gb"] + dgpu["used_gb"])) > 0.02:
            return False
    lone = head.get("lone") or []
    return bool(lone) and all(_bracket_ok(x.get("used_gb"), x.get("ref_gib") or [], x.get("total_gb")) for x in lone)
