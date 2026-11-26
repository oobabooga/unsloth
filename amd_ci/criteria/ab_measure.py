#!/usr/bin/env python3
"""Criteria for probes/ab_probe.py: ROCm gfx1151 Triton conv3d (A) and Linear weight pitch (P) before shipping.
Arms differ by tree / env inside the head step, so nothing is judged between states: the gates prove each requested
item ran on gfx1151 with enough good arms; KEEP / KILL is read from the observations (medians, A/A, LPIPS, counts)."""

from __future__ import annotations

import statistics as st

TITLE = "gfx1151: Triton conv3d VAE (A) + Linear weight pitch with the TunableOp rule (P), Studio end to end"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "windows", "nvidia", "multi_gpu"]
FAM_MIN = {"wan": 8, "klein": 6, "krea": 6, "q21": 4}
ITEM_FAM = {"wane2e": "wan", "klein": "klein", "krea": "krea", "q21": "q21"}


def _ok_arms(d):
    return {t: r for t, r in (d or {}).items() if r.get("verdict") == "ok" and r.get("rc") == 0}


def gates(obs: dict) -> list:
    h = obs.get("head") or {}
    arch = (h.get("env") or {}).get("arch") or ""
    items = [x for x in ((h.get("args") or {}).get("items") or "").split(",") if x]
    out = [("gfx1151", arch.startswith("gfx1151"), arch)]
    for item in items:
        if item == "smoke":
            out.append(("smoke", bool(h.get("smoke_ok")), str({k: (h.get("smoke_gpu") or {}).get(k) for k in
                                                                 ("pitch_all_equal", "conv_ok", "rc")})
                        + f" pytest rc {(h.get('smoke_pytest') or {}).get('rc')}"))
        elif item in ITEM_FAM:
            fam = ITEM_FAM[item]
            n = len(_ok_arms(h.get(fam)))
            need = min(FAM_MIN[fam], len(h.get(fam) or {}) or FAM_MIN[fam])
            out.append((f"{fam} arms ok", n >= need and n > 0, f"{n} ok of {len(h.get(fam) or {})}"))
        elif item == "vae":
            v = [r for r in h.get("vae") or [] if r.get("rc") == 0 and not r.get("missing")]
            out.append(("vae runs ok", len(v) >= 1, f"{len(v)} ok of {len(h.get('vae') or [])}"))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    return False, "measurement only (arms differ by tree / env inside the head step)"


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.median(xs), 3) if xs else None


def _spread(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if len(xs) < 2:
        return None
    m = st.median(xs)
    return round(100 * (max(xs) - min(xs)) / m, 2) if m else None


def arm_summary(rec: dict) -> dict:
    rs = rec.get("renders") or []
    return {k: _med([r.get(k) for r in rs]) for k in ("wall_s", "denoise_s", "vae_plus_post_s", "alloc_gib",
                                                       "reserved_gib", "rss_gib")}


def table(obs: dict) -> str:
    h = obs.get("head") or {}
    L = [f"env {h.get('env')}", f"item seconds {h.get('item_s')}", f"smoke {h.get('smoke_ok')}", ""]
    for fam in ("wan", "klein", "krea", "q21"):
        arms = _ok_arms(h.get(fam))
        if not arms:
            continue
        by: dict = {}
        for tag, rec in arms.items():
            by.setdefault(rec.get("arm") or tag.rstrip("0123456789"), []).append(arm_summary(rec))
        for arm, rows in sorted(by.items()):
            walls = [r["wall_s"] for r in rows]
            L.append(f"{fam} {arm}: wall {_med(walls)} [{_spread(walls)}%] denoise "
                     f"{_med([r['denoise_s'] for r in rows])} vae+post {_med([r['vae_plus_post_s'] for r in rows])} "
                     f"alloc {_med([r['alloc_gib'] for r in rows])} rss {_med([r['rss_gib'] for r in rows])}")
        L.append(f"{fam} counts { {t: r.get('ab_counts') for t, r in arms.items()} }")
        L.append(f"{fam} sha_equal {h.get(fam + '_sha_equal')}")
        sc = h.get(fam + "_scores") or {}
        L.append(f"{fam} scores {str(sc)[:1500]}")
    for r in h.get("vae") or []:
        L.append(f"vae {r.get('mode')} {r.get('tag')} {r.get('arm')} rc {r.get('rc')} "
                 f"{ {k: v for k, v in r.items() if k in ('wall_s', 'decode_s', 'peak_gib', 'counts', 'coverage')} }")
    L.append(f"vae psnr {h.get('vae_psnr')}")
    return "\n".join(L)
