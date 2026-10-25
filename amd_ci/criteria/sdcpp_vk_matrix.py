#!/usr/bin/env python3
"""Criteria: sd.cpp Vulkan flag matrix on gfx1151 (published Vulkan bundle), and whether the head state's Studio
Vulkan flags are slower than the base state's. Pairs with probes/sdcpp_vk_matrix_probe.py. Regression.

Per (case, arm): medians of generate / TE / sampling / decode over the ok runs, relative spread (max - min) / median
of generate as the A/A noise (floored at 2%), and the number of distinct output hashes (1 = reproducible).
head_is_worse: the head's own Vulkan argv renders slower than the base's by more than the larger of their noises,
or fails, or the head's argv is not reproducible (distinct hashes across its repeats) while the base's was.
"""

from __future__ import annotations

import statistics

TITLE = "sd.cpp Vulkan flag matrix on gfx1151 (published Vulkan bundle)"
MODE = "regression"
NEEDS = ["gpu", "linux", "discrete_gpu", "windows", "nvidia"]
NOISE_FLOOR = 0.02


def _sh(obs: dict) -> dict:
    for name in ("base", "head"):
        s = (obs.get(name) or {}).get("shared")
        if s:
            return s
    return (obs or {}).get("shared") or {}


def _ok(runs: list) -> list:
    return [r for r in runs or [] if r.get("rc") == 0 and isinstance(r.get("generate_s"), (int, float))]


def _med(runs: list, key: str):
    vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
    return statistics.median(vals) if vals else None


def _cell(runs: list) -> dict:
    ok = _ok(runs)
    if not ok:
        return {"n": 0, "n_all": len(runs or [])}
    gen = [r["generate_s"] for r in ok]
    med = statistics.median(gen)
    return {"n": len(ok), "n_all": len(runs), "gen": med, "te": _med(ok, "te_s"), "samp": _med(ok, "sampling_s"),
            "dec": _med(ok, "decode_s"), "noise": max(NOISE_FLOOR, (max(gen) - min(gen)) / med),
            "hashes": len({r.get("sha256") for r in ok}), "rss": _med(ok, "peak_rss_gb"),
            "decisions": sorted({d for r in ok for d in r.get("decisions") or []})}


def gates(obs: dict) -> list:
    sh = _sh(obs)
    m = sh.get("matrix") or {}
    out = [("published Vulkan bundle sha256 matches the release", bool((sh.get("bundle_sha256") or {}).get("match")),
            str(sh.get("bundle_sha256"))),
           ("sd-cli --list-devices lists a Vulkan GPU", bool(sh.get("gpu_listed")),
            str((sh.get("list_devices") or {}).get("out", ""))[-300:]),
           ("every matrix arm rendered >= 2 times", bool(m) and all(
               _cell(runs)["n"] >= 2 for arms in m.values() for runs in arms.values()),
            "; ".join(f"{c}/{a}: {_cell(r)['n']}/{len(r)}" for c, arms in m.items() for a, r in arms.items()))]
    for name in ("base", "head"):
        st = (obs.get(name) or {}).get("studio_vulkan") or {}
        out.append((f"{name}: Studio Vulkan flags read and rendered", "flags" in st and len(_ok(st.get("runs"))) >= 1,
                    f"flags {st.get('flags')} error {str(st.get('error'))[-200:]}"))
    return out


def _f(x, nd = 2):
    return "-" if x is None else f"{x:.{nd}f}"


def table(obs: dict) -> str:
    sh = _sh(obs)
    rows = ["| case | arm | flags / env | n | generate s | TE s | sampling s | VAE decode s | noise | distinct hashes | "
            "peak RSS GB | auto decisions |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    arms = sh.get("arms") or {}
    for case, cells in (sh.get("matrix") or {}).items():
        for arm, runs in cells.items():
            c = _cell(runs)
            spec = arms.get(arm) or {}
            rows.append(f"| {case} | {arm} | `{' '.join(spec.get('flags') or []) or '(none)'}` {spec.get('env') or ''} | "
                        f"{c['n']}/{c['n_all']} | {_f(c.get('gen'))} | {_f(c.get('te'))} | {_f(c.get('samp'))} | "
                        f"{_f(c.get('dec'))} | {_f(c.get('noise'), 3)} | {c.get('hashes', '-')} | {_f(c.get('rss'))} | "
                        f"{'; '.join(c.get('decisions') or [])} |")
    rows += ["", "| state | Studio Vulkan flags | runs | generate s | distinct hashes |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        st = (obs.get(name) or {}).get("studio_vulkan") or {}
        c = _cell(st.get("runs"))
        rows.append(f"| {name} | `{' '.join(st.get('flags') or [])}` | {c['n']}/{c['n_all']} | {_f(c.get('gen'))} | "
                    f"{c.get('hashes', '-')} |")
    return "\n".join(rows)


def head_is_worse(base: dict, head: dict):
    b = _cell((base.get("studio_vulkan") or {}).get("runs"))
    h = _cell((head.get("studio_vulkan") or {}).get("runs"))
    if h["n"] == 0:
        return True, "head's Studio Vulkan argv did not render"
    if b["n"] == 0:
        return False, "base's argv did not render; head did"
    noise = max(b["noise"], h["noise"])
    if h["gen"] > b["gen"] * (1 + noise):
        return True, f"head {h['gen']:.2f} s vs base {b['gen']:.2f} s (noise {noise:.3f})"
    if h["hashes"] > 1 and b["hashes"] == 1:
        return True, f"head argv not reproducible ({h['hashes']} hashes over {h['n']} runs)"
    return False, f"head {h['gen']:.2f} s vs base {b['gen']:.2f} s (noise {noise:.3f}), head hashes {h['hashes']}"
