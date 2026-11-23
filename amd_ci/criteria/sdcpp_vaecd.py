#!/usr/bin/env python3
"""Criteria for probes/sdcpp_vaecd_probe.py: does Studio keep the sd.cpp fork's VAE-only direct conv on Vulkan?

Differential. Defect at the base: Studio sends a global ``--conv-direct off`` on Vulkan while this card's own VAE
decision (``off,vae=auto`` arm) is on and decodes faster than ``off`` by more than the A/A spread. Fixed at the head:
the state's own flags carry ``vae=auto``, its runs print the VAE decision ``-> on``, decode faster than the ``off``
arm beyond the A/A spread, and the image stays within rounding of ``off`` (PSNR >= 40 dB). The forced ``vae=on``
arm is reported beside it (a device where forcing is slower than auto shows up there)."""

from __future__ import annotations

import statistics

TITLE = "sd.cpp Vulkan: global --conv-direct off vs off,vae=auto vs forced vae=on (Z-Image Turbo Q4_K_M)"
MODE = "differential"
NEEDS = ["gpu", "exclusive_gpu", "nvidia", "multi_gpu"]
PSNR_FLOOR = 40.0


def _shared(o: dict) -> dict:
    return o.get("shared") or {}


def _runs(o: dict, case: str, arm: str) -> list:
    return [r for r in ((_shared(o).get("matrix") or {}).get(case) or {}).get(arm) or [] if r.get("rc") == 0]


def _cases(o: dict) -> list:
    return list((_shared(o).get("matrix") or {}).keys())


def _vals(runs: list, key: str) -> list:
    return [r[key] for r in runs if r.get(key) is not None]


def _med(xs: list):
    return round(statistics.median(xs), 3) if xs else None


def _spread(xs: list) -> float:
    return (max(xs) - min(xs)) if len(xs) > 1 else 0.0


def _vae_on(runs: list) -> bool:
    return bool(runs) and all(any("conv2d direct auto: " in d and "-> on" in d for d in r.get("decisions") or [])
                              for r in runs)


def _faster(o: dict, case: str, arm: str) -> tuple[bool, str]:
    off, other = _vals(_runs(o, case, "cd_off"), "decode_s"), _vals(_runs(o, case, arm), "decode_s")
    if not off or not other:
        return False, f"{case}: no decode timings"
    noise = max(_spread(off), _spread(other))
    gain = statistics.median(off) - statistics.median(other)
    return gain > noise, f"{case}: decode {statistics.median(off):.3f} -> {statistics.median(other):.3f} s " \
                         f"(gain {gain:.3f}, A/A spread {noise:.3f})"


def gates(obs: dict) -> list:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        sh = _shared(o)
        out.append((f"{name}: bundle sha256 matches the release", bool((sh.get("bundle_sha256") or {}).get("match")),
                    str(sh.get("asset"))))
        out.append((f"{name}: Vulkan GPU listed", bool(sh.get("gpu_listed")), str(sh.get("binary"))))
        st = o.get("studio_vulkan") or {}
        out.append((f"{name}: state flags resolved", "flags" in st, str(st.get("flags") or st.get("error"))[:200]))
        out.append((f"{name}: state runs rendered", bool(st.get("runs")) and all(r.get("rc") == 0 for r in st["runs"]),
                    str([r.get("rc") for r in st.get("runs") or []])))
    head = obs.get("head") or {}
    for case in _cases(head) or ["(none)"]:
        for arm in ("cd_off", "vae_auto", "vae_on"):
            n = len(_runs(head, case, arm))
            out.append((f"matrix {case} {arm}: >= 2 clean repeats", n >= 2, f"{n} runs"))
    return out


def table(obs: dict) -> str:
    head = obs.get("head") or {}
    rows = ["| case | arm | decode s (median, spread) | sampling s | generate s | peak RSS GB | PSNR vs off r0 | "
            "VAE decision |", "|---|---|---|---|---|---|---|---|"]
    for case in _cases(head):
        for arm in ("cd_off", "vae_auto", "vae_on"):
            runs = _runs(head, case, arm)
            dec = _vals(runs, "decode_s")
            dset = sorted({d for r in runs for d in r.get("decisions") or [] if "conv" in d})
            rows.append(f"| {case} | {arm} | {_med(dec)} ({_spread(dec):.3f}) | {_med(_vals(runs, 'sampling_s'))} | "
                        f"{_med(_vals(runs, 'generate_s'))} | {_med(_vals(runs, 'peak_rss_gb'))} | "
                        f"{_med(_vals(runs, 'psnr_vs_cd_off_r0'))} | {'; '.join(dset) or '-'} |")
    rows += ["", "| state | Studio flags | runs: decode s | generate s | peak RSS GB | PSNR vs off r0 | decisions |",
             "|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        st = (obs.get(name) or {}).get("studio_vulkan") or {}
        runs = st.get("runs") or []
        rows.append(f"| {name} | `{' '.join(st.get('flags') or [])}` (accelerator {st.get('accelerator')}) | "
                    f"{_vals(runs, 'decode_s')} | {_vals(runs, 'generate_s')} | {_vals(runs, 'peak_rss_gb')} | "
                    f"{_vals(runs, 'psnr_vs_cd_off_r0')} | "
                    f"{'; '.join(sorted({d for r in runs for d in r.get('decisions') or [] if 'conv' in d})) or '-'} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    st = base.get("studio_vulkan") or {}
    flags = st.get("flags") or []
    if "--conv-direct" not in flags or flags[flags.index("--conv-direct") + 1] != "off":
        return False
    case = (_cases(base) or [None])[0]
    if case is None or not _vae_on(_runs(base, case, "vae_auto")):
        return False
    return _faster(base, case, "vae_auto")[0]


def head_is_fixed(head: dict) -> bool:
    st = head.get("studio_vulkan") or {}
    flags = st.get("flags") or []
    if "--conv-direct" not in flags or "vae=auto" not in flags[flags.index("--conv-direct") + 1]:
        return False
    runs = [r for r in st.get("runs") or [] if r.get("rc") == 0]
    if not _vae_on(runs):
        return False
    case = st.get("case")
    off = _vals(_runs(head, case, "cd_off"), "decode_s")
    mine = _vals(runs, "decode_s")
    if not off or not mine:
        return False
    noise = max(_spread(off), _spread(_vals(_runs(head, case, "vae_auto"), "decode_s")))
    if statistics.median(off) - statistics.median(mine) <= noise:
        return False
    return all((p or 0) >= PSNR_FLOOR for p in _vals(runs, "psnr_vs_cd_off_r0")) and bool(_vals(runs, "psnr_vs_cd_off_r0"))
