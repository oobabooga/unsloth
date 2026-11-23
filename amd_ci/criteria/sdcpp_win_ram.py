#!/usr/bin/env python3
"""Criteria for probes/sdcpp_win_ram_probe.py on Windows gfx1151: the sd.cpp offload CPU VAE tile without psutil.

Differential. Defect at the base: with psutil unimportable, the base state's host RAM reading is None, so
`_offload_vae_flags` emits --vae-tiling with no --vae-tile-size (sd.cpp's 256 px default), while the same state with
psutil importable passes a RAM-sized tile on the same box. Fixed at the head: without psutil the head reads the same
total RAM psutil reports, emits the same flags as with psutil (a tile > 256 px), and that render succeeds with no
sd.cpp tile reduction / OOM retry. Decode s and PSNR vs the untiled decode are reported beside it.
"""

from __future__ import annotations

TITLE = "sd.cpp offload CPU VAE tile on Windows without psutil (GlobalMemoryStatusEx fallback), gfx1151"
MODE = "differential"
NEEDS = ["gpu", "exclusive_gpu", "nvidia", "discrete_gpu"]


def _ch(o: dict, cond: str) -> dict:
    return (o.get("choice") or {}).get(cond) or {}


def _arm(o: dict, cond: str, size: int):
    return next((a for a in o.get("arms") or [] if a.get("cond") == cond and a["size"] == size), None)


def _run(o: dict, a) -> dict:
    return (o.get("runs") or {}).get((a or {}).get("key")) or {}


def _ok(r: dict) -> bool:
    return r.get("rc") == 0 and bool(r.get("png"))


def _tile(flags) -> int | None:
    flags = list(flags or [])
    if "--vae-tiling" not in flags:
        return None
    return int(flags[flags.index("--vae-tile-size") + 1]) if "--vae-tile-size" in flags else 256


def gates(obs: dict) -> list:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        p = o.get("prepared") or {}
        vk = (p.get("bundles") or {}).get("vulkan") or {}
        out.append((f"{name}: Vulkan bundle sha256 matches the release", bool((vk.get("sha256") or {}).get("match")),
                    str(vk.get("asset"))))
        out.append((f"{name}: Vulkan GPU listed", bool(vk.get("gpu_listed")), str(vk.get("description"))))
        out.append((f"{name}: public models fetched (no token)", bool(p.get("models_ok")),
                    str({k: v.get("error", "ok") for k, v in (p.get("model_fetch") or {}).items()})))
        for cond in ("nopsutil", "psutil"):
            ch = _ch(o, cond)
            out.append((f"{name}/{cond}: chooser ran and emitted flags", bool(ch.get("flags")),
                        str(ch.get("flags") or ch.get("error"))[-300:]))
            has = isinstance((ch.get("raw") or {}).get("psutil"), dict)
            out.append((f"{name}/{cond}: psutil {'importable' if cond == 'psutil' else 'blocked'}",
                        has == (cond == "psutil"), str((ch.get("raw") or {}).get("psutil"))[:200]))
            out.append((f"{name}/{cond}: pixel-unit sd-cli", bool(ch.get("tile_in_pixels")), str(ch.get("tile_in_pixels"))))
        refs = [r for r in (o.get("runs") or {}).values() if r.get("tile_passed") is None]
        out.append((f"{name}: untiled reference rendered at every size",
                    {r["size"] for r in refs if _ok(r)} >= set(o.get("sizes") or [0]),
                    str(sorted({(r["size"], r.get("rc")) for r in refs}))))
    return out


def _f(x, nd = 2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def table(obs: dict) -> str:
    rows = ["| state | python | psutil | host_ram_mib (total, avail) | commit MiB | spare MiB | flags | tile | size | rc | "
            "decode s | VAE buffer MiB | reduce/OOM | PSNR vs untiled |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        o = obs.get(name) or {}
        for cond in ("nopsutil", "psutil"):
            ch = _ch(o, cond)
            sr = ch.get("state_readings") or {}
            ps = (ch.get("raw") or {}).get("psutil")
            for size in o.get("sizes") or []:
                a = _arm(o, cond, size)
                r = _run(o, a)
                rows.append(
                    f"| {name} | {cond} | {ps.get('version') if isinstance(ps, dict) else 'absent'} | "
                    f"{sr.get('diffusion_small_host.host_ram_mib')} | {sr.get('diffusion_small_host.host_commit_mib')} | "
                    f"{ch.get('spare_mib')} | `{' '.join(ch.get('flags') or [])}` | {_tile(ch.get('flags'))} | {size} | "
                    f"{r.get('rc')} | {_f(r.get('decode_s'))} | {_f(r.get('vae_buffer_mib'), 0)} | "
                    f"{'; '.join(r.get('reduce_or_oom') or []) or '-'} | {_f((a or {}).get('psnr_vs_untiled'))} |")
    rows += ["", "Raw OS readings (head, nopsutil):", "", "```", str((_ch(obs.get("head") or {}, "nopsutil")).get("raw")), "```"]
    return "\n".join(rows)


def base_shows_defect(base: dict) -> tuple[bool, str]:
    no, yes = _ch(base, "nopsutil"), _ch(base, "psutil")
    if base.get("platform") != "win32":
        return False, "not Windows"
    tn, ty = _tile(no.get("flags")), _tile(yes.get("flags"))
    sr = no.get("state_readings") or {}
    ram = sr.get("diffusion_small_host.host_ram_mib")
    unread = isinstance(ram, list) and ram[1] is None
    shown = unread and tn == 256 and (ty or 0) > 256
    return shown, f"nopsutil host_ram_mib {ram}, tile {tn}; psutil tile {ty}"


def _state_fixed(o: dict) -> tuple[bool, str]:
    no, yes = _ch(o, "nopsutil"), _ch(o, "psutil")
    tn, ty = _tile(no.get("flags")), _tile(yes.get("flags"))
    if no.get("flags") != yes.get("flags"):
        return False, f"nopsutil flags {no.get('flags')} != psutil flags {yes.get('flags')}"
    if (tn or 0) <= 256:
        return False, f"tile {tn}: still sd.cpp's 256 px default"
    rn = (no.get("state_readings") or {}).get("diffusion_small_host.host_ram_mib")
    ry = (yes.get("state_readings") or {}).get("diffusion_small_host.host_ram_mib")
    if not (isinstance(rn, list) and isinstance(ry, list) and rn[0] == ry[0] and rn[1] is not None):
        return False, f"nopsutil reading {rn} vs psutil {ry}"
    for size in o.get("sizes") or []:
        r = _run(o, _arm(o, "nopsutil", size))
        if not _ok(r):
            return False, f"{size}px: rc {r.get('rc')}"
        if r.get("reduce_or_oom"):
            return False, f"{size}px: sd.cpp reduced/retried: {r['reduce_or_oom']}"
    return True, f"nopsutil reads {rn} (psutil {ry}), tile {tn} = psutil tile, renders OK"


def head_is_fixed(head: dict) -> tuple[bool, str]:
    return _state_fixed(head)


def state_is_fixed(name: str, obs: dict):
    return _state_fixed(obs)
