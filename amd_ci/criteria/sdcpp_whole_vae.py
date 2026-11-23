#!/usr/bin/env python3
"""Criteria for probes/sdcpp_whole_vae_probe.py: PR 18999 (offload VAE decoded in one tile when the budget holds it)
on gfx1151, FLUX.2 klein 4B Q4_K_M, published sd.cpp bundles.

Differential. Defect at the base: at some measured size the base state's own offload flags tile the decode below
the image (tile < size), and that render is visibly off the untiled decode of the same latent (PSNR < 40 dB).
Fixed at the head: at EVERY size the head's own Vulkan-class flags pass a tile >= the image, the render succeeds
with no sd.cpp tile reduction / OOM retry, matches the untiled decode (PSNR >= 40 dB), and the VAE compute buffer
sd.cpp reports stays within the PR's table value times the margin the chooser applied for that placement
(GPU 1.15, CPU 1.3). The forced arms (GPU 1024 tiles, GPU 2048 whole, CPU 2048 whole) are reported beside it.
"""

from __future__ import annotations

TITLE = "sd.cpp offload VAE: tiled (base) vs whole-image tile (head), FLUX.2 klein 4B, gfx1151"
MODE = "differential"
NEEDS = ["gpu", "exclusive_gpu", "nvidia", "discrete_gpu"]
PSNR_FLOOR = 40.0
TABLE = {1536: 6336, 2048: 17040}  # PR 18999 _FLUX2_VAE_WHOLE_MIB (L4, CUDA)
MARGIN = {"VRAM": 1.15, "RAM": 1.3}


def _runs(o: dict) -> dict:
    return o.get("runs") or {}


def _arms(o: dict, prefix: str) -> list:
    return [a for a in o.get("arms") or [] if a["name"].startswith(prefix)]


def _table(o: dict) -> dict:
    t = ((o.get("choice") or {}).get("vulkan") or {}).get("table")
    return {int(k): v for k, v in t.items()} if t else dict(TABLE)


def _where(rec: dict) -> str:
    w = " ".join(rec.get("vae_buffer_where") or [])
    return "RAM" if "RAM" in w and "VRAM" not in w else "VRAM"


def _ok(rec: dict) -> bool:
    return rec.get("rc") == 0 and bool(rec.get("png"))


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
        ch = (o.get("choice") or {}).get("vulkan") or {}
        out.append((f"{name}: state's own Vulkan offload flags resolved", bool(ch.get("flags")),
                    f"{ch.get('chooser')}: {ch.get('flags') or str(ch.get('error'))[-300:]}"))
        refs = [r for r in _runs(o).values() if r.get("tile_passed") is None and r.get("backend") == "vulkan"]
        out.append((f"{name}: untiled Vulkan reference rendered at every size",
                    {r["size"] for r in refs if _ok(r)} >= set(o.get("sizes") or [0]),
                    str(sorted({(r['size'], r.get('rc')) for r in refs}))))
    return out


def _f(x, nd = 2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def table(obs: dict) -> str:
    rows = ["| state | arm | backend | size | tile passed | rc | decode s | VAE buffer MiB (where) | table MiB | ratio | "
            "tiles | reduce/OOM | PSNR vs untiled | cached |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        o = obs.get(name) or {}
        tab = _table(obs.get("head") or {})
        for a in o.get("arms") or []:
            if name != "base" and not a["name"].startswith("state_"):
                continue  # forced arms are shared: shown once
            r = _runs(o).get(a["key"]) or {}
            buf = r.get("vae_buffer_mib")
            t = tab.get(a["size"])
            rows.append(f"| {name} | {a['name']} | {a['backend']} | {a['size']} | {_f(a['tile_passed'])} | {r.get('rc')} | "
                        f"{_f(r.get('decode_s'))} | {_f(buf, 0)} ({' '.join(r.get('vae_buffer_where') or [])}) | {t} | "
                        f"{_f(buf / t if buf and t else None)} | {r.get('vae_tiles')} | "
                        f"{'; '.join(r.get('reduce_or_oom') or []) or '-'} | {_f(a.get('psnr_vs_untiled'))} | "
                        f"{bool(r.get('cached'))} |")
    rows += ["", "| state | class | chooser | card free / limit MiB | host spare MiB | flags |", "|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        for klass, ch in ((obs.get(name) or {}).get("choice") or {}).items():
            rows.append(f"| {name} | {klass} | {ch.get('chooser')} | {ch.get('card_free_mib')} / {ch.get('card_limit_mib')} | "
                        f"{ch.get('host_spare_mib')} | `{' '.join(ch.get('flags') or [])}` |")
    rows += ["", "Untiled references (same argv minus tiling and --vae-on-cpu):", "",
             "| backend | size | rc | decode s | VAE buffer MiB (where) | reduce/OOM |", "|---|---|---|---|---|---|"]
    seen = set()
    for name in ("base", "head"):
        for r in _runs(obs.get(name) or {}).values():
            if r.get("tile_passed") is None and r["key"] not in seen:
                seen.add(r["key"])
                rows.append(f"| {r['backend']} | {r['size']} | {r.get('rc')} | {_f(r.get('decode_s'))} | "
                            f"{_f(r.get('vae_buffer_mib'), 0)} ({' '.join(r.get('vae_buffer_where') or [])}) | "
                            f"{'; '.join(r.get('reduce_or_oom') or []) or '-'} |")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    for a in _arms(base, "state_base_vulkan"):
        r = _runs(base).get(a["key"]) or {}
        tile = a.get("tile_passed")
        if _ok(r) and tile is not None and tile < a["size"] and (a.get("psnr_vs_untiled") or 99) < PSNR_FLOOR:
            return True
    return False


def _state_fixed(o: dict, prefix: str) -> tuple[bool, str]:
    arms = _arms(o, prefix)
    sizes = set(o.get("sizes") or [])
    if not arms or {a["size"] for a in arms} < sizes:
        return False, f"{prefix}: not run at every size"
    tab = _table(o)
    for a in arms:
        r = _runs(o).get(a["key"]) or {}
        tile, size = a.get("tile_passed"), a["size"]
        if not _ok(r):
            return False, f"{size}px: rc {r.get('rc')}"
        if tile is None or tile < size:
            return False, f"{size}px: head passed tile {tile}, not a whole tile"
        if r.get("reduce_or_oom"):
            return False, f"{size}px: sd.cpp reduced/retried: {r['reduce_or_oom']}"
        if (a.get("psnr_vs_untiled") or 0) < PSNR_FLOOR:
            return False, f"{size}px: PSNR vs untiled {a.get('psnr_vs_untiled')}"
        buf, t = r.get("vae_buffer_mib"), tab.get(size)
        if buf is None or t is None:
            return False, f"{size}px: no VAE buffer line ({buf}) or no table value ({t})"
        if buf > t * MARGIN[_where(r)]:
            return False, f"{size}px: VAE buffer {buf:.0f} MiB > table {t} x {MARGIN[_where(r)]}"
    return True, "whole tile, OK, matches untiled, buffer within table x margin at every size"


def head_is_fixed(head: dict) -> bool:
    return _state_fixed(head, "state_head_vulkan")[0]


def state_is_fixed(name: str, obs: dict):
    ok, why = _state_fixed(obs, f"state_{name}_vulkan")
    return ok, why
