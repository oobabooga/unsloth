#!/usr/bin/env python3
"""Criteria: does the whole-map decode remove the per-tile GroupNorm colour shift of a FLUX.2-class VAE decoding a
latent bigger than the stock tile? Judges probes/flux2_full_map_probe.py (base = origin/main, head = + full-map patch).

base_shows_defect: the base tiled decode (tiled_decode taken) is < 35 dB PSNR vs the untiled decode.
head_is_fixed: the head tiled decode is >= 45 dB vs untiled with the full-map counter >= 1; the head kill-switch arm
(UNSLOTH_FLUX_VAE_FULL_MAP=0) is < 35 dB with the counter 0; the small (untiled-path) decode is bit-identical base vs
head.
Correctness only (``--shared-gpu``): peak MiB and times are informative.
"""

from __future__ import annotations

TITLE = "gfx1151: FLUX.2 VAE whole-map decode of a would-be-tiled 2048 px latent (correctness, shared GPU)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "xpu", "mlx", "windows", "discrete_gpu"]

BAD_DB = 35.0
FIXED_DB = 45.0
SANE_PHOTO_DB = 20.0
PEAK_SLACK = 1.25

_SEEN: dict = {}


def _states(obs: dict) -> dict:
    return {n: v for n, v in obs.items() if not n.startswith("_")}


def _d(state: dict, name: str) -> dict:
    return state.get("decodes", {}).get(name, {})


def _db(state: dict, name: str, key: str = "psnr_vs_untiled") -> float:
    v = _d(state, name).get(key)
    return float(v) if isinstance(v, (int, float)) else float("nan")


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    st = _states(obs)
    _SEEN.clear()
    _SEEN.update(st)
    names = ("untiled", "tiled", "tiled_kill", "small", "small_aa")
    out = [("runner is gfx1151 with ROCm torch in every state",
            bool(st) and all(v.get("arch") == "gfx1151" and v.get("hip") for v in st.values()),
            ", ".join(f"{k}={v.get('arch')} hip={v.get('hip')} torch={v.get('torch')} "
                      f"diffusers={v.get('diffusers')}" for k, v in st.items()))]
    out.append(("no setup error; every decode ran, finite, in every state",
                all("setup_error" not in v and "error" not in v
                    and all(_d(v, n).get("finite") is True for n in names) for v in st.values()),
                "; ".join(f"{k}: setup={str(v.get('setup_error', 'ok'))[:160]} err={str(v.get('error', 'ok'))[:200]} "
                          + " ".join(f"{n}={str(_d(v, n).get('error', _d(v, n).get('finite')))[:120]}" for n in names)
                          for k, v in st.items())))
    b, h = st.get("base", {}), st.get("head", {})
    own = (b.get("has_full_map_hook") is False and h.get("has_full_map_hook") is True
           and all(str(v.get("module_file", "")).startswith(str(v.get("checkout", "\0"))) for v in st.values()))
    out.append(("each state imports its own tree (base: no full_map_decode hook, head: hook; module under checkout)",
                own, ", ".join(f"{k}: {v.get('module_file')} hook={v.get('has_full_map_hook')}"
                               for k, v in st.items())))
    out.append(("the VAE class is AutoencoderKLFlux2 and the tiled wrapper installed in every state",
                all(v.get("vae_class") == "AutoencoderKLFlux2" and v.get("installed") is True for v in st.values()),
                ", ".join(f"{k}: {v.get('vae_class')} installed={v.get('installed')} stock_tile={v.get('stock_tile_latent')}"
                          for k, v in st.items())))
    out.append(("the tiled path was taken by the large decode and not by the small one, in every state",
                all(_d(v, n).get("tiled_decode_calls", 0) >= 1 for v in st.values() for n in ("tiled", "tiled_kill"))
                and all(_d(v, n).get("tiled_decode_calls") == 0 for v in st.values() for n in ("small", "small_aa")),
                "; ".join(f"{k}: " + " ".join(f"{n}={_d(v, n).get('tiled_decode_calls')}" for n in names[1:])
                          for k, v in st.items())))
    out.append((f"the untiled decode reproduces the photo (>= {SANE_PHOTO_DB} dB) in every state",
                all(_db(v, "untiled", "psnr_vs_photo") >= SANE_PHOTO_DB for v in st.values()),
                ", ".join(f"{k}: {_db(v, 'untiled', 'psnr_vs_photo')} dB" for k, v in st.items())
                + f"; latent equal: {b.get('latent_sha256') == h.get('latent_sha256')}, "
                  f"small latent equal: {b.get('small_latent_sha256') == h.get('small_latent_sha256')}"))
    ht = _d(h, "tiled")
    est, peak = ht.get("full_map_decode_mib"), ht.get("peak_alloc_mib")
    out.append(("head tiled (whole-map) decode peak allocated <= 1.25 x its full_map_decode_mib estimate",
                isinstance(est, (int, float)) and isinstance(peak, (int, float)) and peak <= PEAK_SLACK * est,
                f"peak {peak} MiB, estimate {est} MiB, ratio "
                f"{(peak / est) if isinstance(est, (int, float)) and est and isinstance(peak, (int, float)) else '-'}"))
    return out


def table(obs: dict) -> str:
    st = _states(obs)
    rows = ["| state | decode | PSNR vs untiled dB | PSNR vs photo dB | full-map calls | tiled_decode calls | "
            "peak alloc MiB | estimate MiB | chunked SDPA calls | free MiB before | time s | sha256[:12] |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in st.items():
        for dn, d in v.get("decodes", {}).items():
            rows.append(f"| {name} | {dn} | {d.get('psnr_vs_untiled', '-')} | {d.get('psnr_vs_photo', '-')} | "
                        f"{d.get('full_map_calls', '-')} | {d.get('tiled_decode_calls', '-')} | "
                        f"{d.get('peak_alloc_mib', '-')} | {d.get('full_map_decode_mib', '-')} | {d.get('chunked_sdpa_calls', '-')} | "
                        f"{d.get('free_mib_before', '-')} | {d.get('time_s', '-')} | "
                        f"{str(d.get('sha256', d.get('error', '-')))[:12]} |")
    rows.append("")
    for name, v in st.items():
        rows.append(f"- {name}: arch={v.get('arch')} torch={v.get('torch')} hip={v.get('hip')} "
                    f"diffusers={v.get('diffusers')} fused_runtime_ok={v.get('fused_runtime_ok')} "
                    f"latent={v.get('latent_shape')} stock_tile={v.get('stock_tile_latent')} "
                    f"full_map_mib_estimate={v.get('full_map_decode_mib_estimate', '-')} elapsed={v.get('elapsed_s')} s")
    if "base" in st and "head" in st:
        b, h = st["base"], st["head"]
        rows.append(f"- small ({b.get('small_size')} px) decode sha256 equal base vs head: "
                    f"{_d(b, 'small').get('sha256') == _d(h, 'small').get('sha256')}; in-state A/A equal: "
                    f"base {_d(b, 'small').get('sha256') == _d(b, 'small_aa').get('sha256')}, "
                    f"head {_d(h, 'small').get('sha256') == _d(h, 'small_aa').get('sha256')}")
        rows.append(f"- untiled decode sha256 equal base vs head: "
                    f"{_d(b, 'untiled').get('sha256') == _d(h, 'untiled').get('sha256')}; "
                    f"head kill-switch tiled == base tiled: "
                    f"{_d(b, 'tiled').get('sha256') == _d(h, 'tiled_kill').get('sha256')}")
    rows.append("")
    rows.append(f"Defect: tiled < {BAD_DB} dB vs untiled. Fixed: head tiled >= {FIXED_DB} dB with the counter >= 1, "
                f"head kill switch < {BAD_DB} dB with the counter 0, small decode bit-identical base vs head. "
                "Shared GPU: peak MiB and times are informative only.")
    return "\n".join(rows)


def base_shows_defect(base: dict) -> bool:
    t = _d(base, "tiled")
    return t.get("tiled_decode_calls", 0) >= 1 and _db(base, "tiled") < BAD_DB


def head_is_fixed(head: dict) -> bool:
    base = _SEEN.get("base", {})
    t, k = _d(head, "tiled"), _d(head, "tiled_kill")
    small_h, small_b = _d(head, "small").get("sha256"), _d(base, "small").get("sha256")
    return (_db(head, "tiled") >= FIXED_DB and t.get("full_map_calls", 0) >= 1
            and _db(head, "tiled_kill") < BAD_DB and k.get("full_map_calls") == 0
            and small_h is not None and small_h == small_b)
