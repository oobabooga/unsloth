#!/usr/bin/env python3
"""Criteria for probes/wanrep_probe.py (diagnosis, head only).

Gates: the default arms D1..D3 rendered. REGRESSION (= the non-reproducibility CONFIRMED) when any two default
processes give different final frames. The table lists, per comparison, identical / first divergent stage and the
per-step magnitudes, so the switch arms can be read off directly."""
from __future__ import annotations

TITLE = "gfx1151 Linux: Wan2.2-TI2V-5B cross-process reproducibility (INT8 attention off), diagnosis"
MODE = "regression"
NEEDS = ["gpu", "rocm", "nvidia", "windows", "windows_rocm_wddm"]


def _head(obs: dict) -> dict:
    return obs.get("head") or {}


def gates(obs: dict) -> list:
    arms = _head(obs).get("arms") or {}
    out = []
    for t in ("D1", "D2", "D3"):
        r = arms.get(t) or {}
        out.append((f"{t} rendered", r.get("verdict") == "ok" and len(r.get("renders") or []) >= 1,
                    f"{r.get('verdict')} {str(r.get('error') or '')[:300]}"))
    return out


def head_is_worse(base: dict, head: dict) -> tuple:
    comp = head.get("comparisons") or {}
    lines, bad = [], False
    for k in ("D1r0|D2r0", "D1r0|D3r0", "D1r1|D2r1", "D1r1|D3r1"):
        c = comp.get(k)
        if c is None:
            continue
        if not c.get("identical"):
            bad = True
        lines.append(f"{k}: {'identical' if c.get('identical') else 'DIFFER'} first={c.get('first_divergence')} "
                     f"mad={c.get('frames_mad')}")
    return bad, "; ".join(lines) or "no comparisons"


def table(obs: dict) -> str:
    h = _head(obs)
    rows = ["| arm | verdict | load s | render s | frames sha | flash buckets | best_configs | tunableop |",
            "|---|---|---|---|---|---|---|---|"]
    for t, r in (h.get("arms") or {}).items():
        rows.append(f"| {t} | {r.get('verdict')} | {r.get('load_s')} | "
                    f"{[x.get('wall_s') for x in r.get('renders') or []]} | "
                    f"{[x.get('frames_sha') for x in r.get('renders') or []]} | "
                    f"{len(r.get('flash_buckets') or {}) if isinstance(r.get('flash_buckets'), dict) else r.get('flash_buckets')} | "
                    f"{r.get('n_best_configs')} | {r.get('tunableop')} |")
    rows += ["", "| pair | identical | first divergence | step rel L2 (model_output) | frames MAD / PSNR |",
             "|---|---|---|---|---|"]
    for k, c in (h.get("comparisons") or {}).items():
        steps = [round(s.get("model_output"), 6) if isinstance(s.get("model_output"), float) else s.get("model_output")
                 for s in c.get("steps") or []]
        rows.append(f"| {k} | {c.get('identical')} | {c.get('first_divergence')} | {steps} | "
                    f"{c.get('frames_mad')} / {c.get('frames_psnr')} |")
    rows += ["", f"inductor pick diffs: {h.get('inductor_pick_diff')}"]
    return "\n".join(rows)
