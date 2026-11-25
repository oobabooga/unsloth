#!/usr/bin/env python3
"""Criteria for probes/r3_probe.py: gfx1151 measurements R3-05..R3-08 on the head tree. Nothing differs between
states (arms are env / config inside the head), so nothing is judged between them: the gates prove each item ran on
gfx1151 and the table carries the numbers; keep / kill is decided from the observations by the reader."""

from __future__ import annotations

import math
import statistics as st

TITLE = "gfx1151: R3-05..R3-08 diffusion inference measurements (VAE NHWC, W8A8 tiles, INT8 attention, pread load)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "windows", "multi_gpu"]


def _ok(rs):
    return [r for r in rs or [] if not r.get("error") and r.get("rc") == 0 and not r.get("missing")]


def gates(obs: dict) -> list:
    h = obs.get("head") or {}
    arch = (h.get("env") or {}).get("arch") or ""
    conf = _ok(h.get("gemm_confirm"))
    return [("gfx1151", arch.startswith("gfx1151"), arch),
            ("gemm confirm arms", len(conf) >= 4, f"{len(conf)} ok"),
            ("attn arms", len(_ok(h.get("attn"))) >= 12, f"{len(_ok(h.get('attn')))} ok"),
            ("vae arms", len(_ok(h.get("vae"))) >= 9, f"{len(_ok(h.get('vae')))} ok"),
            ("load arms", len(_ok(h.get("load"))) >= 4 or bool(h.get("load_skip")),
             f"{len(_ok(h.get('load')))} ok, skip={h.get('load_skip')}")]


def head_is_worse(base: dict, head: dict) -> tuple:
    return False, "measurement only (head tree, arms differ by env / config)"


def _g(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def table(obs: dict) -> str:
    h = obs.get("head") or {}
    L = [f"env {h.get('env')}", ""]
    for r in _ok(h.get("gemm_confirm")):
        rows = [x for x in r.get("rows") or [] if x.get("studio_ms") and x.get("arm_ms")]
        L.append(f"gemm {r.get('arm')}: geomean studio/arm {_g([x['studio_ms'] / x['arm_ms'] for x in rows])} "
                 f"exact {all(x.get('exact') for x in rows)} n {len(rows)}")
    by: dict = {}
    for r in _ok(h.get("attn")):
        by.setdefault((r["shape"], r["arm"]), []).append(r)
    for (shape, arm), rs in sorted(by.items()):
        L.append(f"attn {shape} {arm}: med ms {[round(x['med_ms'], 1) for x in rs]} rel_l2 "
                 f"{[round(x['rel_l2'], 5) for x in rs]}")
    byv: dict = {}
    for r in _ok(h.get("vae")):
        byv.setdefault((r["model"], r["arm"]), []).append(r)
    for (m, arm), rs in sorted(byv.items()):
        L.append(f"vae {m} {arm}: med s {[round(x['med_s'], 3) for x in rs]} cl inputs "
                 f"{[x.get('conv_cl_inputs') for x in rs]}/{rs[0].get('conv_calls')}")
    L.append(f"vae psnr {h.get('vae_psnr')}")
    for r in _ok(h.get("load")):
        L.append(f"load {r['arm']}: {r.get('load_s'):.1f} s rss {r.get('peak_rss_gib'):.2f} GiB resident_before "
                 f"{r.get('resident_before')} checksum {r.get('checksum')}")
    if h.get("load_skip"):
        L.append(f"load skipped: {h['load_skip']}")
    return "\n".join(L)


_ = st
