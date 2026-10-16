#!/usr/bin/env python3
"""Criteria: on-demand SSM kernel install, 9f5b13325 (base, setup installed the kernels) vs
PR 13971 head 7092b2bdf (head, on demand). `main` is recorded (decisions only), not judged.

Regression mode. The head is worse when, for any model, on this real hardware:
  - a load / train / decode that worked at the base fails at the head;
  - the head's install blocks start a source build (a non-wheel `pip/uv install` of
    causal-conv1d / mamba-ssm) the base did not start for that model;
  - a model needing no kernel spends > 2 s more in the head's install blocks than the base's;
  - with the same kernels installed, peak memory rises > 5 % or load-to-first-step rises more than
    max(3 x the base's repeat spread, 25 %).
"""

from __future__ import annotations

import math
import re

TITLE = "On-demand SSM kernels: 9f5b13325 (setup-time) vs 7092b2bdf (on demand)"
MODE = "regression"
NEEDS: list[str] = ["rocm", "gpu", "linux", "windows", "nvidia"]

_SRC = re.compile(r"(pip|uv)\b.*\binstall\b.*\b(causal[-_]conv1d|mamba[-_]ssm)\b", re.I)


def _data(r) -> dict:
    return (r or {}).get("data") or {}


def _calls(m: dict) -> list:
    out = []
    for part in ("infer", "train_block"):
        out += _data(m.get(part)).get("calls") or []
    return out


def _source_builds(m: dict) -> list[str]:
    hits = []
    for c in _calls(m):
        cmd = c.get("run") or c.get("popen") or ""
        if _SRC.search(cmd) and ".whl" not in cmd and "uninstall" not in cmd:
            hits.append(cmd[:160])
    return hits


def _block_s(m: dict) -> float:
    return sum(float(_data(m.get(p)).get("seconds") or 0) for p in ("infer", "train_block"))


def _runs(m: dict) -> list[dict]:
    return [_data(r) for r in (m.get("run") or [])]


def _ok(run: dict) -> bool:
    ls = run.get("losses") or []
    return (len(ls) == 4 and all(math.isfinite(x) for x in ls) and not run.get("load_error")
            and not run.get("train_error"))


def _kset(m: dict) -> tuple:
    k = _data(m.get("kernels"))
    return (k.get("causal-conv1d"), k.get("mamba-ssm"))


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        ms = o.get("models") or {}
        plain = ms.get("plain") or {}
        r = _runs(plain)
        out.append((f"{name} probe produced per-model results", bool(ms),
                    f"rc={o.get('_probe_rc')} models={list(ms)}"))
        out.append((f"{name} plain model trained 4 finite steps", bool(r) and _ok(r[0]),
                    f"{(r[0] if r else {}).get('losses')} {(r[0] if r else {}).get('load_error') or ''}"))
        out.append((f"{name} setup step ran", _data(o.get("setup")).get("error") is None and bool(_data(o.get("setup"))),
                    f"{_data(o.get('setup')).get('error')} rc={(o.get('setup') or {}).get('rc')}"))
        out.append((f"{name} started with no SSM kernels", _data(o.get("kernels_before")).get("causal-conv1d") is None
                    and _data(o.get("kernels_before")).get("mamba-ssm") is None, str(_data(o.get("kernels_before")))))
    return out


def table(obs: dict) -> str:
    lines = []
    st = ["base", "main", "head"]
    lines.append("### Decisions per state (kernels requested at load)")
    lines.append("| model | " + " | ".join(st) + " |")
    lines.append("|---|" + "---|" * len(st))
    keys = list(((obs.get("head") or {}).get("decide") or {}).keys())
    for k in keys:
        cells = []
        for s in st:
            d = _data(((obs.get(s) or {}).get("decide") or {}).get(k)).get("decision") or {}
            cells.append(f"cc1d={d.get('cc1d')} ssm={d.get('ssm')} req={d.get('required')}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("")
    for s in ("base", "head"):
        o = obs.get(s) or {}
        sd = _data(o.get("setup"))
        lines.append(f"### {s} ({str(o.get('commit'))[:9]}) setup SSM step: {sd.get('seconds')} s, "
                     f"calls={len(sd.get('calls') or [])}, error={sd.get('error')}, "
                     f"kernels after setup={_data(o.get('kernels_after_setup'))}, hipcc={o.get('hipcc')}, "
                     f"state wall={o.get('state_wall_s')} s")
        lines.append("")
        lines.append("| model | infer ok | install blocks s | source builds | kernels after | path | ok | "
                     "load->1st step s | losses | peak GB | decode | kernels seen |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, m in (o.get("models") or {}).items():
            if "skipped" in m:
                lines.append(f"| {k} | skipped: {m['skipped']} |||||||||||")
                continue
            inf = _data(m.get("infer"))
            tb = _data(m.get("train_block"))
            runs = _runs(m)
            r0 = runs[0] if runs else {}
            l1 = "/".join(str(x.get("load_to_first_step_s")) for x in runs)
            err = inf.get("error") or tb.get("error") or r0.get("load_error") or r0.get("train_error") or ""
            lines.append(
                f"| {k} | {inf.get('ok')} | {_block_s(m):.1f} | {len(_source_builds(m))} | {_kset(m)} | "
                f"{r0.get('path')} | {_ok(r0)} {err[:80]} | {l1} | "
                f"{[round(x, 4) for x in (r0.get('losses') or [])]} | "
                f"{round(r0.get('peak_alloc_gb') or 0, 3)} | "
                f"{'ok' if r0.get('decode_tokens') else (r0.get('decode_error') or '')[:60]} | "
                f"{', '.join((r0.get('kernels') or [])[:3])[:120]} |")
        lines.append("")
    return "\n".join(lines)


def head_is_worse(base: dict, head: dict):
    bad = []
    bm, hm = base.get("models") or {}, head.get("models") or {}
    for k, b in bm.items():
        h = hm.get(k) or {}
        if "skipped" in b or "skipped" in h:
            if "skipped" in h and "skipped" not in b:
                bad.append(f"{k}: head ran out of budget, base did not")
            continue
        br, hr = _runs(b), _runs(h)
        if br and hr and _ok(br[0]) and not _ok(hr[0]):
            bad.append(f"{k}: trains at base, fails at head ({hr[0].get('load_error') or hr[0].get('train_error')})")
        if br and hr and br[0].get("decode_tokens") and not hr[0].get("decode_tokens"):
            bad.append(f"{k}: decode works at base, fails at head")
        if _data(b.get("infer")).get("ok") is not False and _data(h.get("infer")).get("ok") is False:
            bad.append(f"{k}: inference SSM step fails at head only: {_data(h.get('infer')).get('responses')}")
        if _data(h.get("train_block")).get("error") and not _data(b.get("train_block")).get("error"):
            bad.append(f"{k}: training SSM block raises at head only: {_data(h.get('train_block')).get('error')}")
        nb, nh = len(_source_builds(b)), len(_source_builds(h))
        if nh > nb:
            bad.append(f"{k}: head starts {nh} source build(s), base {nb}; "
                       f"{_block_s(h):.0f} s vs {_block_s(b):.0f} s in install blocks")
        if k == "plain" and _block_s(h) - _block_s(b) > 2.0:
            bad.append(f"plain: install blocks {_block_s(h):.2f} s vs {_block_s(b):.2f} s")
        if _kset(b) == _kset(h) and br and hr and _ok(br[0]) and _ok(hr[0]):
            pb, ph = br[0].get("peak_alloc_gb") or 0, hr[0].get("peak_alloc_gb") or 0
            if pb and ph > pb * 1.05:
                bad.append(f"{k}: peak {ph:.3f} GB vs {pb:.3f} GB")
            tb = [x.get("load_to_first_step_s") for x in br if x.get("load_to_first_step_s")]
            th = [x.get("load_to_first_step_s") for x in hr if x.get("load_to_first_step_s")]
            if tb and th:
                spread = (max(tb) - min(tb)) if len(tb) > 1 else 0
                allow = max(3 * spread, 0.25 * min(tb))
                if min(th) - min(tb) > allow:
                    bad.append(f"{k}: load->first step {min(th):.1f} s vs {min(tb):.1f} s (allow {allow:.1f})")
    return (bool(bad), "; ".join(bad) if bad else
            "every model that loads, trains and decodes at the base does so at the head, with no new "
            "source build, no added latency for kernel-free models, and equal memory")
