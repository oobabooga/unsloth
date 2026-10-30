#!/usr/bin/env python3
"""Criteria (T7h.3): GPT-OSS inference on gfx1151 at the HEAD (zoo main with #2153 / #2161 / #2163 / #2166): flex is the
default inference route, the eager kill switch restores eager, greedy tokens agree, flex prefill memory is below eager,
and load_in_4bit generate + 3 SFT steps work. Pairs with probes/t7h3_gptoss_probe.py (head only; the base is skipped).

Gates (INCONCLUSIVE when any fails): the scenario ran at the head from its own zoo checkout on a HIP gfx1151 GPU, and the
route counter saw forwards in both arms (so the arm comparison is not vacuous).
REGRESSION (head-only judgement, the base records a skip): any of
  - flex arm did not route through flex, or the kill switch arm did (route counts);
  - greedy tokens differ between arms where every step's top-2 margin exceeds 1e-3 (a real disagreement, not a tie);
  - flex prefill peak >= eager prefill peak at 4k (no memory saving);
  - 4-bit generate or the 3 SFT steps raised, or an SFT loss / grad norm is non-finite.
"""

from __future__ import annotations

import math
import os

TITLE = "T7h.3: GPT-OSS flex inference, eager kill switch, load_in_4bit generate + SFT on gfx1151 (zoo head)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "discrete_gpu", "nvidia", "multi_gpu", "mig", "xpu", "mlx"]


def _norm(p):
    return os.path.normcase(os.path.abspath(p)) if p else ""


def _rounds(head):
    return ((head.get("scenario") or {}).get("sections") or {}).get("flex_random", {}).get("rounds") or []


def gates(obs: dict) -> list[tuple[str, bool, str]]:
    head = obs.get("head") or {}
    sc = head.get("scenario") or {}
    t = sc.get("torch") or {}
    out = [("head: scenario ran", head.get("rc") == 0 and bool(sc.get("sections")),
            f"rc={head.get('rc')} done={sc.get('done')} {str(head.get('stderr_tail'))[-300:]}")]
    f, co = head.get("unsloth_zoo_file") or "", head.get("checkout") or ""
    out.append(("head: unsloth_zoo imported from its own checkout", bool(f) and _norm(f).startswith(_norm(co)),
                f"{f} (checkout {co}, zoo {str(head.get('zoo_sha'))[:9]})"))
    out.append(("head: HIP gfx1151 GPU", bool(t.get("is_available")) and bool(t.get("hip")) and "gfx1151" in str(t.get("arch")),
                f"{t}"))
    rs = _rounds(head)
    seen = all(sum((r.get("route_counts") or {}).values()) > 0 for r in rs) and len(rs) >= 2
    out.append(("head: route counter saw forwards in every round", seen,
                "; ".join(f"{r.get('arm')} {r.get('route_counts')}" for r in rs) or "no rounds"))
    return out


def _tokens_disagree(a, b, margin):
    return a is not None and b is not None and a != b and (margin is None or margin > 1e-3)


def head_is_worse(base: dict, head: dict) -> tuple[bool, str]:
    sec = (head.get("scenario") or {}).get("sections") or {}
    bad, notes = [], []
    rs = sec.get("flex_random", {}).get("rounds") or []
    for r in rs:
        rc = r.get("route_counts") or {}
        if r.get("arm") == "flex" and (rc.get("flex", 0) == 0 or rc.get("eager", 0) > 0):
            bad.append(f"flex arm route counts {rc}")
        if r.get("arm") == "eager" and rc.get("flex", 0) > 0:
            bad.append(f"kill-switch arm still routed flex {rc}")
        for k, v in r.items():
            if k.endswith("_error"):
                bad.append(f"{r.get('arm')} {k}: {str(v)[:200]}")
    eg = [r for r in rs if r.get("arm") == "eager"]
    fx = [r for r in rs if r.get("arm") == "flex"]
    if eg and fx:
        m = min(x for x in [eg[0].get("min_margin"), fx[0].get("min_margin")] if x is not None) if any(
            x is not None for x in [eg[0].get("min_margin"), fx[0].get("min_margin")]) else None
        if _tokens_disagree(eg[0].get("tokens"), fx[0].get("tokens"), m):
            bad.append(f"random-model greedy tokens differ (min top-2 margin {m})")
        pe = [r.get("prefill_4096_peak_mib") for r in eg if r.get("prefill_4096_peak_mib") is not None]
        pf = [r.get("prefill_4096_peak_mib") for r in fx if r.get("prefill_4096_peak_mib") is not None]
        if pe and pf:
            notes.append(f"prefill 4k peak MiB eager {pe} vs flex {pf}")
            if min(pf) >= min(pe):
                bad.append("flex prefill 4k peak not below eager")
        p2e = [r.get("prefill_2048_peak_mib") for r in eg]
        p2f = [r.get("prefill_2048_peak_mib") for r in fx]
        notes.append(f"prefill 2k peak MiB eager {p2e} vs flex {p2f}")
    t4 = sec.get("tiny_4bit") or {}
    if t4.get("error"):
        bad.append(f"tiny 4-bit: {str(t4['error'])[:300]}")
    ge, gf = t4.get("gen_eager") or {}, t4.get("gen_flex") or {}
    for arm, g in (("eager", ge), ("flex", gf)):
        if g.get("error"):
            bad.append(f"4-bit generate {arm}: {str(g['error'])[:300]}")
    if (gf.get("route_counts") or {}).get("flex", 0) == 0 and not gf.get("error") and gf:
        bad.append(f"4-bit flex arm never routed flex {gf.get('route_counts')}")
    if _tokens_disagree(ge.get("tokens"), gf.get("tokens"), min([x for x in (ge.get("min_margin"), gf.get("min_margin")) if x is not None] or [None]) if any(x is not None for x in (ge.get("min_margin"), gf.get("min_margin"))) else None):
        bad.append("4-bit greedy tokens differ between arms")
    sft = t4.get("sft") or {}
    if sft.get("error"):
        bad.append(f"4-bit SFT: {str(sft['error'])[:300]}")
    elif sft:
        vals = (sft.get("losses") or []) + (sft.get("grad_norms") or [])
        if len(sft.get("losses") or []) < 3 or not all(math.isfinite(v) for v in vals):
            bad.append(f"4-bit SFT non-finite or short: {sft}")
        notes.append(f"4-bit SFT losses {sft.get('losses')} grad norms {sft.get('grad_norms')}")
    notes.append(f"Linear4bit modules {t4.get('linear4bit_modules')}, bitsandbytes {t4.get('bitsandbytes') or t4.get('bitsandbytes_error')}")
    if bad:
        return True, "; ".join(bad) + " | " + "; ".join(notes)
    return False, "head-only check passed: " + "; ".join(notes)


def table(obs: dict) -> str:
    head = obs.get("head") or {}
    sec = (head.get("scenario") or {}).get("sections") or {}
    tr = sec.get("triton") or {}
    rows = ["| item | value |", "|---|---|",
            f"| triton | {tr.get('version')} {tr.get('dists')} target {tr.get('target')} {tr.get('error') or ''} |",
            f"| Triton kernel launch | {tr.get('kernel_ok')} |",
            f"| torch.compile on GPU | {tr.get('compile_gpu_ok')} ({tr.get('compile_gpu_s')} s) {tr.get('compile_gpu_error') or ''} |"]
    for r in sec.get("flex_random", {}).get("rounds") or []:
        rows.append(f"| random GPT-OSS {r.get('arm')} | routes {r.get('route_counts')}, prefill 2k "
                    f"{r.get('prefill_2048_peak_mib')} MiB, 4k {r.get('prefill_4096_peak_mib')} MiB, "
                    f"min margin {r.get('min_margin')}, tokens[:8] {(r.get('tokens') or [])[:8]} |")
    t4 = sec.get("tiny_4bit") or {}
    for arm in ("eager", "flex"):
        g = t4.get(f"gen_{arm}") or {}
        rows.append(f"| tiny 4-bit generate {arm} | routes {g.get('route_counts')}, tokens {g.get('tokens')}, "
                    f"min margin {g.get('min_margin')} {str(g.get('error') or '')[:200]} |")
    rows.append(f"| tiny 4-bit SFT | {t4.get('sft')} |")
    return "\n".join(rows)
