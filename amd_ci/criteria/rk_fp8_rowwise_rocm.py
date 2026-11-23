#!/usr/bin/env python3
"""Criteria: FP8 rowwise out_dtype on ROCm with torch >= 2.14 (unsloth #16598 follow-up; judges
probes/rk_fp8_rowwise_rocm_probe.py; base = origin/main, head = base + the fix diff).

The defect: on ROCm with torch >= 2.14, unsloth's routing sends fp16 / fp32 activations to torch._scaled_mm with an
fp16 / fp32 out_dtype, and hipBLASLt's rowwise path raises ValueError ("hipblaslt rowwise _scaled_mm only supports
BFloat16 output").

base_shows_defect: under some torch >= 2.14 ROCm build, the base's own routing (from its checkout) sends fp16 or fp32
to torch._scaled_mm AND a raw rowwise _scaled_mm with that out_dtype raises the hipBLASLt bf16-only error on this card.
On a card torch does not let reach _scaled_mm at all (its arch gate: gfx942 / gfx950 / gfx12xx / gfx1250 only) the
base routes everything to dequant, the defect cannot occur, and the verdict is VOID by design.
head_is_fixed: under every torch, the head routes no fp16 / fp32 activation to a _scaled_mm out_dtype the raw call
refused on this card.
"""

from __future__ import annotations

TITLE = "gfx1151 Linux: FP8 rowwise _scaled_mm out_dtype on ROCm (torch Studio venv and torch >= 2.14)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "linux", "exclusive_gpu", "discrete_gpu", "gpu_partitions"]
HIPBLASLT_BF16_ONLY = "hipblaslt rowwise _scaled_mm only supports BFloat16 output"


def _runs(o):
    return list(((o or {}).get("runs") or {}).values())


def _ver(t):
    try:
        return tuple(int(x) for x in str(t).split("+")[0].split(".")[:2])
    except Exception:  # noqa: BLE001
        return (0, 0)


def _one(x, n):
    return str(x or "").replace("\n", " ").replace("|", "/")[:n]


def gates(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out = []
    for name, o in (("base", b), ("head", h)):
        runs = _runs(o)
        out.append((f"{name}: every interpreter ran on Linux ROCm gfx1151",
                    bool(runs) and all(r.get("rc") == 0 and not r.get("error") and r.get("platform") == "linux"
                                       and r.get("hip") and str(r.get("arch", "")).startswith("gfx1151") for r in runs),
                    "; ".join(f"{r.get('torch')} hip {r.get('hip')} {r.get('arch')} rc {r.get('rc')} "
                              f"{_one(r.get('error'), 150)}" for r in runs)))
        out.append((f"{name}: raw rowwise _scaled_mm attempted for bf16 / fp16 / fp32",
                    bool(runs) and all(all(k in (r.get("raw") or {}) for k in ("rowwise_bf16", "rowwise_fp16",
                                                                                 "rowwise_fp32")) for r in runs),
                    f"{len(runs)} interpreter(s)"))
        out.append((f"{name}: routing extracted from this checkout",
                    bool(runs) and all("_probe_fp8_rowwise_backend" in (r.get("extracted") or []) for r in runs),
                    str([r.get("extracted") for r in runs])[:200]))
    out.append(("a torch >= 2.14 ROCm build was among the interpreters",
                any(_ver(r.get("torch")) >= (2, 14) and r.get("hip") for r in _runs(h)),
                str([r.get("torch") for r in _runs(h)])))
    out.append(("same torch builds in both states",
                sorted(str(r.get("torch")) for r in _runs(b)) == sorted(str(r.get("torch")) for r in _runs(h)),
                f"{[r.get('torch') for r in _runs(b)]} / {[r.get('torch') for r in _runs(h)]}"))
    out.append(("head carries the fix, base does not",
                all((r.get("decision") or {}).get("has_fp16_out_broken_fn") for r in _runs(h))
                and not any((r.get("decision") or {}).get("has_fp16_out_broken_fn") for r in _runs(b)), ""))
    return out


def _refused(r, dt):
    raw = (r.get("raw") or {}).get(f"rowwise_{dt}") or {}
    return raw.get("ok") is False, raw.get("error")


def base_shows_defect(base):
    hits, notes = [], []
    for r in _runs(base):
        d = r.get("decision") or {}
        for dt in ("fp16", "fp32"):
            routed = (d.get(dt) or {}).get("to_scaled_mm")
            refused, err = _refused(r, dt)
            notes.append(f"torch {r.get('torch')}: backend {d.get('backend')}, {dt} to _scaled_mm {routed}, raw {dt} "
                         f"{'refused: ' + _one(err, 160) if refused else 'ok'}")
            if _ver(r.get("torch")) >= (2, 14) and routed and refused and HIPBLASLT_BF16_ONLY in str(err):
                hits.append(f"{r.get('torch')} {dt}")
    return bool(hits), ("defect at " + ", ".join(hits) if hits else "not reachable here") + "; " + "; ".join(notes)


def head_is_fixed(head, base = None):
    bad = []
    for r in _runs(head):
        d = r.get("decision") or {}
        for dt in ("fp16", "fp32"):
            refused, err = _refused(r, dt)
            if (d.get(dt) or {}).get("to_scaled_mm") and refused:
                bad.append(f"{r.get('torch')} {dt} routed to _scaled_mm, which refused: {_one(err, 160)}")
    return not bad, "; ".join(bad) or "no fp16 / fp32 activation routed to a refused _scaled_mm out_dtype"


def table(obs):
    lines = ["| state | torch | hip | backend | FP16_OUT_BROKEN | fp16 -> _scaled_mm | fp32 -> _scaled_mm | raw bf16 | "
             "raw fp16 | raw fp32 | unsloth import | CPU test |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for st in ("base", "head"):
        o = obs.get(st) or {}
        for r in _runs(o):
            d, raw, u = r.get("decision") or {}, r.get("raw") or {}, r.get("unsloth") or {}
            cell = lambda k: "ok" if (raw.get(k) or {}).get("ok") else _one((raw.get(k) or {}).get("error"), 110)
            ud = (u.get("decision") or {})
            uc = ("n/a" if not u else (_one(u.get("error"), 80) if u.get("error") else
                  f"backend {ud.get('backend')}, fp16 {(ud.get('fp16') or {}).get('to_scaled_mm')}, from checkout "
                  f"{u.get('from_checkout')}"))
            ct = o.get("cpu_test") or {}
            lines.append(f"| {st} | {r.get('torch')} | {r.get('hip')} | {d.get('backend')} | "
                         f"{d.get('_SCALED_MM_FP16_OUT_BROKEN')} | {(d.get('fp16') or {}).get('to_scaled_mm')} | "
                         f"{(d.get('fp32') or {}).get('to_scaled_mm')} | {cell('rowwise_bf16')} | {cell('rowwise_fp16')} "
                         f"| {cell('rowwise_fp32')} | {uc} | {_one(ct.get('summary') or ct.get('error'), 80)} |")
    return "\n".join(lines)
