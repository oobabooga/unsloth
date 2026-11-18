#!/usr/bin/env python3
"""Criteria for probes/f3win_probe.py: FLUX.3 Klein explicit int8 vs bf16 through Studio (eager on Windows).

Regression mode: the head may not make the explicit int8 request slower on this OS (H vs the I arms' spread) and must
render it bit-identically (H1 vs I1); HA vs IA is the shared act quant's effect on the forced W8A8 path (reported,
bit-identity expected). Gates: both pytest runs collected and every timed arm rendered."""

from __future__ import annotations

import re
import statistics

TITLE = "gfx1151: FLUX.3 Klein explicit int8 vs bf16, s/step and profile split (Studio, exclusive GPU)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "multi_gpu"]
STEPS_DEFAULT = 16
CATS = (("w8a8 gemm", r"w8a8|_mm_kernel|native_rdna_w8a8_mm"), ("rotate+quant", r"rotq|quant"),
        ("bf16 gemm", r"Cijk|gemm|Gemm|GEMM|hipblaslt|rocblas|matmul|addmm|mm_"),
        ("attention", r"attn|attention|flash|fmha|softmax|sdpa|_fwd"),
        ("dequant / cast", r"copy|cast|to_copy|convert|mul_|elementwise|vectorized|unrolled"))


def _arms(head: dict) -> dict:
    return head.get("arms") or {}


def _steps(rec: dict) -> int:
    p = rec.get("params") or {}
    return int(p.get("steps") or p.get("num_inference_steps") or STEPS_DEFAULT)


def _per_step(rec: dict) -> list:
    out = []
    for r in rec.get("renders") or []:
        d = r.get("denoise_s")
        if d:
            out.append(d / _steps(rec))
    return out


def _side(head: dict, side: str) -> list:
    vals = []
    for rec in _arms(head).values():
        if rec.get("side") == side and rec.get("verdict") == "ok":
            ps = _per_step(rec)
            if ps:
                vals.append(statistics.median(ps))
    return vals


def _walls(head: dict, side: str) -> list:
    vals = []
    for rec in _arms(head).values():
        if rec.get("side") == side and rec.get("verdict") == "ok":
            w = [r["wall_s"] for r in rec.get("renders") or [] if r.get("wall_s")]
            if w:
                vals.append(statistics.median(w))
    return vals


def _pair(head: dict, a: str, b: str) -> dict:
    for p in ((head.get("scores") or {}).get("pairs") or []):
        if p.get("a") == f"klein_{a}" and p.get("b") == f"klein_{b}":
            return p
    return {}


def _identical(p: dict) -> bool:
    return bool(p.get("rows")) and all(r.get("identical") for r in p["rows"])


def gates(obs: dict) -> list:
    head, base = obs.get("head") or {}, obs.get("base") or {}
    timed = [t for t, r in _arms(head).items() if r.get("side") != "W"]
    bad = [t for t in timed if _arms(head)[t].get("verdict") != "ok" or not _per_step(_arms(head)[t])]
    return [("base pytest collected", bool((base.get("pytest") or {}).get("cases")),
             str((base.get("pytest") or {}).get("rc"))),
            ("head pytest collected", bool((head.get("pytest") or {}).get("cases")),
             str((head.get("pytest") or {}).get("rc"))),
            ("timed arms rendered", bool(timed) and not bad, f"{len(timed)} arms, failed {bad}"),
            ("H1 vs I1 scored", bool(_pair(head, "H1", "I1").get("rows")), str(head.get("score_rc")))]


def head_is_worse(base: dict, head: dict) -> tuple:
    hp = head.get("pytest") or {}
    failed = sorted(k for k, v in (hp.get("cases") or {}).items() if v == "failed")
    if failed:
        return True, f"head tests failed: {failed[:5]}"
    i, h = _side(head, "I"), _side(head, "H")
    if not i or not h:
        return False, "no I / H timing"
    spread = (max(i) - min(i)) if len(i) > 1 else 0.0
    if h[0] > max(i) + max(spread, 0.02 * statistics.median(i)):
        return True, f"head explicit int8 {h[0]:.3f} s/step vs base {statistics.median(i):.3f} (spread {spread:.3f})"
    if not _identical(_pair(head, "H1", "I1")):
        return True, "head explicit int8 render differs from base"
    return False, f"H {h[0]:.3f} vs I {statistics.median(i):.3f} s/step, renders identical"


def _split(prof: dict) -> dict:
    out = {c: 0.0 for c, _ in CATS}
    out["other"] = 0.0
    for k in prof.get("kernels") or []:
        for c, rx in CATS:
            if re.search(rx, k["name"]):
                out[c] += k["ms"]
                break
        else:
            out["other"] += k["ms"]
    return {k: round(v, 1) for k, v in out.items()}


def table(obs: dict) -> str:
    head = obs.get("head") or {}
    L = [f"torch {head.get('torch')} hip {head.get('hip')} triton {head.get('triton')} arch {head.get('arch')} "
         f"platform {head.get('platform')}", "",
         "| arm | s/step per round (median of renders) | median s/step | s/image | vs I | LPIPS vs B1 |",
         "|---|---|---|---|---|---|"]
    vs = {p["a"]: p for p in ((head.get("scores") or {}).get("vs_ref") or [])}
    i_med = statistics.median(_side(head, "I")) if _side(head, "I") else None
    for side in ("B", "I", "IA", "HA", "H"):
        v = _side(head, side)
        if not v:
            continue
        med = statistics.median(v)
        w = _walls(head, side)
        lp = (vs.get(f"klein_{side}1") or {}).get("lpips_mean")
        rel = f"{(med / i_med - 1) * 100:+.1f}%" if i_med else ""
        L.append(f"| {side} | {', '.join(f'{x:.3f}' for x in v)} | {med:.3f} | "
                 f"{statistics.median(w):.2f} | {rel} | {lp} |" if w else f"| {side} | {v} | {med:.3f} | | {rel} | {lp} |")
    L += ["", "| pair | identical | LPIPS mean | PSNR |", "|---|---|---|---|"]
    for a, b in (("HA1", "IA1"), ("H1", "I1"), ("I2", "I1"), ("IA2", "IA1")):
        p = _pair(head, a, b)
        L.append(f"| {a} vs {b} | {_identical(p)} | {p.get('lpips_mean')} | {p.get('psnr_mean')} |")
    L += ["", "| arm (round 1) | profiled call wall ms | device ms | split (ms) |", "|---|---|---|---|"]
    for side in ("B", "I", "IA", "HA", "H"):
        rec = _arms(head).get(f"{side}1") or {}
        prof = rec.get("prof") or {}
        if prof:
            L.append(f"| {side} | {prof.get('wall_ms')} | {prof.get('device_ms')} | {_split(prof)} |")
    for side in ("I", "HA"):
        prof = (_arms(head).get(f"{side}1") or {}).get("prof") or {}
        if prof.get("kernels"):
            L += ["", f"Top kernels {side}1:", ""] + [f"- {k['ms']} ms x{k['count']} {k['name'][:120]}"
                                                     for k in prof["kernels"][:15]]
    return "\n".join(L)
