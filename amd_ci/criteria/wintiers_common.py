#!/usr/bin/env python3
"""Shared readers for criteria/wintiers_win.py and criteria/wintiers_lnx.py (probes/wintiers_probe.py)."""

from __future__ import annotations

import statistics

STEPS = {"klein": 16, "q21": 10, "wan": 8}


def arms(head: dict) -> dict:
    return head.get("arms") or {}


def fam_arms(head: dict, fam: str) -> dict:
    return {k.split("_", 1)[1]: v for k, v in arms(head).items() if v.get("family") == fam}


def families(head: dict) -> list:
    out = []
    for v in arms(head).values():
        if v.get("family") not in out:
            out.append(v.get("family"))
    return out


def steps(rec: dict, fam: str) -> int:
    p = rec.get("params") or {}
    return int(p.get("steps") or p.get("num_inference_steps") or STEPS.get(fam, 1))


def per_step(rec: dict, fam: str) -> list:
    return [r["denoise_s"] / steps(rec, fam) for r in rec.get("renders") or [] if r.get("denoise_s")]


def med(xs: list):
    return statistics.median(xs) if xs else None


def tier(rec: dict) -> dict:
    lines = rec.get("engage_lines") or []
    end = rec.get("status_end") or {}
    optims = end.get("speed_optims") or []
    compiled = ("compiled" in optims) if isinstance(optims, list) and optims else any(
        "deferred profile engaged" in x or ("optims=" in x and "compiled" in x) for x in lines)
    w8a8 = any("W8A8" in x and ("linears" in x or "RDNA Triton" in x) for x in lines)
    tq = end.get("transformer_quant") or (rec.get("status") or {}).get("transformer_quant")
    probe = next((x.split("regional compile probe", 1)[1][:90] for x in lines if "regional compile probe" in x), None)
    return {"quant": tq or "-", "compiled": bool(compiled), "w8a8": w8a8, "speed": end.get("speed_mode"),
            "probe": probe}


def tier_str(t: dict) -> str:
    return f"{t['quant']} / {'compiled' if t['compiled'] else 'eager'} / {'W8A8' if t['w8a8'] else 'weight-only'}" \
        if t["quant"] not in ("-", None, "off") else f"bf16 / {'compiled' if t['compiled'] else 'eager'}"


def pair(head: dict, fam: str, a: str, b: str) -> dict:
    sc = (head.get("scores") or {}).get(fam) or {}
    for p in (sc.get("pairs") or []) + (sc.get("vs_ref") or []):
        if p.get("a") == f"{fam}_{a}" and p.get("b") == f"{fam}_{b}":
            return p
    return {}


def side_meds(head: dict, fam: str, side: str) -> list:
    out = []
    for tag, rec in fam_arms(head, fam).items():
        if rec.get("side") == side and rec.get("verdict") == "ok":
            m = med(per_step(rec, fam))
            if m:
                out.append(m)
    return out


def timing(head: dict, fam: str) -> dict:
    b, h = side_meds(head, fam, "B"), side_meds(head, fam, "H")
    if not b or not h:
        return {}
    spread = max((max(b) - min(b)) if len(b) > 1 else 0.0, (max(h) - min(h)) if len(h) > 1 else 0.0)
    return {"base": med(b), "head": med(h), "spread": spread, "delta": med(h) - med(b),
            "rel": (med(h) - med(b)) / med(b)}


def decisions_table(obs: dict) -> str:
    rows = ["| state | platform | compile runtime | RDNA W8A8 ready | compile probe | klein explicit int8 | "
            "klein explicit W8A8 | qwen-2.1 auto W8A8 | qwen-image auto W8A8 |", "|---|---|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        d = (obs.get(name) or {}).get("decisions") or {}
        r = d.get("reasons_first") or {}

        def yn(fam, key):
            v = (r.get(fam) or {}).get(key)
            return "yes" if v else "no"

        cp = d.get("compile_probe") or {}
        rows.append(f"| {name} | {d.get('platform')} | {d.get('torch_compile_runtime_available')} | "
                    f"{d.get('rocm_rdna_w8a8_ready')} | {cp.get('ok')} ({str(cp.get('why'))[:50]}) | "
                    f"{yn('flux.3-klein', 'explicit_int8')} | {yn('flux.3-klein', 'explicit_w8a8')} | "
                    f"{yn('qwen-image-2.1', 'auto_w8a8')} | {yn('qwen-image', 'auto_w8a8')} |")
    return "\n".join(rows)


def arms_table(head: dict) -> str:
    rows = ["| family | arm | tree | tier engaged | s/step (median of timed) | renders | load s | LPIPS vs B1 | "
            "PSNR vs B1 |", "|---|---|---|---|---|---|---|---|---|"]
    for fam in families(head):
        for tag, rec in fam_arms(head, fam).items():
            ps = per_step(rec, fam)
            p = pair(head, fam, tag, "B1") if tag != "B1" else {}
            rows.append(f"| {fam} | {tag} | {rec.get('tree')} | {tier_str(tier(rec))} | "
                        f"{round(med(ps), 3) if ps else rec.get('verdict')} | {len(ps)} | {rec.get('load_s')} | "
                        f"{p.get('lpips_mean', '-') if p else '-'} | {p.get('psnr_mean', '-') if p else '-'} |")
    rows += ["", "| family | base s/step | head s/step | delta | A/A spread | H1 vs B1 LPIPS | B2 vs B1 LPIPS | "
             "H2 vs H1 LPIPS |", "|---|---|---|---|---|---|---|---|"]
    for fam in families(head):
        t = timing(head, fam)
        if not t:
            continue
        rows.append(f"| {fam} | {t['base']:.3f} | {t['head']:.3f} | {t['delta']:+.3f} ({t['rel'] * 100:+.1f}%) | "
                    f"{t['spread']:.3f} | {pair(head, fam, 'H1', 'B1').get('lpips_mean')} | "
                    f"{pair(head, fam, 'B2', 'B1').get('lpips_mean')} | {pair(head, fam, 'H2', 'H1').get('lpips_mean')} |")
    return "\n".join(rows)


def common_gates(obs: dict) -> list:
    head, base = obs.get("head") or {}, obs.get("base") or {}
    out = []
    for name, o in (("base", base), ("head", head)):
        out.append((f"{name} pytest collected", bool((o.get("pytest") or {}).get("cases")),
                    f"rc {(o.get('pytest') or {}).get('rc')}, passed {(o.get('pytest') or {}).get('passed')}, "
                    f"failed {(o.get('pytest') or {}).get('failed')}"))
        d = o.get("decisions") or {}
        out.append((f"{name} decisions ran on the GPU", d.get("verdict") == "ok" and bool(d.get("gpu")),
                    str(d.get("gpu") or d.get("error"))[:120]))
    timed = {t: r for t, r in arms(head).items()}
    bad = [t for t, r in timed.items() if r.get("verdict") != "ok" or not per_step(r, r.get("family"))]
    out.append(("every arm rendered", bool(timed) and not bad, f"{len(timed)} arms, failed {bad}"))
    return out
