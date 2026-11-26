#!/usr/bin/env python3
"""Criteria: do concurrent image requests from different managed accounts batch on this AMD GPU, keep every
member's pixels within LPIPS 1e-3 of its solo render, and serve no fewer images per minute than the base?
Judges probes/imgconc_probe.py.

Timing (head state's rows): ABBA arms, one fresh process each, N clients each with its own account. Ratio = median
images/min over the head arms / median over the base arms; noise = max(A/A spread of each arm, 2%). Worse = ratio
below 1 - noise, or a batched member past LPIPS 1e-3 of its solo render. A cell with no batch larger than 1 at the
head, or any batch at the base, is void (gate)."""

from __future__ import annotations

import statistics as st

TITLE = "gfx1151: Studio concurrent image clients, cross-account batching, head vs base (ABBA, exclusive GPU)"
MODE = "regression"
NEEDS = ["gpu", "rocm", "exclusive_gpu", "nvidia", "discrete_gpu", "windows", "multi_gpu"]
LPIPS_MAX = 1e-3


def _rows(obs: dict, arm: str) -> list:
    head = obs.get("head") or obs
    return [r for r in head.get("rows") or [] if r.get("arm") == arm and not r.get("error") and r.get("rc") == 0]


def _spread(values: list) -> float:
    return (max(values) - min(values)) / st.median(values) if len(values) > 1 else 0.0


def summary(obs: dict) -> dict:
    base, head = _rows(obs, "base"), _rows(obs, "head")
    b = [r["images_per_min"] for r in base if r.get("images_per_min")]
    h = [r["images_per_min"] for r in head if r.get("images_per_min")]
    out = {"base_ipm": b, "head_ipm": h}
    if b and h:
        out["ratio"] = st.median(h) / st.median(b)
        out["noise"] = max(_spread(b), _spread(h), 0.02)
    lp = [c.get("lpips") for r in head for c in r.get("solo_checks") or [] if c.get("batched_in", 1) > 1]
    out["batched_checks"] = len(lp)
    out["lpips_max"] = max([x for x in lp if x is not None], default = None)
    out["lpips_known"] = bool(lp) and all(x is not None for x in lp)
    out["bitwise_all"] = all(c.get("bitwise") for r in head for c in r.get("solo_checks") or []
                             if c.get("batched_in", 1) > 1)
    return out


def gates(obs: dict) -> list:
    head_obs = obs.get("head") or {}
    base, head = _rows(obs, "base"), _rows(obs, "head")
    rows = head_obs.get("rows") or []
    s = summary(obs)
    return [
        ("full ABBA ran after the smoke", not head_obs.get("staged_stop"), str(head_obs.get("staged_stop") or "")),
        ("gfx1151", str(head_obs.get("arch") or "").startswith("gfx1151"), str(head_obs.get("arch"))),
        ("every arm finished", bool(rows) and all(r.get("rc") == 0 and not r.get("error") for r in rows),
         "; ".join(f"{r.get('label')}: rc {r.get('rc')} {str(r.get('error') or '')[:200]}" for r in rows)),
        ("base and head arms", bool(base) and bool(head), f"{len(base)} base, {len(head)} head"),
        ("each arm imported its own tree",
         all("/base/" in str(r.get("imported_from")) for r in base)
         and all("/base/" not in str(r.get("imported_from")) for r in head),
         "; ".join(str(r.get("imported_from")) for r in rows)),
        ("head load batches any size", all((r.get("request_batching") or {}).get("mode") == "any" for r in head),
         str([r.get("request_batching") for r in head])),
        ("a batch larger than 1 at the head", bool(head) and all((r.get("max_batch") or 0) > 1 for r in head),
         str([r.get("call_sizes") for r in head])),
        ("batches mixed accounts at the head", any(max(r.get("accounts_per_batch") or [0]) > 1 for r in head),
         str([r.get("accounts_per_batch") for r in head])),
        ("no batch at the base", bool(base) and all((r.get("max_batch") or 0) == 1 for r in base),
         str([r.get("call_sizes") for r in base])),
        ("every member got its own prompt", all(r.get("own_prompt_ok") for r in base + head), ""),
        ("batched members checked against solo", s["batched_checks"] > 0 and s["lpips_known"],
         f"{s['batched_checks']} checks, lpips known {s['lpips_known']}"),
    ]


def head_is_worse(base: dict, head: dict) -> tuple:
    s = summary({"head": head})
    bad = []
    if "ratio" in s and s["ratio"] < 1 - s["noise"]:
        bad.append(f"images/min ratio {s['ratio']:.3f} < 1 - {s['noise']:.3f}")
    if s["lpips_max"] is not None and s["lpips_max"] > LPIPS_MAX and not s["bitwise_all"]:
        bad.append(f"a batched member is LPIPS {s['lpips_max']:.5f} from its solo render (> {LPIPS_MAX})")
    detail = (f"images/min ratio {s.get('ratio', float('nan')):.3f} (noise {s.get('noise', float('nan')):.3f}), "
              f"max LPIPS vs solo {s['lpips_max']}")
    return (bool(bad), "; ".join(bad) if bad else detail)


def table(obs: dict) -> str:
    head = obs.get("head") or {}
    lines = ["| arm | images/min | p50 s | p95 s | call sizes | log batches | accounts per batch | peak GiB | "
             "max LPIPS vs solo | bitwise |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in head.get("rows") or []:
        checks = [c for c in r.get("solo_checks") or [] if c.get("batched_in", 1) > 1] or (r.get("solo_checks") or [])
        lp = [c.get("lpips") for c in checks if c.get("lpips") is not None]
        lines.append(
            f"| {r.get('label')} | {r.get('images_per_min')} | {r.get('latency_p50_s')} | {r.get('latency_p95_s')} | "
            f"{r.get('call_sizes')} | {r.get('batch_log_lines')} | {r.get('accounts_per_batch')} | "
            f"{r.get('peak_reserved_gib')} | {max(lp) if lp else None} | "
            f"{all(c.get('bitwise') for c in checks) if checks else None} |")
    for r in head.get("smoke_rows") or []:
        lines.append(f"| {r.get('label')} (512 px, 1 request each) | {r.get('images_per_min')} | {r.get('latency_p50_s')} | "
                     f"{r.get('latency_p95_s')} | {r.get('call_sizes')} | {r.get('batch_log_lines')} | "
                     f"{r.get('accounts_per_batch')} | {r.get('peak_reserved_gib')} | | |")
    lines.append("")
    lines.append(f"Full arms at {head.get('size')} px, {head.get('clients')} clients x {head.get('requests')} requests; "
                 f"request_batching {[ (r.get('label'), (r.get('request_batching') or {}).get('mode')) for r in head.get('rows') or []]}.")
    s = summary(obs)
    if "ratio" in s:
        lines.append("")
        lines.append(f"Head / base images per minute: {s['ratio']:.3f} (A/A noise {s['noise']:.3f}).")
    return "\n".join(lines)
