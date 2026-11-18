#!/usr/bin/env python3
"""Criteria (unslothai/unsloth#17738): hybrid packing on gfx1151 should beat the padded path in useful
tokens per second with the same losses. Differential: base packed <= padded (defect), head packed > padded."""

from __future__ import annotations

TITLE = "gfx1151 hybrid packing: useful tokens/s packed vs padded, base versus head"
MODE = "differential"
NEEDS = ["rocm", "gpu", "integrated_gpu", "linux", "windows", "exclusive_gpu"]
MIN_GAIN = 1.10


def _states(obs):
    return {k: v for k, v in obs.items() if not k.startswith("_")}


def _tps(o, mode):
    return (((o.get("modes") or {}).get(mode) or {}).get("summary") or {}).get("useful_tok_per_s")


def gates(obs):
    out = []
    for name, o in _states(obs).items():
        for mode, rec in (o.get("modes") or {}).items():
            ok = rec.get("rc") == 0 and not rec.get("error") and rec.get("summary") is not None
            out.append((f"{name}/{mode}: trained", ok, str(rec.get("error") or rec.get("summary"))[:200]))
            out.append((f"{name}/{mode}: ROCm torch", bool(rec.get("hip")), str(rec.get("hip"))))
        packed = (o.get("modes") or {}).get("packed") or {}
        if packed:
            out.append((f"{name}/packed: varlen hybrid path patched",
                        (packed.get("hybrid_stats") or {}).get("patched", 0) > 0, str(packed.get("hybrid_stats"))))
        padded = (o.get("modes") or {}).get("padded") or {}
        if padded:
            shapes = {tuple(s["shape"][:1]) for s in padded.get("steps") or []}
            out.append((f"{name}/padded: batched rows (not packed)", shapes and all(s[0] > 1 for s in shapes),
                        str(sorted(shapes))))
        if "bench" in o:
            out.append((f"{name}: attention bench ran", bool(o["bench"].get("cases")), str(o.get("bench_rc"))))
    return out


def table(obs):
    rows = ["| state | mode | s/step (median) | useful tok/s | tokens/step | peak GB | attn fast/fallback | first losses |",
            "|---|---|---|---|---|---|---|---|"]
    for name, o in _states(obs).items():
        for mode, rec in (o.get("modes") or {}).items():
            s = rec.get("summary") or {}
            st = rec.get("hf_packed_stats") or {}
            losses = [round(x["loss"], 4) for x in (rec.get("steps") or [])[:6]]
            rows.append(f"| {name} | {mode} | {s.get('median_s_per_step', '-')} | {s.get('useful_tok_per_s', '-')} | "
                        f"{s.get('mean_tokens_per_step', '-')} | {s.get('peak_gb', '-')} | "
                        f"{st.get('fast', '-')}/{st.get('fallback', '-')} | {losses or rec.get('error', '-')} |")
    for name, o in _states(obs).items():
        for case in (o.get("bench") or {}).get("cases", []):
            rows.append("")
            rows.append(f"bench {name} {case['label']} docs={case['n_docs']} max_doc={case['max_doc']}")
            rows.append("| candidate | ms fwd+bwd | err out | err dq | err dk | err dv | error |")
            rows.append("|---|---|---|---|---|---|---|")
            for r in case["results"]:
                f = lambda k: f"{r[k]:.3g}" if isinstance(r.get(k), float) else "-"
                rows.append(f"| {r['cand']} | {f('ms_median')} | {f('err_out')} | {f('err_dq')} | {f('err_dk')} | "
                            f"{f('err_dv')} | {r.get('error', '')[:120]} |")
    return "\n".join(rows)


def base_shows_defect(base):
    p, d = _tps(base, "packed"), _tps(base, "padded")
    return bool(p and d) and p < MIN_GAIN * d


def head_is_fixed(head):
    p, d = _tps(head, "packed"), _tps(head, "padded")
    return bool(p and d) and p >= MIN_GAIN * d
