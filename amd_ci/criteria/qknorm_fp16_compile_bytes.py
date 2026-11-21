#!/usr/bin/env python3
"""Criteria: fast_qk_norm_rope eager vs torch.compile (Inductor) bytes on gfx1151.

  base_shows_defect  some eager vs Inductor case differs (expected: fp16), or the fp16
                     test_compiled_no_graph_breaks_same_bytes ids fail in pytest
  head_is_fixed      every dtype x packed x backend case is byte-equal, and no test in
                     tests/test_qk_norm_rope.py fails at the head

Gates (non-vacuity): both states ran on ROCm (torch.version.hip set), imported their own checkout's
qk_norm_rope, ran all 12 cases without error and with zero graph breaks, and pytest ran and collected.
"""

from __future__ import annotations

TITLE = "fast_qk_norm_rope: eager vs torch.compile bytes per dtype on gfx1151"
MODE = "differential"
# nvidia: checked locally, not here; windows: Linux ROCm only.
NEEDS = ["rocm", "gpu", "nvidia", "windows"]

FP16_IDS = "test_compiled_no_graph_breaks_same_bytes"


def _run(o):
    return (o or {}).get("run") or {}


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        r = _run(o)
        h = r.get("header") or {}
        ck = str(o.get("checkout") or "<none>")
        out.append((f"{name} ran on ROCm", bool(h.get("hip")), str(h)[:300] or f"rc={r.get('rc')} {str(r.get('stderr_tail'))[-300:]}"))
        out.append((f"{name} imported its own qk_norm_rope", str(h.get("qk_file", "")).startswith(ck),
                    f"{h.get('qk_file')} vs {ck}"))
        res = r.get("results") or []
        errs = [x for x in res if x.get("error")]
        out.append((f"{name} ran all 12 cases", len(res) == 12 and not errs,
                    f"{len(res)} cases; errors: {[x.get('error') for x in errs][:2]}"))
        gb = [x for x in res if x.get("graph_breaks")]
        out.append((f"{name} no graph breaks", not gb, str([(x['dtype'], x['packed'], x['backend']) for x in gb])[:200]))
        p = o.get("pytest") or {}
        out.append((f"{name} pytest ran", p.get("rc") in (0, 1),
                    f"rc={p.get('rc')} {str(p.get('stderr_tail') or '')[-200:]}"))
        out.append((f"{name} pytest collected",
                    (p.get("n_passed", 0) + p.get("n_failed", 0) + p.get("n_skipped", 0)) > 0,
                    f"passed {p.get('n_passed', 0)}, failed {p.get('n_failed', 0)}, skipped {p.get('n_skipped', 0)}"))
    return out


def table(obs):
    rows = ["| state | dtype | packed | backend | equal | per output: n_diff / max_abs / max_ulp; err eager vs compiled (fp64 ref) |",
            "|---|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        for r in _run(obs.get(name)).get("results") or []:
            cells = []
            for k, v in (r.get("outputs") or {}).items():
                cells.append(f"{k}: {v['n_diff']}/{v.get('max_abs', 0):.3g}/{v.get('max_ulp', 0)}; "
                             f"{v['err_eager']:.3g} vs {v['err_compiled']:.3g}")
            rows.append(f"| {name} | {r['dtype']} | {r['packed']} | {r['backend']} | {r.get('equal')} | "
                        f"{' ; '.join(cells) or r.get('error', '')} |")
    rows += ["", "| state | exp | dtype | packed | detail |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        for e in _run(obs.get(name)).get("exps") or []:
            detail = {k: v for k, v in e.items() if k not in ("exp", "dtype", "packed")}
            if "outputs" in detail:
                detail["outputs"] = {k: (v["n_diff"], v.get("max_abs", 0), v.get("max_rel", 0)) for k, v in detail["outputs"].items()}
            rows.append(f"| {name} | {e['exp']} | {e['dtype']} | {e['packed']} | {str(detail)[:400]} |")
    rows += ["", "| state | kernel | dir | options | asm counts |", "|---|---|---|---|---|"]
    for name in ("base", "head"):
        for b in _run(obs.get(name)).get("builds") or []:
            opts = {k: b.get(k) for k in ("num_warps", "num_stages", "enable_fp_fusion", "allow_flush_denorm",
                                          "arch", "warp_size", "waves_per_eu", "sanitize_overflow") if k in b}
            rows.append(f"| {name} | {b['name']} | {b['dir'][:10]} | {opts} | {b.get('asm_counts')} |")
    rows += ["", "| state | pytest passed | failed | skipped | failing tests |", "|---|---|---|---|---|"]
    for name in ("base", "head", "merge"):
        p = (obs.get(name) or {}).get("pytest")
        if not p:
            continue
        bad = ", ".join(f"`{t.split('::')[-1]}`" for t in p.get("failed", []) + p.get("errors", [])) or "none"
        rows.append(f"| {name} | {p.get('n_passed', 0)} | {p.get('n_failed', 0)} | {p.get('n_skipped', 0)} | {bad} |")
    h = _run(obs.get("head")).get("header")
    if h:
        rows += ["", f"Head: {h}"]
    return "\n".join(rows)


def base_shows_defect(base):
    res = _run(base).get("results") or []
    p = base.get("pytest") or {}
    bad = [t for t in p.get("failed", []) + p.get("errors", []) if FP16_IDS in t]
    return any(r.get("backend") == "inductor" and r.get("equal") is False for r in res) or bool(bad)


def head_is_fixed(head):
    res = _run(head).get("results") or []
    p = head.get("pytest") or {}
    return (len(res) == 12 and all(r.get("equal") is True for r in res)
            and p.get("rc") in (0, 1) and not (p.get("failed") or p.get("errors"))
            and p.get("n_passed", 0) > 0)
