#!/usr/bin/env python3
"""Criteria: Studio's Windows ROCm wide-head SDPA guard under torch.compile (unsloth #17976 follow-up; judges
probes/rk_wide_head_compile_probe.py; base = origin/main, head = base + the fix diff).

base_shows_defect: with Studio's guard installed first and unsloth's guard keeping it, a head-512 attention cannot be
compiled fullgraph (Dynamo backend "eager", no codegen involved) and torch._dynamo.explain reports a graph break.
head_is_fixed: fullgraph compiles on "eager" and "aot_eager" (and "inductor" when the inductor control compiles on
this box), explain reports 0 breaks, every compiled output matches the guarded eager output, no fused SDPA kernel ran
in a compiled call, and the process-wide SDPA backend flags are restored afterwards.
"""

from __future__ import annotations

TITLE = "gfx1151 Windows: Studio wide-head SDPA guard under torch.compile (fullgraph, graph breaks, parity)"
MODE = "differential"
NEEDS = ["gpu", "rocm", "windows", "windows_rocm_wddm", "exclusive_gpu", "nvidia", "discrete_gpu", "linux"]
MATCH = 0.01  # rel L2 compiled vs guarded eager; both are math-backend bf16, measured ~1e-3 or below
REF = 0.02  # rel L2 vs a float32 math reference; a broken fused kernel at head 512 is far above this


def _le(x, lim):
    return x is not None and x <= lim


def _one(x, n):
    return str(x or "").replace("\n", " ").replace("|", "/")[:n]


def _fg(o, b):
    return ((o or {}).get("fullgraph") or {}).get(b) or {}


def gates(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out = []
    for name, o in (("base", b), ("head", h)):
        out.append((f"{name}: probe ran", not o.get("error") and o.get("rc") == 0,
                    f"rc {o.get('rc')}, error {o.get('error')}"))
        out.append((f"{name}: Windows ROCm gfx1151, tensors on the GPU",
                    o.get("platform") == "win32" and bool(o.get("hip")) and str(o.get("arch", "")).startswith("gfx1151")
                    and o.get("device_type") == "cuda" and not o.get("faked_windows_rocm"),
                    f"{o.get('platform')} hip {o.get('hip')} {o.get('arch')} {o.get('device_type')}"))
        out.append((f"{name}: Studio guard installed from this checkout, unsloth's kept it",
                    bool(o.get("studio_installed")) and bool(o.get("studio_wrapped"))
                    and bool(o.get("studio_guard_from_checkout")) and o.get("unsloth_kept_studio") is True,
                    f"installed {o.get('studio_installed')}, from checkout {o.get('studio_guard_from_checkout')}, "
                    f"unsloth kept it {o.get('unsloth_kept_studio')} ({(o.get('unsloth_guard') or {}).get('fix_returned')})"))
        e = o.get("eager") or {}
        out.append((f"{name}: guard engaged eagerly (math only, correct)",
                    e.get("ops") == ["_scaled_dot_product_attention_math"] and e.get("finite") is True
                    and _le(e.get("rel_l2_vs_fp32_math"), REF),
                    f"ops {e.get('ops')}, rel L2 {e.get('rel_l2_vs_fp32_math')}"))
    out.append(("same torch build in both states", b.get("torch") == h.get("torch") and bool(b.get("torch")),
                f"{b.get('torch')} / {h.get('torch')}"))
    out.append(("head carries the fix, base does not",
                bool(h.get("studio_has_allow_in_graph_fn")) and not b.get("studio_has_allow_in_graph_fn"),
                f"head {h.get('studio_has_allow_in_graph_fn')}, base {b.get('studio_has_allow_in_graph_fn')}"))
    return out


def base_shows_defect(base):
    fg = _fg(base, "eager")
    ex = base.get("explain") or {}
    # explain's graph_break_count can read 0 while it lists a break reason (torch 2.12, measured): count both.
    breaks = max(ex.get("graph_break_count") or 0, len(ex.get("break_reasons") or []))
    shown = fg.get("ok") is False and breaks >= 1
    return shown, (f"fullgraph (Dynamo eager) ok {fg.get('ok')}: {_one(fg.get('error'), 300)}; "
                   f"explain breaks {breaks}, reasons {_one(ex.get('break_reasons'), 300)}")


def head_is_fixed(head, base = None):
    want = ["eager", "aot_eager"] + (["inductor"] if (head.get("inductor_control") or {}).get("ok") else [])
    bad = []
    for b in want:
        r = _fg(head, b)
        if not (r.get("ok") and r.get("finite") and _le(r.get("rel_l2_vs_eager"), MATCH)
                and _le(r.get("rel_l2_vs_fp32_math"), REF) and not r.get("fused_ops")
                and r.get("backends_restored") and r.get("held_after") == [0, None]):
            bad.append(f"{b}: {r}")
    ex = head.get("explain") or {}
    if ex.get("graph_break_count") != 0 or ex.get("break_reasons"):
        bad.append(f"explain breaks {ex.get('graph_break_count')} {ex.get('break_reasons')}")
    dc = head.get("default_compile") or {}
    if not (dc.get("ok") and _le(dc.get("rel_l2_vs_eager"), MATCH)):
        bad.append(f"default compile {dc}")
    if not head.get("backends_restored_end") or head.get("held_end") != [0, None]:
        bad.append(f"flags restored {head.get('backends_restored_end')}, held {head.get('held_end')}")
    return not bad, (f"judged backends {want}; " + ("; ".join(x[:400] for x in bad) if bad else
                     "all compiled fullgraph, 0 breaks, outputs match eager, no fused SDPA op, flags restored"))


def table(obs):
    lines = ["| state | backend | fullgraph | rel L2 vs eager | rel L2 vs fp32 | fused ops | flags restored | error |",
             "|---|---|---|---|---|---|---|---|"]
    for st in ("base", "head"):
        o = obs.get(st) or {}
        for b in ("eager", "aot_eager", "inductor"):
            r = _fg(o, b)
            lines.append(f"| {st} | {b} | {r.get('ok')} | {r.get('rel_l2_vs_eager')} | {r.get('rel_l2_vs_fp32_math')} "
                         f"| {r.get('fused_ops')} | {r.get('backends_restored')} | {_one(r.get('error'), 160)} |")
        ex = o.get("explain") or {}
        dc = o.get("default_compile") or {}
        lines.append(f"| {st} | explain / default / inductor control | graphs {ex.get('graph_count')}, breaks "
                     f"{ex.get('graph_break_count')} | default ok {dc.get('ok')} rel {dc.get('rel_l2_vs_eager')} | | "
                     f"| control {(o.get('inductor_control') or {}).get('ok')} | "
                     f"{_one(ex.get('break_reasons'), 200)} |")
    return "\n".join(lines)
