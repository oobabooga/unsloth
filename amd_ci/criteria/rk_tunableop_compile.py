#!/usr/bin/env python3
"""Criteria: Studio's ROCm TunableOp tune hooks under a whole-module compile (unsloth #17299 follow-up; judges
probes/rk_tunableop_compile_probe.py; base = origin/main, head = base + the fix diff).

base_shows_defect: with UNSLOTH_DIFFUSION_ROCM_TUNABLEOP=tune and the hooks installed on a real gfx1151, the module
cannot be compiled fullgraph (Dynamo backend "eager", no codegen involved), and a fullgraph=False compile drops the
module frame (Dynamo "unimplemented" on the hook's lock) or graph-breaks.
head_is_fixed: fullgraph compiles on "eager" and "aot_eager" (and "inductor" when it compiled the base's forward-only
control, i.e. when inductor works on this box), outputs match the uncompiled module, the fullgraph=False compile has
no unimplemented frame and no break, AND TunableOp tuning still engages on eager calls: a steady eager call switches
tuning on and back off, the hook depth returns to 0, and TunableOp holds more results after it than before (the CSV
written by the post hook has rows).
"""

from __future__ import annotations

TITLE = "gfx1151 Linux: ROCm TunableOp tune hooks under a whole-module torch.compile"
MODE = "differential"
NEEDS = ["gpu", "rocm", "linux", "exclusive_gpu", "multi_gpu", "windows", "discrete_gpu"]
MATCH = 0.02  # rel L2 compiled vs eager bf16 GEMMs; tuned and default GEMM solutions differ in the last bits only


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
        out.append((f"{name}: Linux ROCm gfx1151, GPU tensors",
                    o.get("platform") == "linux" and bool(o.get("hip")) and str(o.get("arch", "")).startswith("gfx1151")
                    and o.get("device_type") == "cuda" and not o.get("faked_rocm"),
                    f"{o.get('platform')} hip {o.get('hip')} {o.get('arch')} {o.get('device_type')}"))
        st = o.get("state_after_install") or {}
        out.append((f"{name}: hooks installed in tune mode from this checkout",
                    bool(o.get("installed")) and o.get("hooks") == [1, 1] and st.get("tune") is True
                    and st.get("active") is True and bool(o.get("module_from_checkout")),
                    f"eligible {o.get('eligible')}, installed {o.get('installed')}, hooks {o.get('hooks')}, "
                    f"tune {st.get('tune')}, file {st.get('file')}"))
    out.append(("same torch build in both states", b.get("torch") == h.get("torch") and bool(b.get("torch")),
                f"{b.get('torch')} / {h.get('torch')}"))
    out.append(("head carries the fix, base does not",
                bool(h.get("head_has_compile_guard")) and not b.get("head_has_compile_guard"),
                f"head {h.get('head_has_compile_guard')}, base {b.get('head_has_compile_guard')}"))
    return out


def base_shows_defect(base):
    fg = _fg(base, "eager")
    ex = base.get("explain") or {}
    dropped = (ex.get("unimplemented_count") or 0) + (ex.get("graph_break_count") or 0)
    return (fg.get("ok") is False and dropped >= 1,
            f"fullgraph (Dynamo eager) ok {fg.get('ok')}: {_one(fg.get('error'), 300)}; fullgraph=False: "
            f"unimplemented {ex.get('unimplemented_count')}, breaks {ex.get('graph_break_count')}, frames ok "
            f"{ex.get('frames_ok')}/{ex.get('frames_total')}, {_one(ex.get('unimplemented'), 200)}")


def head_is_fixed(head, base = None):
    want = ["eager", "aot_eager"]
    ind = _fg(head, "inductor")
    codegen_failed = any(t in str(ind.get("error") or "") for t in ("InductorError", "BackendCompilerFailed"))
    if ind.get("ok") or not codegen_failed:
        want.append("inductor")  # judged unless the failure is inductor's own codegen on this box
    bad = []
    for b in want:
        r = _fg(head, b)
        if not (r.get("ok") and r.get("finite") and _le(r.get("rel_l2_vs_eager"), MATCH)
                and r.get("depth_after") == 0 and r.get("tuning_on_after") is False):
            bad.append(f"{b}: {r}")
    ex = head.get("explain") or {}
    if (ex.get("unimplemented_count") or 0) or (ex.get("graph_break_count") or 0):
        bad.append(f"fullgraph=False dropped frames / broke: {ex}")
    dc = head.get("default_compile") or {}
    if not (dc.get("ok") and _le(dc.get("rel_l2_vs_eager"), MATCH)):
        bad.append(f"default compile {dc}")
    t = head.get("eager_tune") or {}
    grew = (t.get("results_after_steady") or 0) > (t.get("results_before") or 0)
    if not (t.get("tuning_enable_calls") == [True, False] and t.get("depth") == 0 and not t.get("tuning_on_after")
            and grew and t.get("file_exists") and (t.get("file_rows") or 0) > 0):
        bad.append(f"eager tuning: {t}")
    return not bad, (f"judged backends {want}; " + ("; ".join(_one(x, 400) for x in bad) if bad else
                     f"all compiled fullgraph and match eager, no dropped frame; eager tuning engaged "
                     f"(results {t.get('results_before')} -> {t.get('results_after_steady')}, CSV rows "
                     f"{t.get('file_rows')})"))


def table(obs):
    lines = ["| state | backend | fullgraph | rel L2 vs eager | depth after | tuning on after | error |",
             "|---|---|---|---|---|---|---|"]
    for st in ("base", "head"):
        o = obs.get(st) or {}
        for b in ("eager", "aot_eager", "inductor"):
            r = _fg(o, b)
            lines.append(f"| {st} | {b} | {r.get('ok')} | {r.get('rel_l2_vs_eager')} | {r.get('depth_after')} | "
                         f"{r.get('tuning_on_after')} | {_one(r.get('error'), 160)} |")
        ex, dc, t = o.get("explain") or {}, o.get("default_compile") or {}, o.get("eager_tune") or {}
        lines.append(f"| {st} | fullgraph=False aot_eager | ok {dc.get('ok')} rel {dc.get('rel_l2_vs_eager')} | "
                     f"unimplemented {ex.get('unimplemented_count')}, breaks {ex.get('graph_break_count')}, frames "
                     f"{ex.get('frames_ok')}/{ex.get('frames_total')} | | | {_one(ex.get('unimplemented'), 160)} |")
        lines.append(f"| {st} | eager tuning (fresh shape x2) | enable calls {t.get('tuning_enable_calls')} | results "
                     f"{t.get('results_before')} -> {t.get('results_after_first')} -> {t.get('results_after_steady')} | "
                     f"{t.get('depth')} | {t.get('tuning_on_after')} | CSV rows {t.get('file_rows')} |")
    return "\n".join(lines)
