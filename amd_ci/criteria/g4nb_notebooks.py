#!/usr/bin/env python3
"""Criteria: AMD Gemma-4 26B-A4B Text + Vision notebooks on gfx1151, before vs after the merged
routed MoE kernel PRs (unsloth 61afc99cd, unsloth-zoo 58d20ca8 incl. #1580 / #1582).

Regression mode. Head is WORSE when any of:
  * a notebook that ran to completion at the base fails at the head
  * a head notebook completes but no generate call took the routed path
    (routed_moe_forward returned non-None 0 times during generate): the kernels never engaged
  * a head generate call raised
Reported, not judged (single run each, no A/A spread, sampled decoding): per-call tok/s and
training step time.

Gates (non-vacuity): base imported PyPI unsloth 2026.9.14 / unsloth_zoo 2026.9.9 with no
moe_routed module; head imported the git builds (unsloth_zoo has moe_routed, census wrapped);
torch is a ROCm build on gfx1151; each notebook passed in at least one arm (both failing
alike = environment, not a comparison).
"""

from __future__ import annotations

TITLE = "AMD Gemma-4 26B-A4B Text + Vision notebooks on gfx1151: PyPI (before) vs merged routed MoE kernels (after)"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia"]
TAGS = ("g4_Text", "g4_Vision")


def _nb(o, tag):
    return ((o or {}).get("notebooks") or {}).get(tag) or {}


def _v(o, tag):
    return _nb(o, tag).get("versions") or {}


def gates(obs):
    out = []
    b, h = obs.get("base") or {}, obs.get("head") or {}
    for tag in TAGS:
        bv, hv = _v(b, tag), _v(h, tag)
        out.append((f"{tag} base imported PyPI unsloth 2026.9.14 / zoo 2026.9.9",
                    bv.get("unsloth") == "2026.9.14" and bv.get("unsloth_zoo") == "2026.9.9",
                    f"unsloth {bv.get('unsloth')}, zoo {bv.get('unsloth_zoo')}"))
        bc = (_nb(b, tag).get("census") or {})
        hc = (_nb(h, tag).get("census") or {})
        out.append((f"{tag} base has no moe_routed; head has it (census wrapped)",
                    bc.get("module_present") is False and hc.get("module_present") is True and hc.get("wrapped") is True,
                    f"base present={bc.get('module_present')}, head present={hc.get('module_present')} wrapped={hc.get('wrapped')}"))
        hi = h.get("head_install") or {}
        out.append((f"{tag} head runs the git builds (install rc 0, moe_routed importable)",
                    hi.get("rc") == 0 and hc.get("module_present") is True,
                    f"install rc={hi.get('rc')}, unsloth {hv.get('unsloth')}, zoo {hv.get('unsloth_zoo')}"))
        for name, v in (("base", bv), ("head", hv)):
            out.append((f"{tag} {name} torch is ROCm on gfx1151", bool(v.get("hip")) and str(v.get("arch") or "").startswith("gfx1151"),
                        f"torch {v.get('torch')} hip {v.get('hip')} arch {v.get('arch')}"))
        out.append((f"{tag} passed in at least one arm", _nb(b, tag).get("passed") or _nb(h, tag).get("passed"),
                    f"base passed={_nb(b, tag).get('passed')} (cell {_nb(b, tag).get('failing_cell')}), "
                    f"head passed={_nb(h, tag).get('passed')} (cell {_nb(h, tag).get('failing_cell')})"))
    return out


def _gen_summary(n):
    return "; ".join(f"c{g.get('cell')}: {g.get('new_tokens')} tok / {g.get('seconds')} s = {g.get('tok_per_s')} tok/s"
                     + (f" routed {g['routed']['returned']}/{g['routed']['calls']}" if g.get("routed") else "")
                     + (f" ERR {g['error']}" if g.get("error") else "")
                     for g in n.get("generates") or []) or "-"


def _steps(n):
    s = n.get("steps") or []
    if not s:
        return "-"
    rest = s[1:] or s
    return f"n={len(s)}, first {s[0]} s, median(2..) {sorted(rest)[len(rest) // 2]} s, mean(2..) {round(sum(rest) / len(rest), 3)} s"


def table(obs):
    rows = ["| notebook | arm | PASS | failing cell | generate calls (tok/s, routed returned/calls) | step time | census generate (calls/returned/declined, nf4/bf16) | decline reasons | census outside generate | losses |",
            "|" + "---|" * 10]
    vers = {}
    for tag in TAGS:
        for arm in ("base", "head"):
            n = _nb(obs.get(arm), tag)
            c = (n.get("census") or {})
            g, o = c.get("generate") or {}, c.get("other") or {}
            vers.setdefault(arm, n.get("versions"))
            rows.append(f"| {tag} | {arm} | {'PASS' if n.get('passed') else 'FAIL rc=' + str(n.get('rc'))} "
                        f"| {'' if n.get('passed') else str(n.get('failing_cell')) + ': `' + str(n.get('failing_cell_src')) + '`'} "
                        f"| {_gen_summary(n)} | {_steps(n)} "
                        f"| {g.get('calls')}/{g.get('returned')}/{g.get('declined')}, {g.get('nf4_routed')}/{g.get('bf16_moe')} "
                        f"| {g.get('decline_reasons')} | {o.get('calls')}/{o.get('returned')}/{o.get('declined')} {o.get('decline_reasons')} "
                        f"| {[round(x, 3) for x in (n.get('train') or {}).get('losses', [])]} |")
    for arm, v in vers.items():
        if v:
            rows.append(f"\n{arm} versions: " + ", ".join(f"{k}={v[k]}" for k in ("torch", "hip", "arch", "device", "rocm_info", "bitsandbytes", "bnb_lib",
                                                                                   "transformers", "trl", "peft", "unsloth", "unsloth_zoo", "triton") if k in v))
    for arm in ("base", "head"):
        for tag in TAGS:
            n = _nb(obs.get(arm), tag)
            if n and not n.get("passed"):
                rows.append(f"\n{arm} {tag} traceback tail:\n```\n{(n.get('traceback') or n.get('log_tail') or '')[-1800:]}\n```")
    return "\n".join(rows)


def head_is_worse(base, head):
    problems, notes = [], []
    for tag in TAGS:
        b, h = _nb(base, tag), _nb(head, tag)
        if b.get("passed") and not h.get("passed"):
            problems.append(f"{tag}: passes at base, fails at head in cell {h.get('failing_cell')} (`{h.get('failing_cell_src')}`)")
        elif not b.get("passed") and not h.get("passed"):
            notes.append(f"{tag}: fails in both arms (base cell {b.get('failing_cell')}, head cell {h.get('failing_cell')})")
        elif not b.get("passed") and h.get("passed"):
            notes.append(f"{tag}: fails at base (cell {b.get('failing_cell')}), passes at head")
        g = ((h.get("census") or {}).get("generate") or {})
        if h.get("generates"):
            if not g.get("returned"):
                problems.append(f"{tag}: routed MoE path never engaged during head generate "
                                f"(calls {g.get('calls')}, declined {g.get('declined')}, reasons {g.get('decline_reasons')})")
            else:
                notes.append(f"{tag}: routed engaged {g.get('returned')}/{g.get('calls')} generate-time MoE calls "
                             f"(nf4 {g.get('nf4_routed')}, bf16 {g.get('bf16_moe')}, declined {g.get('declined')} {g.get('decline_reasons')})")
            if g.get("compiling_calls"):
                notes.append(f"{tag}: {g['compiling_calls']} census hits were under torch.compile tracing (counts approximate)")
        for x in h.get("generates") or []:
            if x.get("error"):
                problems.append(f"{tag}: head generate in cell {x.get('cell')} raised {x['error']}")
        bg, hg = b.get("generates") or [], h.get("generates") or []
        for i, (x, y) in enumerate(zip(bg, hg)):
            if x.get("tok_per_s") and y.get("tok_per_s"):
                notes.append(f"{tag} gen#{i + 1} (cell {y.get('cell')}): {x['tok_per_s']} -> {y['tok_per_s']} tok/s "
                             f"({y['tok_per_s'] / x['tok_per_s']:.2f}x)")
        bs, hs = b.get("steps") or [], h.get("steps") or []
        if len(bs) > 1 and len(hs) > 1:
            mb = sorted(bs[1:])[len(bs[1:]) // 2]
            mh = sorted(hs[1:])[len(hs[1:]) // 2]
            notes.append(f"{tag} median step (excl. first): {mb} s -> {mh} s ({mh / mb:.2f}x)")
    if problems:
        return True, "; ".join(problems) + ". Also: " + "; ".join(notes)
    return False, "; ".join(notes)
