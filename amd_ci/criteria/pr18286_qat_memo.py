#!/usr/bin/env python3
"""Criteria (unsloth #18286, head = 2727a180e6, merge base 40cbeb7c5f + PR commits): LoRA + QAT
fake quantizes each frozen base weight once (unsloth/models/_qat_weight_memo.py).

Differential. The base shows the "defect" (no memo) when tests/test_qat_weight_memo.py (the head's file, copied in)
fails or errors at the base. The head is fixed when:
  - every test in the file passes at the head (none failed / errored / skipped; a skip is NOT RUN, never a pass)
  - the head's default cell trains with finite losses wherever the base's did, and its step 1-2 losses equal the
    base's within fp tolerance; the eval forward is finite
  - the memo is installed and armed after training (qat_weight_memo_stats armed True, installed > 0), and the eval
    forward dropped the cache (eval_drops grows) when anything was cached
  - the kill switch (UNSLOTH_QAT_WEIGHT_MEMO=0) cell trains finite, stores nothing, and matches the base losses at
    every logged step within the same tolerance
Gates: each state's tree is exactly 40cbeb7c5f / be911c33f7; torchao QAT + peft importable in the venv (else the
tests skip and nothing is measured); pytest ran and collected; the base cell trained (a comparison needs it).
"""
from __future__ import annotations

import math

TITLE = "unsloth #18286 (head 2727a180e6): QAT LoRA fake-quantized weight memo, base vs head"
MODE = "differential"
NEEDS = ["rocm", "gpu", "windows", "nvidia", "exclusive_gpu"]
TREES = {"base": "beb44b744795956cb68e1797290392a003238d6d", "head": "597c5a8d7cf4ddf10961dfe10aadfd361d3760fc"}
RTOL, ATOL = 1e-3, 1e-4


def _cases(o):
    return ((o or {}).get("test") or {}).get("cases") or {}


def _cell(o, k = "default"):
    return ((o or {}).get("cells") or {}).get(k) or {}


def _imports(o):
    d = (o or {}).get("deps") or {}
    return d.get("after") or d.get("before") or {}


def _close(x, y):
    return x is not None and y is not None and math.isfinite(x) and math.isfinite(y) and abs(x - y) <= ATOL + RTOL * abs(y)


def _losses(c):
    return {int(k): v for k, v in (c.get("losses") or {}).items()}


def gates(obs):
    out = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        tree = (o.get("pin") or {}).get("tree")
        out.append((f"{name} tree is {TREES[name][:10]}", tree == TREES[name], f"{tree} {str((o.get('pin') or {}).get('steps'))[:300]}"))
        im = _imports(o)
        out.append((f"{name} venv has torchao QAT + peft", bool(im.get("torchao_qat")) and bool(im.get("peft")),
                    f"torch {im.get('torch')} hip {im.get('hip')} torchao {im.get('torchao')} peft {im.get('peft')} "
                    f"zoo {im.get('unsloth_zoo')} {str(im.get('torchao_qat_error') or im.get('error') or '')[:200]}"))
        t = o.get("test") or {}
        out.append((f"{name} pytest ran and collected", t.get("rc") in (0, 1, 2) and len(_cases(o)) > 0,
                    f"rc={t.get('rc')} cases={len(_cases(o))} {'' if _cases(o) else str(t.get('tail'))[-300:]}"))
        c = _cell(o)
        uf, ck = str(c.get("unsloth_file") or "").replace("\\", "/").lower(), str(o.get("checkout") or "").replace("\\", "/").lower()
        out.append((f"{name} cell imported its own unsloth", bool(uf) and bool(ck) and uf.startswith(ck), uf or str(c.get("error"))[:300]))
    b = _cell(obs.get("base"))
    out.append(("base cell trained (reference losses exist)", bool(_losses(b)),
                f"stage={b.get('stage')} {str(b.get('error') or '')[:300]}"))
    return out


def base_shows_defect(base):
    return any(c.get("outcome") in ("failed", "error") for c in _cases(base).values())


def problems(head):
    bad = []
    cs = _cases(head)
    # The one skip the test itself declares: int8-int4 when the unwrapped quantizer also fails fullgraph compile
    # (torch 2.11 dynamo + torch.ao). Any other skip is NOT RUN; the weight-only int4 param must run and pass.
    allowed = {"test_compile_trace_unchanged[int8-int4]"}
    for t, c in cs.items():
        short = t.split("::")[-1]
        if c.get("outcome") == "skipped" and short in allowed:
            continue
        if c.get("outcome") != "passed":
            bad.append(f"test {short} {c.get('outcome')}")
    if not any(t.endswith("test_compile_trace_unchanged[int4]") and c.get("outcome") == "passed" for t, c in cs.items()):
        bad.append("test_compile_trace_unchanged[int4] did not run and pass")
    if not cs:
        bad.append("no head tests")
    ref = _losses({"losses": (head.get("base_ref") or {}).get("losses")})
    d = _cell(head)
    hl = _losses(d)
    if not d.get("finite_losses"):
        bad.append(f"default cell: losses not finite / missing (stage {d.get('stage')}, {str(d.get('error'))[:200]})")
    if not ref:
        bad.append("no base reference losses in the head observation")
    for s in (1, 2):
        if not _close(hl.get(s), ref.get(s)):
            bad.append(f"default step {s}: head {hl.get(s)} vs base {ref.get(s)}")
    if d.get("eval_logits_finite") is not True:
        bad.append("default eval forward not finite")
    st, ev = d.get("memo_after_train") or {}, d.get("memo_after_eval") or {}
    if st.get("armed") is not True or not st.get("installed"):
        bad.append(f"memo not installed+armed after training: {str(st)[:200]}")
    # eval_drops counts only when something was cached (a fully compiled run caches nothing: reported, not judged).
    if (st.get("cached_bytes") or 0) > 0 and not (ev.get("eval_drops", 0) > st.get("eval_drops", 0)):
        bad.append(f"eval did not drop the cache: eval_drops {st.get('eval_drops')} -> {ev.get('eval_drops')}")
    off = _cell(head, "memo_off")
    ol = _losses(off)
    if not off.get("finite_losses"):
        bad.append(f"memo_off cell: losses not finite / missing ({str(off.get('error'))[:200]})")
    if (off.get("memo_after_train") or {}).get("stores"):
        bad.append(f"memo_off stored {off['memo_after_train']['stores']} weights")
    for s in sorted(ref):
        if not _close(ol.get(s), ref.get(s)):
            bad.append(f"memo_off step {s}: {ol.get(s)} vs base {ref.get(s)}")
    return bad


def head_is_fixed(head):
    return not problems(head)


def table(obs):
    lines = []
    for name in ("base", "head"):
        o = obs.get(name) or {}
        cs = _cases(o)
        lines.append(f"- {name} tests (tree `{str((o.get('pin') or {}).get('tree'))[:10]}`): "
                     + ", ".join(f"`{t.split('::')[-1]}` {c.get('outcome')}" for t, c in sorted(cs.items())))
        for t, c in sorted(cs.items()):
            if c.get("outcome") in ("failed", "error", "skipped"):
                lines.append(f"  - {c.get('outcome')} `{t}`: {' '.join(c.get('message', '').split())[:240]}")
    lines += ["", "| state | cell | versions | fq linears | losses | finite | eval finite | memo after train | memo after eval | error |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, cells in (("base", ("default",)), ("head", ("default", "memo_off"))):
        for k in cells:
            c = _cell(obs.get(name), k)
            v = c.get("versions") or {}
            ls = ", ".join(f"{s}:{x:.5f}" for s, x in sorted(_losses(c).items()))
            mem = lambda m: {x: m.get(x) for x in ("installed", "armed", "stores", "hits", "evictions", "eval_drops", "over_budget", "oom_disabled", "cached_bytes")} if isinstance(m, dict) and "absent" not in m else "absent"  # noqa: E731
            lines.append(f"| {name} | {k} | torch {v.get('torch')} torchao {v.get('torchao')} peft {v.get('peft')} | "
                         f"{c.get('fake_quantized_linears')} | {ls} | {c.get('finite_losses')} | {c.get('eval_logits_finite')} | "
                         f"{mem(c.get('memo_after_train'))} | {mem(c.get('memo_after_eval'))} | "
                         f"{' '.join(str(c.get('error') or '').split())[:200]} |")
    lines += ["", "test_compile_trace_unchanged control (same fullgraph trace, int8-int4 QAT LoRA block):", "",
              "| state | memo | memo module | traced | frame_count | equal | error |", "|---|---|---|---|---|---|---|"]
    for name in ("base", "head"):
        for tag, r in sorted(((obs.get(name) or {}).get("compile_ctl") or {}).items()):
            lines.append(f"| {name} | {tag} | {r.get('memo_module')} | {r.get('traced')} | {r.get('frame_count')} | {r.get('equal')} | "
                         f"{' '.join(str(r.get('trace_error') or r.get('error') or '').split())[:260]} |")
    h = obs.get("head") or {}
    if h:
        p = problems(h)
        lines += ["", "Head problems: " + ("; ".join(p) if p else "none")]
    return "\n".join(lines)
