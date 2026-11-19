#!/usr/bin/env python3
"""Criteria (unsloth #18462): block-compile mem-efficient SDPA backward alignment guard, CUDA-only change.
REGRESSION mode on gfx1151 (ROCm: the PR's gate is off, so base and head should behave identically).
head_is_worse: a test id that passes / skips at base fails at head (tests/test_block_compile.py), any failure in the PR's
new test file at head, or a tiny Gemma-4 padded LoRA step (S = 37, 38, 36, 40) whose outcome differs from base
(pass vs fail), or whose step losses / grad norms differ from base (rel > 1e-4).
Gates: both states imported their own checkout on a gfx1151 HIP device; head has the gate and it reads False (ROCm);
the tiny model trained at least at one length on base (else the cell measured nothing).
"""
from __future__ import annotations

TITLE = "unsloth #18462: block compile efficient-backward alignment guard (CUDA-only), base vs head on gfx1151"
MODE = "regression"
NEEDS = ["rocm", "gpu", "nvidia", "windows"]
SL = ("37", "38", "36", "40")
NEW = "tests/test_block_sdpa_efficient_backward_guard.py"
OLD = "tests/test_block_compile.py"


def _own(o):
    f = str(((o or {}).get("info") or {}).get("bc_file") or "").replace("\\", "/")
    ck = str((o or {}).get("checkout") or "").replace("\\", "/")
    return bool(f) and bool(ck) and f.startswith(ck)


def gates(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    out = []
    for name, o in (("base", b), ("head", h)):
        i = o.get("info") or {}
        out.append((f"{name}: gfx1151 HIP device", bool(i.get("hip")) and "gfx1151" in str(i.get("arch")), str(i)[:300]))
        out.append((f"{name}: _block_compile imported from its own checkout", _own(o), str(i.get("bc_file"))))
    hi, bi = h.get("info") or {}, b.get("info") or {}
    out.append(("head has the gate (reads False on ROCm), base does not",
                hi.get("has_gate") is True and hi.get("gate") is False and bi.get("has_gate") is False,
                f"head has={hi.get('has_gate')} gate={hi.get('gate')}; base has={bi.get('has_gate')}"))
    trained = [s for s in SL if ((b.get("train") or {}).get(s) or {}).get("result") == "PASS"]
    out.append(("base trained the tiny Gemma-4 at >= 1 length", bool(trained), f"base PASS at {trained}"))
    t = ((h.get("tests") or {}).get(OLD) or {}).get("cases") or {}
    out.append((f"{OLD} collected at head", bool(t), f"{len(t)} cases"))
    return out


def _failed(o, tf):
    c = (((o or {}).get("tests") or {}).get(tf) or {}).get("cases") or {}
    return {k for k, v in c.items() if v == "failed"}


def head_is_worse(base, head):
    bad = []
    new = sorted(_failed(head, OLD) - _failed(base, OLD))
    if new:
        bad.append(f"{OLD}: newly failing at head {new}")
    t = (head.get("tests") or {}).get(NEW) or {}
    if not t.get("cases") or _failed(head, NEW):
        bad.append(f"{NEW} at head: rc={t.get('rc')} n={len(t.get('cases') or {})} failed={sorted(_failed(head, NEW))}")
    for s in SL:
        cb = (base.get("train") or {}).get(s) or {}
        ch = (head.get("train") or {}).get(s) or {}
        if (cb.get("result") == "PASS") != (ch.get("result") == "PASS"):
            bad.append(f"S={s}: base {cb.get('result')} head {ch.get('result')}")
            continue
        sb, sh = cb.get("steps") or {}, ch.get("steps") or {}
        if set(sb) != set(sh):
            bad.append(f"S={s}: steps base {sorted(sb)} head {sorted(sh)}")
            continue
        for k in sb:
            for m in ("loss", "grad_norm"):
                x, y = float(sb[k][m]), float(sh[k][m])
                if abs(x - y) > 1e-4 * max(abs(x), 1e-6):
                    bad.append(f"S={s} step {k} {m}: base {x} head {y}")
    return (bool(bad), "; ".join(bad) or "no test fails at head that passed at base; the PR's tests pass; tiny "
                                      "Gemma-4 padded LoRA steps identical to base at S = 37, 38, 36, 40")


def table(obs):
    b, h = obs.get("base") or {}, obs.get("head") or {}
    L = ["| S (padded) | base result | head result | base step losses / grad norms | head step losses / grad norms | block compiled (head) |",
         "|---|---|---|---|---|---|"]
    fmt = lambda st: ", ".join(f"{v['loss']:.6f}/{v['grad_norm']:.6f}" for _, v in sorted((st or {}).items(), key = lambda kv: int(kv[0])))
    for s in SL:
        cb = (b.get("train") or {}).get(s) or {}
        ch = (h.get("train") or {}).get(s) or {}
        L.append(f"| {s} | {cb.get('result')} | {ch.get('result')} | {fmt(cb.get('steps'))} | {fmt(ch.get('steps'))} | "
                 f"{ch.get('block_compiled')} |")
    L += ["", "| state | test file | rc | passed | failed | skipped | failing ids |", "|---|---|---|---|---|---|---|"]
    for name, o in (("base", b), ("head", h)):
        for tf, t in (o.get("tests") or {}).items():
            if t.get("absent"):
                L.append(f"| {name} | {tf} | absent | | | | |")
                continue
            c = t.get("cases") or {}
            n = {k: sum(1 for v in c.values() if v == k) for k in ("passed", "failed", "skipped")}
            L.append(f"| {name} | {tf} | {t.get('rc')} | {n['passed']} | {n['failed']} | {n['skipped']} | "
                     f"{sorted(k.split('::')[-1] for k, v in c.items() if v == 'failed')} |")
    L.append(f"\nhead info {h.get('info')}; base commit {b.get('commit')}; head commit {h.get('commit')}")
    L.append(f"pip {h.get('pip')}; gemma4 transformers {h.get('gemma4_transformers')}; build {str(h.get('build') or b.get('build'))[:300]}")
    return "\n".join(L)
